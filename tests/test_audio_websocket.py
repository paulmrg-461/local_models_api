import pytest
from fastapi.testclient import TestClient

from app.api.audio_ws import get_analyze_audio_use_case, MAX_PRECONFIG_BUFFER, MAX_SESSION_BUFFER
from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import (
    ActionItem,
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
    TranscriptSegment,
)
from app.main import app


class FakeASR(ASRGateway):
    # record each call for assertions
    calls = []

    def transcribe(self, audio_bytes: bytes, language: str) -> list[TranscriptSegment]:
        # store the length of bytes received so tests can inspect
        FakeASR.calls.append(len(audio_bytes))
        return [
            TranscriptSegment(
                speaker="S1",
                start=0.0,
                end=1.0,
                text="hello",
            )
        ]


class FakeConversationAnalyzer(ConversationAnalysisGateway):
    def analyze(
        self,
        session_id: str,
        language: str,
        segments: list[TranscriptSegment],
    ) -> AudioSessionAnalysis:
        return AudioSessionAnalysis(
            session_id=session_id,
            language=language,
            transcript=segments,
            summary="summary",
            action_items=[
                ActionItem(
                    title="title",
                    description="description",
                    steps=["one"],
                )
            ],
            risks=["risk"],
        )


def override_use_case() -> AnalyzeAudioSessionUseCase:
    asr = FakeASR()
    analyzer = FakeConversationAnalyzer()
    return AnalyzeAudioSessionUseCase(asr, analyzer)


app.dependency_overrides[get_analyze_audio_use_case] = override_use_case


def test_audio_websocket_final_result_message():
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as websocket:
        # the server now sends a "ready" greeting immediately after accept
        greeting = websocket.receive_json()
        assert greeting["type"] == "ready"

        websocket.send_json(
            {
                "type": "config",
                "session_id": "abc",
                "sample_rate": 16000,
                "encoding": "pcm16",
                "language": "es",
            }
        )
        websocket.send_bytes(b"\x00\x01")
        websocket.send_json({"type": "end_of_stream"})

        message = websocket.receive_json()

        assert message["type"] == "final_result"
        analysis = message["analysis"]
        assert analysis["session_id"] == "abc"
        assert analysis["language"] == "es"
        assert analysis["summary"] == "summary"
        assert len(analysis["transcript"]) == 1
        assert analysis["transcript"][0]["text"] == "hello"
        assert len(analysis["action_items"]) == 1
        assert len(analysis["risks"]) == 1


def test_audio_websocket_client_disconnect():
    """Ensure the server handles a client that drops mid‑session gracefully.

    When the connection closes the handler now treats it as an implicit
    ``end_of_stream`` and runs the analysis logic (although no result is
    returned because the socket is gone). We verify that the ASR gateway
    was invoked with the buffered bytes.
    """
    FakeASR.calls.clear()
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as websocket:
        websocket.send_json({"type": "config", "session_id": "xyz"})
        websocket.send_bytes(b"abc123")
        websocket.close()  # simulate abrupt client closure

    # should have invoked ASR once with the buffered audio length. the
    # gateway receives whatever payload we constructed (wav header + raw
    # samples), so we simply assert that it was called and the value is
    # non‑zero rather than hard‑coding a byte count.
    assert len(FakeASR.calls) == 1
    assert FakeASR.calls[0] > 0


def test_bytes_before_config_buffered():
    """If bytes arrive before the config message we buffer them and use them after config."""
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as websocket:
        # the server greeting
        assert websocket.receive_json()["type"] == "ready"

        # send a small chunk before config
        websocket.send_bytes(b"preconfig")
        websocket.send_json({
            "type": "config",
            "session_id": "buff",
            "sample_rate": 16000,
            "encoding": "pcm16",
            "language": "es",
        })
        websocket.send_bytes(b"postconfig")
        websocket.send_json({"type": "end_of_stream"})

        msg = websocket.receive_json()
        assert msg["type"] == "final_result"
        # the fake ASR doesn't inspect bytes but we at least ensure there's no error


def test_end_of_stream_without_config_uses_buffer():
    """Send bytes first, no config, then end_of_stream; server should still reply."""
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as websocket:
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(b"onlythis")
        websocket.send_json({"type": "end_of_stream"})
        msg = websocket.receive_json()
        assert msg["type"] == "final_result"


def test_pre_config_buffer_limit():
    """Ensure the pre‑config buffer never grows past the cap.

    Sending more than ``MAX_PRECONFIG_BUFFER`` bytes before a config message
    should not crash the server; the excess is dropped and the final result is
    still delivered. The fake ASR gateway records how many bytes were
    actually passed to it so we assert that the call length is smaller than
    what we sent.
    """
    client = TestClient(app)
    big = b"A" * (MAX_PRECONFIG_BUFFER + 1024)

    with client.websocket_connect("/ws/audio") as websocket:
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(big)
        websocket.send_json({
            "type": "config",
            "session_id": "big",
            "sample_rate": 16000,
            "encoding": "pcm16",
            "language": "es",
        })
        websocket.send_json({"type": "end_of_stream"})
        msg = websocket.receive_json()
        assert msg["type"] == "final_result"

    # the ASR gateway should have been called, and the argument length must be
    # less than the original big blob because some data were discarded.
    assert FakeASR.calls
    assert FakeASR.calls[0] < len(big)


def test_session_buffer_limit_closes():
    """If a configured session receives too much audio we close the socket.

    We monkey‑patch the module constant to a very small value so that the
    overflow happens quickly and deterministically.
    """
    client = TestClient(app)
    # temporarily shrink the limit to provoke the error
    import app.api.audio_ws as aws
    aws.MAX_SESSION_BUFFER = 10

    with client.websocket_connect("/ws/audio") as websocket:
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "config", "session_id": "tiny"})
        # send more than the tiny limit
        websocket.send_bytes(b"X" * 20)
        # the server should have closed the connection; any receive will raise
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


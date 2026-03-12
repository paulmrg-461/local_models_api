import os
import logging
import tempfile
import re
import io
from typing import List, Optional
from funasr import AutoModel
from pydub import AudioSegment
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Interfaces de tu app
from app.domain.audio.interfaces import ASRGateway, TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"
DEFAULT_DEVICE = "cuda:0"

class SenseVoiceASRGateway(ASRGateway):
    def __init__(self) -> None:
        model_id = os.getenv("SENSEVOICE_MODEL_ID", DEFAULT_SENSEVOICE_MODEL)
        device = os.getenv("SENSEVOICE_DEVICE", DEFAULT_DEVICE)
        
        try:
            logger.info("Cargando SenseVoiceSmall en modo ULTRA-ESTRICTO (hf, es)...")
            # Forzamos vad_model=None para evitar alucinaciones chinas del VAD
            self._model = AutoModel(
                model=model_id,
                trust_remote_code=True,
                device=device,
                hub="hf",
                disable_update=True
            )
            logger.info("SenseVoice cargado exitosamente.")
            
        except Exception as e:
            logger.error(f"Fallo al cargar SenseVoice: {e}")
            raise e

    def _prepare_audio(self, audio_bytes: bytes) -> str:
        """Convierte y normaliza a WAV 16kHz Mono para SenseVoice."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Forzamos 16kHz Mono
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Normalización agresiva y un poco de boost
            audio = audio.normalize()
            audio = audio + 6 # +6dB extra
            
            # Agregamos 0.5s de silencio al inicio y final para ayudar al modelo
            silence = AudioSegment.silent(duration=500, frame_rate=16000)
            audio = silence + audio + silence
            
            # Guardamos a un temporal .wav
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            audio.export(tmp_path, format="wav")
            return tmp_path
        except Exception as e:
            logger.error("Error pydub en preprocesamiento: %s", e)
            # Fallback simple
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)
            return tmp_path

    def _clean_text(self, text: str) -> str:
        """
        Limpia etiquetas de SenseVoice pero DEJA pasar todo lo demás para depuración.
        """
        if not text:
            return ""

        # 1. Quitar tags de SenseVoice <|speech|>, <|applause|>, etc.
        text = re.sub(r"<\|.*?\|>", "", text)
        
        # 2. Quitar espacios dobles
        text = re.sub(r'\s+', ' ', text).strip()
            
        return text

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str,
        filename: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        if not audio_bytes:
            return []

        tmp_path = None
        try:
            tmp_path = self._prepare_audio(audio_bytes)
            
            # USAMOS 'auto' porque 'es' no es soportado oficialmente por SenseVoiceSmall
            target_lang = "auto"
            
            logger.info(f"Inferencia SenseVoice (usando {target_lang})...")
            
            # Inferencia con parámetros optimizados para local
            res = self._model.generate(
                input=tmp_path,
                cache={},
                language=target_lang,
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                merge_length_s=15,
            )

            if not res or not isinstance(res, list) or len(res) == 0:
                logger.warning("SenseVoice no devolvió resultados.")
                return []

            # Con merge_vad=False, res es una lista de segmentos. Debemos iterar.
            all_clean_texts = []
            logger.info(f"SenseVoice devolvió {len(res)} segmentos para procesar.")
            logger.info(f"RAW RES: {res}")

            for segment in res:
                raw_text = segment.get("text", "")
                if not raw_text:
                    continue
                
                logger.debug(f"SENSEVOICE RAW SEGMENT: [{raw_text}]")
                clean_segment = self._clean_text(raw_text)
                
                if clean_segment:
                    all_clean_texts.append(clean_segment)

            if not all_clean_texts:
                logger.warning("Todos los segmentos fueron filtrados como ruido.")
                return []
            
            # Unimos los segmentos limpios para formar la transcripción final
            final_text = " ".join(all_clean_texts)
            logger.info("SENSEVOICE CLEAN (unido): [%s]", final_text)

            return [
                TranscriptSegment(
                    speaker="Speaker",
                    start=0.0,
                    end=0.0,
                    text=final_text
                )
            ]

        except Exception as e:
            logger.exception("Error en SenseVoiceASRGateway.transcribe: %s", e)
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning("No se pudo borrar temporal: %s", e)

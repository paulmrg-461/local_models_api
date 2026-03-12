# Guía de Integración Real-Time: Flutter + WebSocket

Esta guía explica cómo enviar audio en tiempo real desde una aplicación Flutter hacia el servidor de transcripción local usando WebSockets.

## 1. Configuración del Cliente WebSocket en Flutter

Usa el paquete `web_socket_channel` para manejar la conexión.

```dart
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;
import 'dart:convert';
import 'dart:typed_data';

class TranscriptionService {
  late WebSocketChannel _channel;
  final String _url = 'ws://TU_IP_LOCAL:8989/ws/audio';

  void connect(String sessionId) {
    _channel = WebSocketChannel.connect(Uri.parse(_url));

    // 1. Escuchar mensajes del servidor
    _channel.stream.listen((message) {
      final data = jsonDecode(message);
      if (data['type'] == 'ready') {
        print('Servidor listo para recibir audio');
        // 2. Enviar configuración inicial
        _sendConfig(sessionId);
      } else if (data['type'] == 'final_result') {
        print('Transcripción final: ${data['analysis']['transcription']}');
      }
    });
  }

  void _sendConfig(String sessionId) {
    final config = {
      'type': 'config',
      'session_id': sessionId,
      'sample_rate': 16000,
      'encoding': 'pcm16',
      'language': 'es',
    };
    _channel.sink.add(jsonEncode(config));
  }

  // 3. Enviar chunks de audio (Uint8List)
  void sendAudioChunk(Uint8List chunk) {
    _channel.sink.add(chunk);
  }

  // 4. Finalizar la grabación
  void stop() {
    _channel.sink.add(jsonEncode({'type': 'end_of_stream'}));
    _channel.sink.close(status.goingAway);
  }
}
```

## 2. Flujo de Comunicación

El protocolo WebSocket diseñado sigue estos pasos:

1.  **Conexión**: El cliente abre la conexión `ws://host:8989/ws/audio`.
2.  **Ready**: El servidor responde con `{"type": "ready"}`.
3.  **Config**: El cliente envía un JSON con la configuración (sample rate, language, etc.).
4.  **Binary Data**: El cliente envía los bytes crudos del audio (PCM 16-bit, 16kHz, Mono).
5.  **End of Stream**: El cliente envía `{"type": "end_of_stream"}` cuando termina de hablar.
6.  **Result**: El servidor procesa todo el audio acumulado y responde con el JSON de la transcripción completa.

## 3. Grabación de Audio en Flutter

Para obtener audio compatible (PCM 16-bit, 16kHz), se recomienda usar el paquete `record`:

```dart
import 'package:record/record.dart';

final record = AudioRecorder();

// Configurar para streaming
final config = RecordConfig(
  encoder: AudioEncoder.pcm16bits,
  sampleRate: 16000,
  numChannels: 1,
);

// Iniciar stream
final stream = await record.startStream(config);

stream.listen((data) {
  // Enviar cada pedazo de audio al WebSocket
  transcriptionService.sendAudioChunk(Uint8List.fromList(data));
});
```

## 4. Registro en el Servidor (Logs y Archivos)

El servidor está configurado para:
1.  Mostrar en la **consola** en tiempo real cuando se recibe audio y cuando se termina de procesar.
2.  Guardar una copia de la transcripción completa en la carpeta `transcriptions/` dentro del servidor.
3.  Cada archivo se guarda con el formato: `transcript_YYYYMMDD_HHMMSS_sessionID.txt`.

---

**Nota Importante**: Asegúrate de que el servidor y el dispositivo Flutter estén en la misma red Wi-Fi y usa la IP privada de tu PC (ej: `192.168.1.15`) en lugar de `127.0.0.1`.

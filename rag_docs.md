### **Guía de Integración para el Servicio de Memorias (RAG)**

He actualizado el backend para que puedas gestionar el historial de memorias y realizar búsquedas semánticas. A continuación, tienes la guía técnica para integrar tu otro servicio (Asistente de Voz) con esta API.

---

### **1. Inserción de Memorias (POST)**
Tu servicio externo debe enviar la interpretación procesada por Whisper y tu modelo VLM a este endpoint. El sistema generará automáticamente los **embeddings vectoriales** para permitir búsquedas futuras.

- **Endpoint:** `POST /user/memories/`
- **Body (JSON):**

```json
{
  "transcript_original": "Hola Omi, ¿puedes escucharme? Estoy corriendo los modelos locales...",
  "interpretation": {
    "summary": "S1 realiza tareas de procesamiento de imágenes y texto utilizando modelos locales en su tarjeta gráfica Nvidia RTX 5070...",
    "action_items": [
      {
        "title": "Actualizar modelos",
        "description": "Actualizar los modelos Whisper y QWEN para mejorar la precisión.",
        "steps": ["Instalar actualizaciones", "Realizar pruebas"]
      }
    ],
    "risks": ["La falta de actualización puede afectar la precisión"]
  }
}
```

---

### **2. Consulta de Historial (GET)**
Para responder a la pregunta **"¿Puedo ver el historial de memories?"**, he actualizado el endpoint principal para que devuelva los registros ordenados por fecha de creación (los más recientes primero).

- **Endpoint:** `GET /user/memories/`
- **Parámetros opcionales:**
    - `limit`: Cantidad de memorias a devolver (por defecto 50).
    - `offset`: Para paginación.
- **Orden:** Descendente por `created_at`.

**Ejemplo de consulta (cURL):**
```bash
curl "http://localhost:8787/user/memories/?limit=10"
```

---

### **3. Búsqueda Semántica / RAG (GET)**
Este es el endpoint que debes usar para responder preguntas específicas sobre el pasado del usuario. No busca por palabras clave exactas, sino por **significado**.

- **Endpoint:** `GET /user/memories/search`
- **Parámetros:**
    - `query`: La pregunta del usuario (ej: "¿Qué hice el lunes?").
    - `limit`: Número de recuerdos relevantes a recuperar (por defecto 5).

**Ejemplo de flujo para tu Chatbot:**
1. El usuario pregunta: *"¿Qué tenía que hacer con la Jetson?"*
2. Tu chatbot llama a: `GET /user/memories/search?query=que+hacer+con+la+jetson`
3. La API devuelve los recuerdos más cercanos semánticamente.
4. Tu chatbot usa esos recuerdos como contexto para generar la respuesta final.

---

### **Detalles Técnicos Importantes**
- **Base de Datos:** Estamos usando **pgvector** en Postgres. Esto permite que las búsquedas sean extremadamente rápidas incluso con miles de recuerdos.
- **Embeddings:** Se utiliza el modelo `text-embedding-004` de Gemini (768 dimensiones).
- **Campos adicionales:** En la respuesta de la API, verás el campo `structured_data` que contiene los `action_items` y `risks` de forma organizada para que tu UI pueda mostrarlos fácilmente.
from fastapi import FastAPI

from app.api.audio_ws import router as audio_router
from app.api.vision_routes import router as vision_router


app = FastAPI()

app.include_router(vision_router)
app.include_router(audio_router)

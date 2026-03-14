from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

from app.api.audio_ws import router as audio_router
from app.api.vision_routes import router as vision_router
from app.api.audio_routes import router as audio_file_router


app = FastAPI()

app.include_router(vision_router)
app.include_router(audio_router)
app.include_router(audio_file_router)

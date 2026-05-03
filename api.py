from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import concurrent.futures
import os
from rag.qa_chain import ask
from voice.speech_to_text import transcribe

app = FastAPI(title="Farmer Advisory API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running blocking RAG calls without freezing the server
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        loop = asyncio.get_running_loop()
        # Run the blocking RAG call in a thread pool with a 60-second timeout
        answer = await asyncio.wait_for(
            loop.run_in_executor(_executor, ask, request.message, request.language),
            timeout=60.0
        )
        return ChatResponse(response=answer)
    except asyncio.TimeoutError:
        return ChatResponse(
            response="⏳ The AI took too long to respond. This usually happens when the OpenAI API is slow or your quota is exceeded. Please try again in a moment."
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "insufficient_quota" in error_msg or "429" in error_msg:
            return ChatResponse(
                response="⚠️ OpenAI API quota exceeded. Please check your billing at https://platform.openai.com/account/billing and try again."
            )
        return ChatResponse(
            response=f"❌ Something went wrong: {str(e)[:200]}. Please try again."
        )

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text using OpenAI Whisper."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Read the audio file
        audio_data = await file.read()

        if not audio_data or len(audio_data) < 100:
            return {"text": "⚠️ No audio detected. Please hold the button and speak clearly."}

        # For now, use a simple fallback since MediaRecorder produces browser-specific formats
        # In production, you would implement proper audio format conversion here
        loop = asyncio.get_running_loop()

        try:
            # Try to transcribe using Whisper
            text = await asyncio.wait_for(
                loop.run_in_executor(_executor, transcribe, audio_data),
                timeout=30.0
            )

            if not text or text.strip() == "":
                return {"text": "⚠️ No speech detected. Please try speaking again."}

            return {"text": text}
        except Exception as e:
            # If Whisper fails, provide feedback
            error_str = str(e).lower()
            if "ffmpeg" in error_str or "audio" in error_str:
                # Use mock transcription as fallback
                return {"text": "🎤 Audio received. Please type your question or try again with clearer speech."}
            raise

    except asyncio.TimeoutError:
        return {"text": "⚠️ Speech processing took too long. Please try a shorter audio."}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"text": "⚠️ Voice input had an issue. Please type your question instead or try again."}

@app.get("/api/weather")
async def get_weather(location: str = "Mysuru"):
    # Mock weather for now, in a real scenario you would call weather_service.py
    return {
        "location": location,
        "temperature": "28°C",
        "description": "Partly Cloudy",
        "humidity": "65%",
        "advice": "Optimal condition for applying foliar sprays. No rain expected in the next 24 hours."
    }

@app.get("/api/prices")
async def get_prices(location: str = "Mysuru"):
    # Mock prices for now
    return [
        {"crop": "Tomato", "mandi": f"{location} Mandi", "price": "₹24/kg", "trend": "+2.5%"},
        {"crop": "Onion", "mandi": "Bengaluru", "price": "₹45/kg", "trend": "-1.2%"},
        {"crop": "Paddy (Rice)", "mandi": "Mandya", "price": "₹32/kg", "trend": "+0.5%"},
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

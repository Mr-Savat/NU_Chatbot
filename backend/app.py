# backend/app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faq_loader import find_best_faq
from gpt2_model import ask_gpt2
from gemini_fallback import ask_ai_async  # Now only uses Groq
from confidence import compute_confidence
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production: specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()

    if not question:
        return JSONResponse({"answer": "⚠️ Please enter a question.", "source": "System"})

    # Step 1: FAQ Lookup
    faq_answer, faq_conf = find_best_faq(question)
    if faq_conf >= 0.7:
        return JSONResponse({"answer": faq_answer, "source": f"FAQ (confidence: {faq_conf:.2f})"})

    # Step 2: GPT-2 (sync call)
    try:
        gpt2_answer = ask_gpt2(question)
        confidence = compute_confidence(question, gpt2_answer)
        if confidence >= 0.7:
            return JSONResponse({"answer": gpt2_answer, "source": f"GPT-2 (confidence: {confidence:.2f})"})
    except Exception as e:
        gpt2_answer = f"⚠️ GPT-2 error: {str(e)}"
        confidence = 0.0

    # Step 3: Groq AI fallback
    ai_answer = await ask_ai_async(question)

    return JSONResponse({
        "answer": ai_answer,
        "source": f"Groq AI (GPT-2 confidence: {confidence:.2f})"
    })


# Serve frontend files
@app.get("/{path:path}")
async def serve_frontend(path: str = ""):
    frontend_path = "../frontend"
    file_path = os.path.join(frontend_path, path)

    if path and os.path.exists(file_path):
        return FileResponse(file_path)

    return FileResponse(os.path.join(frontend_path, "index.html"))
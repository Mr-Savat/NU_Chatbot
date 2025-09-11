# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faq_loader import find_best_faq  # updated import
from gpt2_model import ask_gpt2
from gemini_fallback import ask_gemini
from confidence import compute_confidence  # renamed if needed
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder="../frontend")
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "⚠️ Please enter a question.", "source": "System"})

    # Step 1: FAQ Lookup
    faq_answer, faq_conf = find_best_faq(question)
    if faq_conf >= 0.7:  # threshold for FAQ match
        return jsonify({"answer": faq_answer, "source": f"FAQ (confidence: {faq_conf:.2f})"})

    # Step 2: GPT-2
    try:
        gpt2_answer = ask_gpt2(question)
        confidence = compute_confidence(question, gpt2_answer)
        if confidence >= 0.7:
            return jsonify({"answer": gpt2_answer, "source": f"GPT-2 (confidence: {confidence:.2f})"})
    except Exception as e:
        gpt2_answer = f"⚠️ GPT-2 error: {str(e)}"
        confidence = 0.0

    # Step 3: Gemini fallback (safe even if no API key)
    gemini_answer = ask_gemini(question)
    return jsonify({
        "answer": gemini_answer,
        "source": f"Gemini API (GPT-2 confidence: {confidence:.2f})"
    })


# Serve frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    frontend_path = "../frontend"
    if path and os.path.exists(os.path.join(frontend_path, path)):
        return send_from_directory(frontend_path, path)
    return send_from_directory(frontend_path, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port)

import google.generativeai as genai
import os

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ GEMINI_API_KEY not found. Gemini functionality is disabled.")
# Configure only if the key exists to avoid library errors
if api_key:
    genai.configure(api_key=api_key)

# Function to ask Gemini model
def ask_gemini(question: str) -> str:
    # Check for API key first
    if not os.getenv("GEMINI_API_KEY"):
        return "⚠️ Gemini fallback disabled. No API key provided. Please add your GEMINI_API_KEY to the .env file."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Generate response
        response = model.generate_content(question)
        # Check if Gemini returned something
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "⚠️ Sorry, Gemini did not return a valid answer."
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"
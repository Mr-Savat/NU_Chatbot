# test_model.py
from gpt2_model import ask_gpt2

def main():
    print("🤖 Testing the Fine-Tuned GPT-2 Model for Norton University FAQs\n")
    
    # Test with a few questions
    test_questions = [
        "What is Norton University?",
        "Where is the campus located?",
        "Do you offer scholarships?",
        "What programs does Norton University offer?",
        "Is Norton University accredited?"
    ]

    for question in test_questions:
        print(f"🧠 Q: {question}")
        answer = ask_gpt2(question)
        print(f"💡 A: {answer}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
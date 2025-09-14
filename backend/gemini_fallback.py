import os
from dotenv import load_dotenv
from groq import Groq
import time

# Load environment variables
load_dotenv()

# Load Groq API key
groq_key = os.getenv("GROQ_API_KEY")

groq_client = None
if groq_key:
    groq_client = Groq(api_key=groq_key)
else:
    print("⚠️ GROQ_API_KEY not found")

# Track rate limits
groq_limits = {
    'last_error_time': 0,
    'error_count': 0,
    'cooldown_until': 0
}


def ask_ai(question: str) -> str:
    """
    Groq AI query with rate limit handling
    """
    current_time = time.time()

    # Check if Groq is in cooldown (rate limited)
    if current_time < groq_limits['cooldown_until']:
        remaining_cooldown = int(groq_limits['cooldown_until'] - current_time)
        print(f"⚡ Groq in cooldown ({remaining_cooldown}s), please wait...")
        return "⚠️ Groq API is currently in cooldown due to rate limits. Please try again later."

    # Try Groq if available
    if groq_key:
        result = ask_groq(question)

        # Check if Groq succeeded (not a rate limit error)
        if not is_rate_limit_error(result):
            groq_limits['error_count'] = 0  # Reset error counter on success
            return result

        # Handle rate limit
        print("⚠️ Groq rate limited, setting cooldown...")
        groq_limits['error_count'] += 1
        groq_limits['last_error_time'] = current_time
        # Set cooldown: longer cooldown for repeated errors
        # Max 30min cooldown
        cooldown_minutes = min(30, groq_limits['error_count'] * 5)
        groq_limits['cooldown_until'] = current_time + (cooldown_minutes * 60)
        return "⚠️ Groq API rate limit exceeded. Please try again later."

    return "⚠️ Groq API not available. Please check API key."


async def ask_ai_async(question: str) -> str:
    """
    Async wrapper for ask_ai function
    """
    import asyncio
    from functools import partial

    # Run the synchronous function in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(ask_ai, question))


def is_rate_limit_error(response: str) -> bool:
    """Check if response indicates a rate limit error"""
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "quota",
        "limit exceeded",
        "overload"
    ]

    response_lower = response.lower()
    return any(indicator in response_lower for indicator in rate_limit_indicators)


def ask_groq(question: str) -> str:
    """Groq-specific implementation"""
    if not groq_key:
        return "Groq API not configured"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="groq/compound-mini",
            temperature=0.7,
            max_tokens=1024,
            timeout=10  # Add timeout to prevent hanging
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e).lower()

        # Specific rate limit handling
        if any(indicator in error_msg for indicator in ['rate', '429', 'quota', 'limit']):
            return "Rate limit exceeded. Please try again later."

        return f"Groq error: {str(e)}"

# Utility function to check status


def get_ai_status():
    """Get current AI provider status"""
    current_time = time.time()
    status = {
        'groq_available': groq_key is not None,
        'groq_in_cooldown': current_time < groq_limits['cooldown_until'],
        'cooldown_remaining': max(0, int(groq_limits['cooldown_until'] - current_time)),
        'groq_error_count': groq_limits['error_count']
    }
    return status


# Example usage and testing
if __name__ == "__main__":
    # Check status
    status = get_ai_status()
    print("🔍 AI Status:", status)

    # Make a query
    response = ask_ai("What is artificial intelligence?")
    print("Response:", response)

    # Simulate rate limit (for testing)
    groq_limits['cooldown_until'] = time.time() + 300  # 5 min cooldown
    response = ask_ai("Another question?")
    print("Fallback response:", response)

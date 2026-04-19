"""
GPT post-processing: converts a stream of ASL word predictions into a natural sentence.
"""

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment


def words_to_sentence(words: list[str]) -> str:
    """Send a word list to GPT and return a natural sentence.

    Args:
        words: List of predicted ASL signs e.g. ["hello", "yes", "want", "help"]

    Returns:
        A grammatically correct sentence string.
    """
    if not words:
        return ""

    prompt = (
        f"Convert these ASL sign predictions into a single natural English sentence. "
        f"Only return the sentence, nothing else.\n\nSigns: {', '.join(words)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )

    return response.choices[0].message.content.strip()

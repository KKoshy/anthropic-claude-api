"""Controlling creativity by setting Temperature"""

from anthropic import Anthropic
from log.logger import get_logger
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
log = get_logger(__name__)


def add_user_message(messages: list, text: str):
    log.info(f"Adding user message: {text}")
    user_message = {
        "role": "user",
        "content": text
    }
    messages.append(user_message)

def chat(messages: list, system: Optional[str]=None, temperature: float=1.0) -> str:
    model = 'claude-haiku-4-5'
    client = Anthropic()
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }
    if system:
        params["system"] = system
    reply = client.messages.create(
        **params
    )
    return reply.content[0].text


if __name__=="__main__":
    print("Sherlock Says 🔎")
    user_msg = """
    Share a single sentence from Sherlock Holmes by Sherlock Holmes
    """

    # temperature - 0
    # Almost always - 🔎# "The world is full of obvious things which nobody by any chance ever observes."
    # Low temperature - Model becomes very deterministic, always picking the highest probability token
    messages = []
    add_user_message(messages=messages, text=user_msg)
    dialog = chat(messages=messages, temperature=0)
    print(f"\n🔎{dialog}")
    print(f"{'*'*50}")
    print("\n")

    # temperature - 0.97
    # Hight temperature - Model distributes probability more evenly across all possible tokens, introducing
    # more randomness
    messages = []
    add_user_message(messages=messages, text=user_msg)
    dialog = chat(messages=messages, temperature=0.97)
    print(f"\n🔎{dialog}")


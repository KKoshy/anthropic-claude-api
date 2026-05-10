"""AWS CLI assistant with structured response"""

from anthropic import Anthropic
from dotenv import load_dotenv
from log.logger import get_logger
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

def add_assistant_message(messages: list, text: str):
    log.info(f"Adding assistant message: {text}")
    assistant_message = {
        "role": "assistant",
        "content": text
    }
    messages.append(assistant_message)

def chat(messages: list, system: Optional[str]=None, stop_sequences: Optional[list]=None):
    model = "claude-haiku-4-5"
    client = Anthropic()
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": 0.97
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    reply = client.messages.create(
        **params
    )
    return reply.content[0].text


if __name__ == "__main__":
    system = "You are an AWS expert"
    messages = []
    user_msg = "Generate three different sample AWS CLI commands. Each should be very short"

    add_user_message(messages=messages, text=user_msg)
    add_assistant_message(messages=messages, text="Here are all three commands in a single block without any comments:\n```bash")
    reply = chat(messages=messages, system=system, stop_sequences=["```"])
    log.info(f"AWS commands:\n{reply.strip()}")

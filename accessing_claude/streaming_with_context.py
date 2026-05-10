"""Enabling response streaming with context manager"""

import time
from anthropic import Anthropic
from log.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)

def add_user_message(messages: list, text: str):
    log.info(f"Adding user message: {text}")
    user_message = {
        'role': 'user',
        'content': text
    }
    messages.append(user_message)

def stream_chat(messages: list, temperature: float=0.97):
    model='claude-haiku-4-5'
    client = Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=1000,
        messages=messages,
        temperature=temperature
    ) as stream:
        for text in stream.text_stream:
            print(text, end="")
            # pass

    # To get the complete message from stream object
    # print(stream.get_final_message().content[0].text)

if __name__=="__main__":
    log.info("MadeUp Database")
    user_msg = """
    Write 1 sentence description of a fake database
    """

    messages = []
    add_user_message(messages=messages, text=user_msg)
    stream_chat(messages=messages)

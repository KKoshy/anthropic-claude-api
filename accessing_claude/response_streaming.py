"""This is an example for Response Streaming"""

from anthropic import Anthropic
from anthropic.types import RawContentBlockDeltaEvent
from log.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)

def add_user_message(messages: list, text: str):
    log.info(f"Adding user message: {text}")
    user_message = {
        "role": "user",
        "content": text
    }
    messages.append(user_message)

def chat(messages: list, stream: bool=False, temperature: float=0.97):
    client = Anthropic()
    model = 'claude-haiku-4-5'
    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        stream=stream,
        temperature=temperature
    )
    if stream:
        for event in reply:
            if type(event) is RawContentBlockDeltaEvent:
                print(event.delta.text, end="")
    else:
        return reply.content[0].text


if __name__=="__main__":
    log.info("A comedian with and without stream")
    user_msg = """
    Write a joke in single sentence
    """

    # without stream
    log.info("Without response streaming")
    messages = []
    add_user_message(messages=messages, text=user_msg)
    assistant_msg = chat(messages=messages)
    print(assistant_msg)

    # with stream
    log.info("With response streaming")
    messages = []
    add_user_message(messages=messages, text=user_msg)
    assistant_msg = chat(messages=messages, stream=True)
   

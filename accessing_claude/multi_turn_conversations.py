"""Enabling Multi Turn Converstaions with message list"""

from anthropic import Anthropic
from dotenv import load_dotenv
from log.logger import get_logger

load_dotenv()
log = get_logger(__name__)

def add_user_message(messages: list, text: str):
    log.info(f"Adding user message: {text}")
    user_message = {
        'role': 'user',
        'content': text
    }
    messages.append(user_message)

def add_assistant_message(messages: list, text: str):
    log.info(f"Adding assistant message: {text}")
    assistant_message = {
        'role': 'assistant',
        'content': text
    }
    messages.append(assistant_message)

def chat(message: list) -> str:
    client = Anthropic()
    model = 'claude-haiku-4-5'
    # import pdb; pdb.set_trace()
    assistant_reply = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=message
    )
    log.info(assistant_reply)
    return assistant_reply.content[0].text


if __name__=="__main__":
    chat_history = []

    # first question
    add_user_message(chat_history, 'What is SSD? Answer in one sentence')

    # chat
    reply = chat(chat_history)
    log.info(f"Chat: {reply}")

    # adding assistant reply to history
    add_assistant_message(chat_history, reply)

    # second question
    add_user_message(chat_history, "Add another sentence")

    # chat
    reply = chat(chat_history)
    log.info(f"Chat: {reply}")

"""This is an exercise for Multi Turn conversation"""

from anthropic import Anthropic
from log.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)


def add_user_message(messages: list, text: str):
    # log.info(f"Adding user message: {text}")
    user_message = {
        'role': 'user',
        'content': text
    }
    messages.append(user_message)


def add_assistant_message(messages: list, text: str):
    # log.info(f"Adding assistant message: {text}")
    assistant_message = {
        'role': 'assistant',
        'content': f'{text}; Answer in one sentence'
    }
    messages.append(assistant_message)


def chat(messages: list) -> str:
    model = 'claude-haiku-4-5'
    client = Anthropic()
    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages
    )
    msg = reply.content[0].text
    print(f"\nHChat: {msg}")
    return msg


if __name__ == "__main__":
    log.info("Welcome to Haiku supported chat bot")
    log.info("To quit, enter q")
    messages = []
    while True:
        user_msg = input("\nYou: ").strip()

        # Handling quit event
        if user_msg.lower()=='q':
            print("\nHChat: Powering down...Have a nice day 😄")
            break

        # Handling empty messages
        if not user_msg:
            continue

        add_user_message(messages, user_msg)
        assistant_msg = chat(messages)
        add_assistant_message(messages, assistant_msg)

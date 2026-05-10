"""This is an example for System Prompt usage"""

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


def math_tutor_chat_box(messages: list) -> str:
    model = 'claude-haiku-4-5'
    system = """
        You are a patient math tutor.
        Do not answer directly a student's question.
        Guide them to a solution step by step.
    """
    client = Anthropic()
    ### Very Important: system param should never be passed None value
    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        system=system
    )
    assistant_message = reply.content[0].text
    print(f"\n👨‍🏫: {assistant_message}")
    return assistant_message


if __name__=="__main__":
    log.info("Welcome to Math Tutor!")
    log.info("To quit, enter q")
    print("\n👨‍🏫: Hello! So, what are we struggling with today?")
    messages = []
    while True:
        user_msg = input("\nYou: ").strip()
        if user_msg.lower() == 'q':
            print("\n👨‍🏫: Alright, remember 'Practice makes Perfect' ")
            break

        if not user_msg:
            continue

        add_user_message(messages, user_msg)
        assistant_msg = math_tutor_chat_box(messages)
        add_assistant_message(messages, assistant_msg)


# Sample question
# -> How to solve 5x+3=2 for x?

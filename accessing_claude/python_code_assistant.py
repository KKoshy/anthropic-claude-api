""""This is an example for System Prompt with role Python Coding Expert"""

from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

def add_user_message(messages: list, text: str):
    user_message = {
        'role': 'user',
        'content': text
    }
    messages.append(user_message)


def add_assistant_message(messages: list, text: str):
    assistant_message = {
        'role': 'assistant',
        'content': text
    }
    messages.append(assistant_message)


def chat_box(messages: list, system: Optional[str]=None):
    model = 'claude-haiku-4-5'
    client = Anthropic()
    params = {
        'model': model,
        'max_tokens': 1000,
        'messages': messages,
    }
    if system:
        params['system'] = system
    reply = client.messages.create(**params)
    assistant_msg = reply.content[0].text
    return assistant_msg


if __name__=="__main__":
    print("Welcome to PythonCodeAssistant 🐍")
    print("To quit, enter q")
    system = """
        You are a Python Coding Expert who writes very concise code
    """
    messages = []

    print("\n🐍: What are we building today...?")
    while True:
        user_msg = input("\nYou: ").strip()
        
        if user_msg.lower() == 'q':
            print("\n🐍: Sssaluth...Oh, hope you're aware that you ARE a Parseltongue...")
            break

        if not user_msg:
            continue

        add_user_message(messages, user_msg)
        assistant_msg = chat_box(messages, system)
        print(f"\n🐍: {assistant_msg}")
        add_assistant_message(messages, assistant_msg)



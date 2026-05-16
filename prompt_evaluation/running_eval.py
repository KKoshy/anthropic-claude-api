import os
import json
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

def run_prompt(test_case: dict):
    log.info("Merging prompt and test case input; obtaining result...")
    messages = []
    prompt = f"""
    Please solve the following task:

    {test_case['task']}
    """
    add_user_message(messages=messages, text=prompt)
    output = chat(messages=messages)
    return output

def run_test_case(test_case: dict):
    log.info("Invoking prompt; grading result...")
    output = run_prompt(test_case)

    # TODO: Grading
    score = 10

    return {
        "test_case": test_case,
        "output": output,
        "score": score
    }

def run_eval(dataset: list):
    log.info("Running evaluation on each test case...")
    results = []
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    return results


if __name__=="__main__":
    log.info("Running evaluation on Prompt")

    # collecting dataset
    with open(os.path.join("data", "dataset.json"), "r") as f:
        dataset = json.load(f)

    results = run_eval(dataset=dataset)

    # storing results
    with open(os.path.join("data", "prompt_eval.json"), "w") as f:
        json.dump(results, f)

    

"""This is an example for generating test dataset for Prompt Evaluation"""

import os
import json
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
        "messages": messages
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    reply = client.messages.create(
        **params
    )
    return reply.content[0].text


if __name__=="__main__":
    log.info("Generating dataset")
    user_msg = """
    Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
    that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
    each representing task that requires Python, JSON, or a Regex to complete.

    Example output:
    ```json
    [
        {
            "task": "Description of task",
            "format": "json" or "python" or "regex",
            "solution_criteria": "Key criteria for evaluating the solution"
        },
        ...additional
    ]
    ```

    * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
    * Focus on tasks that do not require writing much code

    Please generate 10 objects.
    """
    messages = []

    add_user_message(messages=messages, text=user_msg)
    add_assistant_message(messages=messages, text="```JSON")

    reply = chat(messages=messages, stop_sequences=["```"])

    # parsing the JSON response
    json_dataset = json.loads(reply.strip())

    # saving the dataset
    with open(os.path.join("data", "dataset.json"), "w") as f:
        json.dump(json_dataset, f)

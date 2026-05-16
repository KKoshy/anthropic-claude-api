"""This file holds example for Prompt evaluation with model based grading"""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from log.logger import get_logger
from statistics import mean
from typing import Optional

load_dotenv()
log = get_logger(__name__)


def add_user_message(messages: list, text: str):
    log.info(f"Adding user message: {text}")
    user_msg = {
        "role": "user",
        "content": text
    }
    messages.append(user_msg)

def add_assistant_message(messages: list, text: str):
    log.info(f"Adding assistant message: {text}")
    assistant_msg = {
        "role": "assistant",
        "content": text
    }
    messages.append(assistant_msg)

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
    log.info("Adding test case to the prompt...")
    messages = []

    prompt = f"""
    Please solve the following task:
    {test_case['task']}
    """

    add_user_message(messages=messages, text=prompt)
    output = chat(messages=messages)
    return output

def grade_by_model(test_case: dict, output: str):
    eval_prompt = f"""
        You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

        Original Task:
        <task>
        {test_case["task"]}
        </task>

        Solution to Evaluate:
        <solution>
        {output}
        </solution>

        Output Format
        Provide your evaluation as a structured JSON object with the following fields, in this specific order:
        - "strengths": An array of 1-3 key strengths
        - "weaknesses": An array of 1-3 key areas for improvement
        - "reasoning": A concise explanation of your overall assessment
        - "score": A number between 1-10

        Respond with JSON. Keep your response concise and direct.
        Example response shape:
        {{
            "strengths": string[],
            "weaknesses": string[],
            "reasoning": string,
            "score": number
        }}
    """

    messages = []
    add_user_message(messages=messages, text=eval_prompt)
    add_assistant_message(messages=messages, text="```json")
    reply = chat(messages=messages, stop_sequences=["```"])
    return json.loads(reply)

def run_eval(test_case: dict):
    log.info("Invoking prompt to grade the test case...")
    output = run_prompt(test_case=test_case)

    # INFO: Evaluation criteria for this prompt
    # Format - Should return only Python, JSON, or Regex without explanation
    # Valid Syntax - Produced code should have valid syntax
    # Task Following - Response should directly address the user's task with accurate code
    # Format, Valid Syntax - Code based evaluation
    # Task Following - Model based evaluation
    # Hence it is a combination

    # TODO: grading
    results = grade_by_model(test_case=test_case, output=output)
    reasoning = results["reasoning"]
    score = results["score"]

    return {
        "test_case": test_case,
        "output": output,
        "reasoning": reasoning,
        "score": score
    }

def run_test_case(dataset: list):
    results = []
    for test_case in dataset:
        result = run_eval(test_case=test_case)
        results.append(result)

    # calculating average score of the prompt
    avg_score = mean([result["score"] for result in results])
    log.info(f"Average score: {avg_score}")

    return results


if __name__=="__main__":
    log.info("Prompt Evaluation: Model based grading")

    # obtaining dataset
    with open(os.path.join("data", "dataset.json"), "r") as f:
        dataset = json.load(f)

    # obtaining results
    results = run_test_case(dataset=dataset)

    # storing results
    with open(os.path.join("data", "model_based_grading.json"), "w") as f:
        json.dump(results, f)

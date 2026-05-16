"""This file holds the complete example for Prompt evaluation with Code based grading and Model based grading"""

import ast
import json
import os
import re
from anthropic import Anthropic
from dotenv import load_dotenv
from log.logger import get_logger
from statistics import mean
from typing import Optional


load_dotenv()
log = get_logger(__name__)

class Chat:
    @staticmethod
    def add_user_message(messages: list, text: str):
        log.info(f"Adding user message: {text}")
        user_message = {
            "role": "user",
            "content": text
        }
        messages.append(user_message)

    @staticmethod
    def add_assistant_message(messages: list, text: str):
        log.info(f"Adding assistant message: {text}")
        assistant_message = {
            "role": "assistant",
            "content": text
        } 
        messages.append(assistant_message)

    @staticmethod
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


class SyntaxValidator:
    @staticmethod
    def validate_json(text: str):
        try:
            json.loads(text.strip())
            return 10
        except json.JSONDecodeError:
            return 0
        
    @staticmethod
    def validate_python(text: str):
        try:
            ast.parse(text.strip())
            return 10
        except SyntaxError:
            return 0
        
    @staticmethod
    def validate_regex(text: str):
        try:
            re.compile(text.strip())
            return 10
        except re.error:
            return 0


class Grader:
    @staticmethod
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
        Chat.add_user_message(messages=messages, text=eval_prompt)
        Chat.add_assistant_message(messages=messages, text="```json")
        reply = Chat.chat(messages=messages, stop_sequences=["```"])
        return json.loads(reply)

    @staticmethod
    def grade_by_code(test_case: dict, output: str):
        if test_case["format"] == "json":
            return SyntaxValidator.validate_json(output)
        elif test_case["format"] == "python":
            return SyntaxValidator.validate_python(output)
        else:
            return SyntaxValidator.validate_regex(output)


class PromptEval:
    @staticmethod
    def run_prompt(test_case: dict):
        log.info("Adding test case to prompt")

        prompt = f"""
        Please solve the following task:

        {test_case['task']}

        * Respond only with Python, JSON or plain Regex
        * Do not add any explanation, description or comments
        """

        messages = []
        Chat.add_user_message(messages=messages, text=prompt)
        output = Chat.chat(messages=messages)
        return output

    @staticmethod
    def run_eval(test_case: dict):
        log.info("Evaluating prompt...")

        output = PromptEval.run_prompt(test_case)

        # INFO: Evaluation criteria for this prompt
        # Format - Should return only Python, JSON, or Regex without explanation
        # Valid Syntax - Produced code should have valid syntax
        # Task Following - Response should directly address the user's task with accurate code
        # Format, Valid Syntax - Code based evaluation
        # Task Following - Model based evaluation
        # Hence it is a combination

        # Format - Handled by updating the prompt
        # Valid Syntax + Prompt - SyntaxValidator
        # Task Following - Model based validation

        # Model grading
        model_result = Grader.grade_by_model(test_case=test_case, output=output)
        reasoning = model_result["reasoning"]
        model_score = model_result["score"]

        # Code grading
        syntax_score = Grader.grade_by_code(test_case=test_case, output=output)

        # Calculating average score
        score = (model_score + syntax_score) / 2

        return {
            "test_case": test_case,
            "output": output,
            "reasoning": reasoning,
            "score": score
        }

    @staticmethod
    def run_test_case(dataset: list):
        results = []
        for test_case in dataset:
            result = PromptEval.run_eval(test_case=test_case)
            results.append(result)

        # calculate average score
        average_score = mean([result["score"] for result in results])
        log.info(f"Average Prompt score: {average_score}")

        return results

if __name__=="__main__":
    log.info("Prompt evaluation by code based and model based grading")

    # obtaining dataset
    with open(os.path.join("data", "dataset.json"), "r") as f:
        dataset = json.load(f)

    # obtaining results
    results = PromptEval.run_test_case(dataset=dataset)

    # storing results
    with open(os.path.join("data", "code_based_grading.json"), "w") as f:
        json.dump(results, f)

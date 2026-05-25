import os
import re
import json

from anthropic import Anthropic
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from log.logger import get_logger
from statistics import mean
from textwrap import dedent
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
    
class PromptEvalReport:
    pass

class DataGenerator:
    @staticmethod
    def render(template: str, variables: dict):
        """
        When building prompts dynamically, you need to insert variables into text. 
        Python's built-in f-strings work, but they have a problem — curly braces are special characters.
        """
        # identifying placeholders 
        placeholders = re.findall(r"{([^{}]+)}", template)
        
        # replace placeholder with actual value
        result = template
        for placeholder in placeholders:
            if placeholder in variables:
                result = result.replace(
                    "{"+placeholder+"}", variables[placeholder]
                )

        # replace nested brackets
        return result.replace("{{", "{").replace("}}", "}")
    
    @staticmethod
    def generate_unique_ideas(task_description: str, prompt_input_spec: dict, no_of_cases: int):
        """
        Ask Claude to brainstorm a list of diverse scenarios 
        Few whys,
        Why format the spec as a string? — Claude needs to read it as plain text inside a prompt
        Why temperature=1.0? — diversity matters here; you don't want 5 nearly identical ideas
        Why dedent()? — removes the awkward leading spaces from indented triple-quoted strings
        """

        # Step-01: build the example inputs for unique idea generation
        # converts {"height": "Athlete's height in cm", "weight": "Athlete's weight in kg"}
        # to "height": str # Athlete's height in cm, "weight": str # Athlete's weight in kg,
        example_prompt_input = ""
        for key, value in prompt_input_spec.items():
            val = value.replace("\n", "\\n")
            example_prompt_input+=f'"{key}": str # {val},'

        # Step-02: prompt for unique idea generation for test cases
        prompt = """
        Generate {no_of_cases} unique ideas for testing this task

        <task_description>
        {task_description}
        </task_description>

        The prompt will receive the following inputs
        <prompt_inputs>
        {prompt_input_spec}
        </prompt_inputs>

        Each idea should represent a distinct scenario or example that tests different aspects of the task.

        Output Format:
        Provide your response as a structured JSON array where each item is a brief description of the idea.
        
        Example:
        ```json
        [
            "Testing with technical computer science terminology",
            "Testing with medical research findings",
            "Testing with complex mathematical concepts",
            ...
        ]
        ```
        
        Ensure each idea is:
        - Clearly distinct from the others
        - Relevant to the task description
        - Specific enough to guide generation of a full test case
        - Quick to solve without requiring extensive computation or multi-step processing
        - Solvable with no more than 400 tokens of output

        Remember, only generate {no_of_cases} unique ideas
        """

        system_prompt = "You are a test scenario designer with expertise in generating diverse, unique test scenarios covering all aspects"

        # Step-03: render the prompt
        rendered_prompt = DataGenerator.render(dedent(prompt), {
            "task_description": task_description,
            "prompt_input_spec": example_prompt_input,
            "no_of_cases": no_of_cases
        })

        # Step-04: generate the ideas with claude
        messages = []
        Chat.add_user_message(messages=messages, text=rendered_prompt)
        Chat.add_assistant_message(messages=messages, text="```json")
        text = Chat.chat(
            messages=messages,
            system=system_prompt,
            stop_sequences=["```"],
        )
        
        return json.loads(text)
    
    @staticmethod
    def generate_test_case(task_description: str, idea: str, prompt_inputs_spec: dict):
        """
        This method takes one such idea and fleshes it out into a full test case with concrete input values and evaluation criteria.
        """

        # Step-01: build the example inputs for unique idea generation
        # converts {"height": "Athlete's height in cm", "weight": "Athlete's weight in kg"}
        example_prompt_inputs = ""
        for key, value in prompt_inputs_spec.items():
            val = value.replace("\n", "\\n")
            example_prompt_inputs += f'"{key}": "EXAMPLE_VALUE", // {val}\n'

        allowed_keys = ", ".join([f'"{key}"' for key in prompt_inputs_spec.keys()])

        # Step-02: prompt for test case generation
        prompt = """
        Generate a single detailed test case for a prompt evaluation based on:
        
        <task_description>
        {task_description}
        </task_description>
        
        <specific_idea>
        {idea}
        </specific_idea>
        
        <allowed_input_keys>
        {allowed_keys}
        </allowed_input_keys>
        
        Output Format:
        ```json
        {{
            "prompt_inputs": {{
            {example_prompt_inputs}
            }},
            "solution_criteria": ["criterion 1", "criterion 2", ...] // Concise list of criteria for evaluating the solution, 1 to 4 items
        }}
        ```
        
        IMPORTANT REQUIREMENTS:
        - You MUST ONLY use these exact input keys in your prompt_inputs: {allowed_keys}        
        - Do NOT add any additional keys to prompt_inputs
        - All keys listed in allowed_input_keys must be included in your response
        - Make the test case realistic and practically useful
        - Include measurable, concise solution criteria
        - The solution criteria should ONLY address the direct requirements of the task description and the generated prompt_inputs
        - Avoid over-specifying criteria with requirements that go beyond the core task
        - Keep solution criteria simple, focused, and directly tied to the fundamental task
        - The test case should be tailored to the specific idea provided
        - Quick to solve without requiring extensive computation or multi-step processing
        - Solvable with no more than 400 tokens of output
        - DO NOT include any fields beyond those specified in the output format

        Here's an example of a sample input with an ideal output:
        <sample_input>

        <sample_task_description>
        Extract topics out of a passage of text
        </sample_task_description>

        <sample_specific_idea>
        Testing with a text that contains multiple nested topics and subtopics (e.g., a passage about renewable energy that covers solar power economics, wind turbine technology, and policy implications simultaneously)
        </sample_specific_idea>

        <sample_allowed_input_keys>
        "content"
        </sample_allowed_input_keys>

        </sample_input>

        <ideal_output>
        ```json
        {
            "prompt_inputs": {
                "content": "The transition to renewable energy encompasses numerous interdependent dimensions. Solar photovoltaic technology has seen dramatic cost reductions, with panel efficiency improving 24% since 2010 while manufacturing costs declined by 89%, making it economically competitive with fossil fuels in many markets. Concurrently, wind energy has evolved through innovative turbine designs featuring carbon-fiber composite blades and advanced control systems that increase energy capture by 35% in low-wind conditions."
            },
            "solution_criteria": [
                "Includes all topics mentioned"   
            ]
        }
        ```
        </ideal_output>
        This is ideal output because the solution criteria is concise and doesn't ask for anything outside of the scope of the task description.
        """

        system_prompt = "You are a test case creator specializing in designing evaluation scenarios."

        # Step-03: render the prompt
        rendered_prompt = DataGenerator.render(dedent(prompt),
            {
                "task_description": task_description,
                "allowed_keys": allowed_keys,
                "idea": idea,
                "example_prompt_inputs": example_prompt_inputs
            }
        )

        # Step-04: form test case
        messages = []
        Chat.add_user_message(messages=messages, text=rendered_prompt)
        Chat.add_assistant_message(messages=messages, text="```json")
        text = Chat.chat(
            messages=messages,
            system=system_prompt,
            stop_sequences=["```"]
        )
        test_case = json.loads(text)
        test_case["scenario"] = idea
        test_case["task_description"] = task_description

        return test_case
    
    @staticmethod
    def generate_dataset(task_description: str, prompt_input_specs: dict, no_of_cases: int, output_file: str):
        """
        Orchestrates the two previous methods — it calls generate_unique_ideas to get a list of ideas, 
        then calls generate_test_case for each idea, and saves the results to a JSON file.

        Uses parallelism — instead of generating test cases one by one, 
        it fires them all at once using ThreadPoolExecutor
        """
        
        # Step-01: gather ideas
        ideas = DataGenerator.generate_unique_ideas(
            task_description=task_description,
            prompt_input_spec=prompt_input_specs,
            no_of_cases=no_of_cases
        )

        # Step-02: form test case from ideas with parallel execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idea = {
                executor.submit(DataGenerator.generate_test_case, task_description, idea, prompt_input_specs): idea for idea in ideas
            }

        # Step-03: collect results and track progress
        dataset = []
        completed = 0
        last_reported_percentage = 0

        for future in as_completed(future_to_idea):
            try:
                result = future.result()
                completed += 1
                current_percentage = (completed/no_of_cases) * 100
                # log for sharing progress every 20%
                milestone_percentage = (current_percentage//20) * 20
                if milestone_percentage > last_reported_percentage:
                    log.info(f"Generated {milestone_percentage} of total cases")
                    last_reported_percentage = milestone_percentage
                dataset.append(result)
            except Exception as e:
                log.error(f"Unable to generate test case: {e}")

        # Step-04: save results to file
        with open(output_file, "w") as f:
            json.dump(dataset, f)

        return dataset

class PromptEvaluator:
    
    @staticmethod
    def grade_output(test_case: dict, output: str, extra_criteria: Optional[str]=None):
        """
        Once your prompt has generated an output, you need to judge how good it is. 
        This method asks Claude to evaluate the output against the test case's solution_criteria and any extra_criteria you provide, 
        returning a structured score with reasoning.
        """

        # Step-01: format prompt inputs into Claude procesable format
        prompt_inputs = ""
        for key, value in test_case["prompt_inputs"].items():
            val = value.replace("\n", "\\n")
            prompt_inputs += f'"{key}": "{val}", \n'

        # Step-02: handle extra criteria
        extra_criteria_section=""
        if extra_criteria:
            extra_criteria_template = """
            Mandatory Requirements - ANY VIOLATION MEANS AUTOMATIC FAILURE (score of 3 or lower):
            <extra_criteria>
            {extra_criteria}
            </extra_criteria>
            """
            extra_criteria_section = DataGenerator.render(
                extra_criteria_template,
                {
                    "extra_criteria": extra_criteria
                }
            )

        # Step-03: form prompt for evaluation
        prompt = """
        Your task is to evaluate the following AI-generated solution with EXTREME RIGOR.

        Original task description:
        <task_description>
        {task_description}
        </task_description>

        Original task inputs:
        <task_inputs>
        {{ {prompt_inputs} }}
        </task_inputs>

        Solution to Evaluate:
        <solution>
        {output}
        </solution>

        Criteria you should use to evaluate the solution:
        <criteria>
        {solution_criteria}
        </criteria>

        {extra_criteria_section}

        Scoring Guidelines:
        * Score 1-3: Solution fails to meet one or more MANDATORY requirements
        * Score 4-6: Solution meets all mandatory requirements but has significant deficiencies in secondary criteria
        * Score 7-8: Solution meets all mandatory requirements and most secondary criteria, with minor issues
        * Score 9-10: Solution meets all mandatory and secondary criteria

        IMPORTANT SCORING INSTRUCTIONS:
        * Grade the output based ONLY on the listed criteria. Do not add your own extra requirements.
        * If a solution meets all of the mandatory and secondary criteria give it a 10
        * Don't complain that the solution "only" meets the mandatory and secondary criteria. Solutions shouldn't go above and beyond - they should meet the exact listed criteria.
        * ANY violation of a mandatory requirement MUST result in a score of 3 or lower
        * The full 1-10 scale should be utilized - don't hesitate to give low scores when warranted

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

        # Step-04: render prompt
        rendered_prompt = DataGenerator.render(prompt, {
            "task_description": test_case["task_description"],
            "prompt_inputs": prompt_inputs,
            "output": output,
            "solution_criteria": "\n".join(test_case["solution_criteria"]),
            "extra_criteria_section": extra_criteria_section
        })

        # Step-05: grade with Claude
        messages = []
        Chat.add_user_message(messages=messages, text=rendered_prompt)
        Chat.add_assistant_message(messages=messages, text="```json")
        eval_text = Chat.chat(
            messages=messages,
            stop_sequences=["```"]
        )
        return json.loads(eval_text)

    @staticmethod
    def run_test_case(test_case: dict, run_prompt_function: Callable, extra_criteria: Optional[str]=None):
        """
        Thin orchestration method — it ties together the prompt execution and grading into one single call.
        """
        
        # Step-01: run the prompt
        output = run_prompt_function(test_case["prompt_inputs"])

        # Step-02: grade the output
        model_grade = PromptEvaluator.grade_output(test_case=test_case, output=output, extra_criteria=extra_criteria)
        model_score = model_grade["score"]
        model_reasoning = model_grade["reasoning"]

        return {
            "output": output,
            "test_case": test_case,
            "score": model_score,
            "reasoning": model_reasoning
        }
    
    @staticmethod
    def run_evaluation(run_prompt_function, 
                       dataset_file, 
                       extra_criteria=None, 
                       output_file="output.json",
                       html_output="output.html"):
        """
        Top-level method. It loads the dataset, runs all test cases in parallel, prints the average score, and saves both a JSON and HTML report.
        """
        
        # Step-01: load dataset
        with open(dataset_file, "r") as f:
            dataset = json.load(f)

        # Step-02: evaluate test cases with parallel execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_test_case = {
            executor.submit(PromptEvaluator.run_test_case, test_case, run_prompt_function, extra_criteria): test_case for test_case in dataset
            }
        
        # Step-03: collect results and track progress
        results = []
        completed = 0
        total = len(dataset)
        last_reported_percentage = 0

        try:
            for future in as_completed(future_to_test_case):
                result = future.result()
                completed += 1
                current_percentage = (completed/total) * 100
                milestone_percentage = (current_percentage//20) * 20
                if milestone_percentage > last_reported_percentage:
                    log.info(f"Generated results for {milestone_percentage}")
                    last_reported_percentage = milestone_percentage

                results.append(result)
        except Exception as e:
            log.error(f"Unable to obtain results for the test case: {e}")

        # Step-04: calculate average score
        avg_score = mean([result["score"] for result in results])
        log.info(f"Average score: {avg_score}")

        # Step-05: storing output
        with open(output_file, "w") as f:
            json.dump(results, f)

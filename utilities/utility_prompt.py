import os
import warnings
from typing import Any, Dict, Optional

import openai
import yaml
from dotenv import find_dotenv, load_dotenv
from utility import normalize_dataset_name

_ = load_dotenv(find_dotenv())  # read local .env file

if os.environ["OPENAI_API_KEY"] == "fakeapikey1234567890":
    warnings.warn(
        "You are using a FAKE OPENAI API key. In order for the code to run properly, please set your real OPENAI API key in '.env' located in the root directory of this project."
    )

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_completion_gpt(
    prompt,
    system_prompt=None,
    history=None,
    model="gpt-3.5-turbo",
    temperature=0,
    seed=0,
):
    """
    Get the completion from GPT
    :param prompt: The prompt to be completed
    :param system_prompt: The system prompt
    :param history: The history of the conversation, of the form [user_prompt, assistant_response]
    :param model: The model to use
    :return: The completion
    """
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []

    if history:
        last_user_prompt, last_assistant_response = history
        messages.append({"role": "user", "content": last_user_prompt})
        messages.append({"role": "assistant", "content": last_assistant_response})

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output, ranging from 0 to 2
        seed=seed,  # use this to get the same result for the same prompt, if system_fingerprint is not changed
    )

    return response.choices[0].message["content"]


PROMPTS_FOLDER = "prompts"
LLM_KEYS = ["GPT-3.5 turbo", "GPT-3.5", "GPT-4", "Claude", "Claude 2"]

# group these into a func
# err_msg = f"Invalid LLM name. Must be one of the following: \n{LLM_KEYS}"
#       raise ValueError(err_msg)


def user_confirm_warning(warning_message):
    print(warning_message)
    user_input = input("Type 'Yes' to confirm: ")
    return user_input.strip().lower() == "yes"


def get_yaml_path(dataset_name: str) -> str:
    """
    Construct the path to the YAML file for the given dataset.
    """
    return os.path.join(PROMPTS_FOLDER, f"{normalize_dataset_name(dataset_name)}.yaml")


def read_prompt(
    dataset_name: str,
    llm_model: Optional[str] = None,
    is_system: Optional[bool] = False,
    is_individual: Optional[bool] = False,
    strategy: Optional[str] = "direct",
    step_idx: Optional[str] = None,
    dir: str = "",
) -> Dict[str, Any]:
    """
    Read the prompt data from the YAML file for the given dataset.
    :param dataset_name: Name of the dataset
    :param llm_model: Name of the LLM model
    :param is_system: Whether the prompt is a system prompt
    :param is_individual: Whether the prompt is an individual prompt
    :param strategy: labeling strategy (direct, func, tool)
    :param step_idx: Index of the step
    :return: Dictionary containing the prompt data
    """
    if strategy == "direct":
        yaml_path = dir + "/" + get_yaml_path(dataset_name)
        if not os.path.isfile(yaml_path):
            return {}  # Return an empty dict if the file does not exist

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file) or {}

        if llm_model in LLM_KEYS:
            data = data.get(llm_model, {})

            if step_idx:
                is_system = is_system or not is_individual
                prompt_type = "system_prompts" if is_system else "individual_prompts"
                data = data.get(prompt_type, {}).get(f"step_{step_idx}", {})
        elif llm_model is not None:
            raise ValueError(
                f"Invalid LLM name. Must be one of the following: \n{LLM_KEYS}"
            )
    elif strategy == "tool":
        yaml_path = dir + "/" + PROMPTS_FOLDER + "/" + "tool.yaml"

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file) or {}
        if is_system:
            data = data["system_prompt"]
        else:
            data = data["individual_prompt"]
    elif strategy == "func":
        yaml_path = dir + "/" + PROMPTS_FOLDER + "/" + "func.yaml"

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file) or {}
        if is_system:
            data = data["system_prompt_function_generation"]
        else:
            if step_idx == 1:
                data = data["function_prompt_natural_language"]
            elif step_idx == 2:
                data = data["function_prompt_function_generation"]
    return data


def write_prompt(dataset_name: str, prompt_data: Dict[str, Any]):
    """
    Write the prompt data to the YAML file for the given dataset.
    """
    yaml_path = get_yaml_path(dataset_name)
    with open(yaml_path, "w") as file:
        yaml.dump(prompt_data, file, default_flow_style=False)


def write_prompt(dataset_name: str, prompt_data: Dict[str, Any]):
    yaml_path = get_yaml_path(dataset_name)
    with open(yaml_path, "w") as file:
        yaml.dump(prompt_data, file, default_flow_style=False)

    print(f"Updated prompts for {dataset_name}.")


def update_prompt(
    dataset_name: str,
    llm_model: str,
    step_idx: str,
    prompt_content: str,
    is_system: bool = False,
    is_individual: bool = False,
):
    """
    Update an individual or system prompt for a specific LLM and step index.
    :param dataset_name: Name of the dataset
    :param llm_model: Name of the LLM model
    :param step_idx: Index of the step
    :param prompt_content: Content of the prompt
    :param is_system: Whether the prompt is a system prompt
    :param is_individual: Whether the prompt is an individual prompt
    """
    warning_message = f"You are about to update the prompt for '{dataset_name}' dataset. This action cannot be undone. Are you sure you want to proceed?"
    if user_confirm_warning(warning_message):
        prompt_data = read_prompt(dataset_name)
        if llm_model not in LLM_KEYS:
            raise ValueError(
                f"Invalid LLM name. Must be one of the following: \n{LLM_KEYS}"
            )

        if llm_model not in prompt_data:
            prompt_data[llm_model] = {"system_prompts": {}, "individual_prompts": {}}

        is_system = is_system or not is_individual
        prompt_type = "system_prompts" if is_system else "individual_prompts"

        # Ensure there's a dictionary for the specific step
        if f"step_{step_idx}" not in prompt_data[llm_model][prompt_type]:
            prompt_data[llm_model][prompt_type][f"step_{step_idx}"] = {}

        # Update the prompt for the specific step
        prompt_data[llm_model][prompt_type][f"step_{step_idx}"] = prompt_content

        # Save the updated prompts
        write_prompt(dataset_name, prompt_data)
    else:
        print("Prompt update cancelled.")

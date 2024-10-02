import io
import os
import re

import gradio as gr
import requests
import tiktoken
from dotenv import load_dotenv
from litellm import completion
from pdfminer.high_level import extract_text

from logger import setup_logger

load_dotenv()

API_KEY = os.getenv("API_KEY")

LITELLM_URL = os.getenv("LITELLM_URL")

DEFAULT_AGENT_CONTEXT = """Your task now is to draft a high-quality review outline for a top-tier {} {} for a submission."""

DEFAULT_AGENT_TASK = """Compose a high-quality peer review of an paper submitted to a top-tier {} {} on OpenReview.
Start by "Review outline:".
And then:
"1. Significance and novelty"
"2. Potential reasons for acceptance"
"3. Potential reasons for rejection", List 4 key reasons. For each of 4 key reasons, use **>=2 sub bullet points** to further clarify and support your arguments in painstaking details.
"4. Suggestions for improvement", List 4 key suggestions.

Be thoughtful and constructive. Write Outlines only.
"""

TOKENIZER = tiktoken.encoding_for_model("gpt-4o")

logger = setup_logger("LLM-Reviewer")
logger.info("Logger setup done")


def create_prompt(instruction: str, task: str, paper: str) -> str:
    """
    Creates a formatted prompt from given instruction, task, and paper context.

    :param instruction: The instruction text to be included in the prompt.
    :param task: The task description to be included in the prompt.
    :param paper: The context of the paper to be included in the prompt.
    :return: A formatted string combining the instruction, paper context, and task.
    """
    text_to_send = f"""
    Instruction:
    {instruction}
    ======
    Paper context :
    {paper}
    ======
    Your task:
    {task}
    """
    return text_to_send


def get_number_of_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def parse_pdf_text(text: str) -> str:
    """
    Processes and cleans text extracted from a PDF.

    The function focuses on removing irrelevant parts like page numbers,
    chapter titles, and special characters. It also isolates the abstract
    if present.

    :param text: The text extracted from a PDF.
    :return: The cleaned and processed text.
    """
    # Split the text at the first occurrence of "abstract"
    abstract_split = re.split(r"(?i)\babstract\b", text, maxsplit=1)

    if len(abstract_split) == 2:
        # Use only the text after "abstract"
        _, text = abstract_split
        text = "Abstract\n" + text
    else:
        # Use the text as is if "abstract" isn't found
        text = abstract_split[0]

    # Remove the text after "References"
    text = text.split("References")[0]

    # Patterns of text to be removed
    patterns_to_remove = [
        r"^\d+\s.*",  # lines starting with a number
        r"^\s*Chapter \w+:.*",  # chapter headings
        r"^\d+\s*$",  # lines with only digits
        r"^[^\w\d\s]+\s*$",  # lines with only special characters
        r"^.Version",  # lines starting with 'Version'
        r"^-?\d+,\d+$",  # numerical lines with comma (e.g., page numbers)
        r"^\s*\S\s*$",  # lines with a single non-whitespace character
    ]

    # Split the text into lines and filter out unwanted lines
    lines = text.split("\n")
    for pattern in patterns_to_remove:
        lines = [line for line in lines if not re.match(pattern, line)]

    # Join the cleaned lines back into a single string
    cleaned_text = "\n".join(lines)
    return cleaned_text


def get_completion(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    try:
        response = completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            api_key=API_KEY,
            base_url=LITELLM_URL,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in getting completion: {e}")
        return str(e)


def calculate_prompt_cost(
    prompt: str, completion: str, model_name: str = "gpt-4o-mini"
) -> float:
    model_dict = {
        "gpt-4o": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.000005,
            "output_cost_per_token": 0.000015,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
        },
        "gpt-4o-mini": {
            "max_tokens": 16384,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.00000060,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
        },
        "o1-mini": {
            "max_tokens": 65536,
            "max_input_tokens": 128000,
            "max_output_tokens": 65536,
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000012,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
        },
        "o1-preview": {
            "max_tokens": 32768,
            "max_input_tokens": 128000,
            "max_output_tokens": 32768,
            "input_cost_per_token": 0.000015,
            "output_cost_per_token": 0.000060,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
        },
    }

    costs = model_dict.get(model_name, "gpt-4o-mini")
    prompt_tokens = get_number_of_tokens(prompt)
    completion_tokens = get_number_of_tokens(completion)

    # Print token information for analysis
    logger.info(f"{model_name} -> tokens in prompt: {prompt_tokens}")
    logger.info(f"{model_name} -> tokens in completion: {completion_tokens}")

    prompt_cost = prompt_tokens * costs["input_cost_per_token"]
    completion_cost = completion_tokens * costs["output_cost_per_token"]
    total_cost = prompt_cost + completion_cost
    return round(total_cost, 4)


def process_pdf(
    file_content: object,
    type_of_paper: list,
    agent_task: str,
    type_of_event: str,
    model_name: str,
) -> tuple:
    try:
        with io.BytesIO(file_content) as pdf_file:
            text = extract_text(pdf_file)
            text = parse_pdf_text(text)

        logger.info(f"PDF processed - {model_name}")
        if len(type_of_paper) == 0:
            type_of_paper = ["Conference"]

        type_of_paper = ", ".join(type_of_paper)

        logger.info(f"Creating review for {type_of_paper} on topic {type_of_event}")

        agent_context = DEFAULT_AGENT_CONTEXT.format(type_of_event, type_of_paper)
        agent_tasks = agent_task.format(type_of_event, type_of_paper)

        prompt = create_prompt(agent_context, agent_tasks, text)

        logger.info(f"Prompt done - {model_name}")

        review = get_completion(prompt, model_name)

        logger.info(f"Review done - {model_name}")

        price_per_request = calculate_prompt_cost(prompt, review, model_name)

        price_markdown = f"""
        #### Price per request: {price_per_request} USD
        """

    except Exception as e:
        logger.error(f"Error in processing PDF: {e}")
        review = str(e)
        price_markdown = "Error in processing PDF"

    return review, price_markdown


def get_models() -> list:
    url = "http://147.175.151.44/models"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        models = response.json()
        models = [model["id"] for model in models["data"]]
        return models
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


drop_down_models = get_models()
drop_down_models = drop_down_models if drop_down_models is not None else ["gpt-4o-mini"]

event_types = [
    "Machine Learning (ML)",
    "Blockchain",
    "Telecommunication",
    "Security",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Artificial Intelligence",
    "Computer Science Education",
]

# Set up the GUI layout with custom CSS for buttons
with gr.Blocks(css=".button {background-color: #4CAF50; color: white;}") as demo:
    # Title of the application
    gr.Markdown(
        """# Paper Reviewer - 0.0.2
        This app uses LLM model to generate a **test** review for your paper.

        * Upload **ONLY PDF** files.
        * Please leave feedback on the generated review (teams or mail).

        For more details, check out the
        [github repository](https://github.com/williambrach/llm-reviewer)."""
    )

    # Layout setup: two columns in a row
    with gr.Row():
        # Left column for input components
        with gr.Column():
            gr.Markdown("## Research paper")
            # File upload component for PDFs
            upload_component = gr.File(
                label="PDF to be reviewed", type="binary", value=None
            )

            gr.Markdown("## Configuration")

            # select model
            drop_down = gr.Dropdown(
                drop_down_models,
                label="LLM Model",
                info="Model for creating the review.",
                value=drop_down_models[0],
            )

            # Textbox for entering the context for the review agent
            # agent_context = gr.Textbox(
            #     label="Reviewer-Agent Context",
            #     info="Agent context / persona for the review.",
            #     placeholder="Enter context for the agent...",
            #     lines=5,
            #     value=DEFAULT_AGENT_CONTEXT,
            # )

            agent_context = gr.CheckboxGroup(
                ["Conference", "Journal", "White paper"],
                label="Type of paper",
                value=["Conference"],
                info="Type of paper",
            )

            event = gr.Dropdown(
                event_types,
                label="Type of conference/journal",
                info="What type of conference/journal is the paper for?",
                value=event_types[0],
            )

            # Textbox for entering the tasks for the review agent
            agent_tasks = gr.Textbox(
                label="Reviewer-Agent Tasks",
                info="Agents tasks for the review.",
                placeholder="Enter tasks for the agent...",
                lines=5,
                value=DEFAULT_AGENT_TASK,
            )

            # Button to trigger the review creation process
            process_button = gr.Button("Create Review", variant="primary")

        # Right column for output components
        with gr.Column():
            gr.Markdown("## Metadata")

            # Placeholder for the price markdown output
            price_markdown = gr.Markdown("###")

            gr.Markdown(
                "----------------------------------------------------------------"
            )
            gr.Markdown("## Generated Review ")
            processed_output = gr.Markdown(label="Review")

        # Link the button to the processing function
        process_button.click(
            fn=process_pdf,
            inputs=[upload_component, agent_context, agent_tasks, event, drop_down],
            outputs=[processed_output, price_markdown],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7799, share=False)

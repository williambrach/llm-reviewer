import io
import os
import re

import gradio as gr
import tiktoken  # noqa: E402
from dotenv import load_dotenv
from openai import OpenAI
from pdfminer.high_level import extract_text

load_dotenv()

DEFAULT_AGENT_CONTEXT = """Your task now is to draft a high-quality review outline for a top-tier Machine Learning (ML) conference for a submission."""

DEFAULT_AGENT_TASK = """Compose a high-quality peer review of an ML paper submitted to a top-tier ML conference on OpenReview.
Start by "Review outline:".
And then:
"1. Significance and novelty"
"2. Potential reasons for acceptance"
"3. Potential reasons for rejection", List 4 key reasons. For each of 4 key reasons, use **>=2 sub bullet points** to further clarify and support your arguments in painstaking details.
"4. Suggestions for improvement", List 4 key suggestions.

Be thoughtful and constructive. Write Outlines only. 
"""

OPEN_AI_KEY = "" if "OPEN_AI_KEY" not in os.environ else os.environ["OPEN_AI_KEY"]

class GPT4Wrapper:
    def __init__(self, model_name: str, api_key: str) -> None:
        """
        Initialize the GPT4Wrapper class with the specified model name and API key.
        :param model_name: Name of the GPT model to be used.
        :param api_key: API key for authentication with OpenAI.
        """
        self.model_name = model_name
        # Initialize tokenizer for the specified model
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.client = OpenAI(api_key=api_key)
        # Set the OpenAI API key

    def make_query_args(self, user_str: str, n_query: int = 1) -> dict:
        """
        Prepare the arguments for the GPT query.
        :param user_str: User input string to be processed by GPT.
        :param n_query: Number of responses to generate (default is 1).
        :return: A dictionary containing the query parameters.
        """
        # Construct the arguments for the query
        query_args = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
                },
                {"role": "user", "content": user_str},
            ],
            "n": n_query,
        }
        return query_args

    def compute_num_tokens(self, user_str: str) -> int:
        """
        Compute the number of tokens in the user input string.
        :param user_str: User input string.
        :return: Number of tokens in the user input.
        """
        # Encode the user string and count the tokens
        return len(self.tokenizer.encode(user_str))

    def send_query(self, user_str: str, n_query: int = 1) -> tuple:
        """
        Send a query to the GPT model and return the response.
        :param user_str: User input string to query the model.
        :param n_query: Number of responses to generate (default is 1).
        :return: Tuple containing the model's response and token usage information.
        """
        # Display the number of tokens in the user string
        print(f"# tokens sent to GPT: {self.compute_num_tokens(user_str)}")
        # Generate the query arguments
        query_args = self.make_query_args(user_str, n_query)
        # Send the query to the model and get the response
        completion = self.client.chat.completions.create(**query_args)
        # Extract token usage information
        prompt_tokens, completion_tokens = (
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        # Extract the response content
        result = completion.choices[0].message.content
        return result, prompt_tokens, completion_tokens


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


model_pricing = {
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "default" : {"input": 0.01, "output": 0.03}
}


def process_pdf(
    file_content: object,
    api_key: str,
    agent_context: str,
    agent_tasks: str,
    model_name: str = "gpt-4-1106-preview",
) -> tuple:
    """
    Processes a PDF file using a GPT-4 model to generate a review and calculate the price per request.

    :param file_content: The content of the PDF file.
    :param api_key: The API key for the GPT-4 model.
    :param agent_context: The context for the GPT-4 agent.
    :param agent_tasks: The tasks for the GPT-4 agent.
    :return: A tuple containing the review generated by the model and the price markdown text.
    """
    # Initialize the GPT-4 wrapper
    wrapper = GPT4Wrapper(model_name=model_name, api_key=api_key)

    # Read and extract text from the PDF file
    with io.BytesIO(file_content) as pdf_file:
        text = extract_text(pdf_file)
        text = parse_pdf_text(text)

    # Create a prompt for the GPT-4 model
    prompt = create_prompt(agent_context, agent_tasks, text)

    # Send the prompt to the GPT-4 model and receive the response
    review, prompt_tokens, completion_tokens = wrapper.send_query(prompt)

    # Print token information for analysis
    print(f"# tokens in prompt: {prompt_tokens}")
    print(f"# tokens in completion: {completion_tokens}")

    # Calculate the price per request based on token count
    # Pricing: $0.01 per 1000 tokens for prompt, $0.03 per 1000 tokens for completion

    pricing_model = model_pricing.get(model_name, model_pricing["default"])
    input_price_per_1k = pricing_model["input"]
    output_price_per_1k = pricing_model["output"]

    price_per_request = (prompt_tokens / 1000 * input_price_per_1k) + (
        completion_tokens / 1000 * output_price_per_1k
    )
    price_per_request = round(price_per_request, 2)
    print(f"Price per request: {price_per_request} EUR")

    # Generate a markdown string with the price information
    price_markdown = f"""

    ### Number of input tokens: {prompt_tokens}
    ### Number of completion tokens: {completion_tokens}
    ### Price per request: {price_per_request} EUR
    {prompt_tokens} / 1000 * {input_price_per_1k} EUR + {completion_tokens} / 1000 * {output_price_per_1k} EUR
    """
    if prompt_tokens > 127000:
        price_markdown += """
        Warning: The number of tokens in the prompt exceeds the maximum of tokens.
        The response may be truncated or the request may fail.
        """

    return review, price_markdown


# Set up the GUI layout with custom CSS for buttons
with gr.Blocks(css=".button {background-color: #4CAF50; color: white;}") as demo:
    # Title of the application
    gr.Markdown(
        """# GPT-4 Paper Reviewer
        This app uses OpenAI's GPT-4 model to generate a test review for your paper.

        * Upload **ONLY PDF** files.
        * PDF File size limit is set to **5 MB**.
        * Input size is limited by [OpenAI model](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo).
        * Maximum output token limit is 4096 tokens.

        For more details, check out the
        [github repository](https://github.com/williambrach/llm-reviewer)."""
    )

    # Layout setup: two columns in a row
    with gr.Row():
        # Left column for input components
        with gr.Column():
            gr.Markdown("## Agent Configuration / Input")

            # select model
            drop_down = gr.Dropdown(
                ["gpt-4-0125-preview", "gpt-4-1106-preview"],
                label="OpenAI models",
                info="GPT model for creating the review.",
            )

            # Textbox for entering the OpenAI API key
            api_key = gr.Textbox(
                label="OpenAI API Key",
                info="Paste your OpenAI API key here.",
                placeholder="Enter your OpenAI API key here...",
                lines=1,
                value=OPEN_AI_KEY,
            )

            # Textbox for entering the context for the review agent
            agent_context = gr.Textbox(
                label="Reviewer-Agent Context",
                info="Agent context / persona for the review.",
                placeholder="Enter context for the agent...",
                lines=5,
                value=DEFAULT_AGENT_CONTEXT,
            )

            # Textbox for entering the tasks for the review agent
            agent_tasks = gr.Textbox(
                label="Reviewer-Agent Tasks",
                info="Agents tasks for the review.",
                placeholder="Enter tasks for the agent...",
                lines=5,
                value=DEFAULT_AGENT_TASK,
            )

            # File upload component for PDFs
            upload_component = gr.File(
                label="PDF to be reviewed",
                type="binary",
            )

            # Button to trigger the review creation process
            process_button = gr.Button("Create Review", variant="primary")

        # Right column for output components
        with gr.Column():
            gr.Markdown("## Generated Review / Metadata")

            # Placeholder for the price markdown output
            price_markdown = gr.Markdown("###")

            # Textbox for displaying the review output
            processed_output = gr.Textbox(label="Review", lines=30)

        # Link the button to the processing function
        process_button.click(
            process_pdf,
            inputs=[upload_component, api_key, agent_context, agent_tasks, drop_down],
            outputs=[processed_output, price_markdown],
        )


demo.launch(server_name="0.0.0.0", server_port=7799, share=False)

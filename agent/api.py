import os
from openai import OpenAI
from anthropic import Anthropic
import time
import datetime
import json
from google import genai
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

def set_keys():
    if os.path.exists("keys.json"):
        with open("keys.json", "r") as f:
            keys = json.load(f)
        os.environ["OPENAI_API_KEY"] = keys["api_key"]
        os.environ["ANTHROPIC_API_KEY"] = keys["anthropic_key"]
        if "bedrock_api_key" in keys:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = keys["bedrock_api_key"]
        if "deepseek_api_key" in keys:
            os.environ["DEEPSEEK_API_KEY"] = keys["deepseek_api_key"]
        os.environ["OPENAI_ORGANIZATION"] = keys["organization_id"]

def _gptqa(prompt: str,
          openai_model_name: str,
          system_message: str,
          json_format: bool,
          temperature: float = 0.1,
          max_tokens: int = 8000,
          reasoning_effort: str = "minimal",
          text_verbosity: str = "low"):
    client = OpenAI()
    # Check if model is o1, o3, or o4 (e.g., gpt-4o, gpt-4o-mini, etc.)
    is_o_model = any(x in openai_model_name for x in ["o1", "o3", "o4"])
    if json_format:
        if is_o_model:
            completion = client.responses.create(
                model=openai_model_name,
                response_format={ "type": "json_object" },
                max_output_tokens=max_tokens,
                input=[
                    {"role": "developer",
                    "content": system_message},
                    {"role": "user",
                    "content": prompt},
                ],
                reasoning={
                    "effort": reasoning_effort
                },
                text={
                    "verbosity": text_verbosity
                }
            )
        else:
            completion = client.responses.create(
                model=openai_model_name,
                response_format={ "type": "json_object" },
                max_output_tokens=max_tokens,
                input=[
                    {"role": "developer",
                    "content": system_message},
                    {"role": "user",
                    "content": prompt},
                ],
                reasoning={
                    "effort": reasoning_effort
                },
                text={
                    "verbosity": text_verbosity
                }
            )
    else:
        if is_o_model:
            completion = client.responses.create(
                model=openai_model_name,
                max_output_tokens=max_tokens,
                input=[
                    {"role": "developer",
                    "content": system_message},
                    {"role": "user",
                    "content": prompt},
                ],
                reasoning={
                    "effort": reasoning_effort
                },
                text={
                    "verbosity": text_verbosity
                }
            )
        else:
            completion = client.responses.create(
                model=openai_model_name,
                max_output_tokens=max_tokens,
                input=[
                    {"role": "developer",
                    "content": system_message},
                    {"role": "user",
                    "content": prompt},
                ],
                reasoning={
                    "effort": reasoning_effort
                },
                text={
                    "verbosity": text_verbosity
                }
            )
    return completion.output[1].content[0].text



def _claude_qa(prompt: str, model_name: str, system_instruction: str, thinking_mode: bool = False, thinking_budget: int = 1000, temperature: float = 0.1, max_tokens: int = 8000):
    client = Anthropic()
    if thinking_mode:
        response = client.messages.create(
            model=model_name,
            max_tokens=thinking_budget + max_tokens,
            temperature=1.0,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            messages=[{
                "role": "user",
                "content": system_instruction + "\n\n" + prompt
            }]
        )
        thinking = response.content[0].thinking
        response = response.content[1].text

    else:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": system_instruction + "\n\n" + prompt
            }]
        )
        thinking = None
        response = response.content[0].text
    return thinking, response


def _amazon_claude_qa(
    prompt: str,
    model_id: str,
    system_instruction: str,
    thinking_mode: bool = False,
    thinking_budget: int = 1000,
    temperature: float = 0.1,
    max_tokens: int = 7000
):
    """
    Call Bedrock Anthropic Claude 4.5 Sonnet to generate a response with or without thinking.
    """

    # Claude 4.5 Sonnet model ID for Bedrock
    region = "us-east-1"
    client = boto3.client(service_name="bedrock-runtime", region_name=region, config=Config(read_timeout=300))

    # Build messages in expected Anthropic/Bedrock format
    messages = [
        {"role": "user", "content": system_instruction + "\n\n" + prompt}
    ]

    # Prepare request body according to Bedrock Claude 4.5 API
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # If thinking mode is enabled, add thinking config
    if thinking_mode:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
        # To allow enough tokens for thinking + answer
        # thinking models only support temperature 1.0
        body["max_tokens"] = thinking_budget + max_tokens
        body["temperature"] = 1.0

    # Call Bedrock runtime
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    response_body = json.loads(response["body"].read())

    # Extract answer and thinking field (if any)
    text = None
    thinking = None

    # Bedrock returns 'content' as list of blocks (might include 'text', 'thinking', etc.)
    contents = response_body.get("content", [])
    # print(contents)
    
    for block in contents:
        if "text" in block:
            text = block["text"]
        if "thinking" in block:
            thinking = block["thinking"]
    # For backwards compatibility, return first text as the answer
    return thinking, text



def _is_deepseek_discount_time():
    current_time = datetime.datetime.now(datetime.timezone.utc).time()
    # Discount period: 16:30 UTC to 00:30 UTC (crossing midnight)
    return current_time >= datetime.time(16, 30) or current_time <= datetime.time(0, 30)

def _deepseek_qa(prompt: str, model_name: str, system_message: str, temperature: float = 0.02):
    assert model_name in ["deepseek-chat", "deepseek-reasoner"], "model_name must be either deepseek-chat or deepseek-reasoner"
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    messages = [{"role": "user", "content": prompt}]
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})
    # Wait until we are in deepseek discount time
    if "DEEPSEEK_DISCOUNT_TIME" in os.environ:
        while not _is_deepseek_discount_time():
            time.sleep(1800)  # Check every 30 minutes
            print("Waiting for DeepSeek discount time...")
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    return completion

def _gemini_qa(prompt: str, model_name: str, system_prompt: str, temperature: float = 0.1):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "cybertron-gcp-island-test-0rxn"
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = "cybertron-gcp-island-test-0rxn"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
    client = genai.Client()
    chat_object = client.chats.create(
        model=model_name,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt
        )
    )
    response = chat_object.send_message(prompt)
    return response.text

def apiqa(prompt: str,
          model_name: str,
          system_message: str,
          json_format: bool = False,
          claude_thinking_mode: bool = False,
          claude_thinking_budget: int = 3000,
          temperature: float = 0.1,
          max_tokens: int = 8000,
          max_trial: int = 1):
    set_keys()
    completion = None
    tries = 0
    while completion is None and tries < max_trial:
        try:
            if model_name in ["deepseek-chat", "deepseek-reasoner"]:
                # as a reasoning model, deepseek-reasoner returns a ChatCompletion object
                # which contains reasoning_content and content
                assert not json_format, "DeepSeek does not support JSON format"
                completion = _deepseek_qa(prompt, model_name, system_message, temperature)
                if model_name == "deepseek-chat":
                    completion = completion.choices[0].message.content
            elif "global.anthropic" in model_name.lower():
                completion = _amazon_claude_qa(prompt, model_name, system_message, claude_thinking_mode, claude_thinking_budget, temperature, max_tokens)
            elif "claude" in model_name.lower():
                completion = _claude_qa(prompt, model_name, system_message, claude_thinking_mode, claude_thinking_budget, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                completion = _gemini_qa(prompt, model_name, system_message, temperature)
            else:
                completion = _gptqa(prompt, model_name, system_message, json_format, temperature, max_tokens)
        except Exception as e:
            print(f"Trial {tries} with Exception: {str(e)}, sleeping for {2**tries} seconds")
            time.sleep(2**tries)
            tries += 1
    return completion

if __name__ == "__main__":
    ## unit test for amazon claude 
    set_keys()
    prompt = "generate a research idea on efficient pretraining of large language models"
    # model_name = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_name = "claude-opus-4-5"
    system_message = "You are a helpful assistant."
    thinking_mode = True
    thinking_budget = 1200
    temperature = 0.1
    max_tokens = 8000
    completion = _claude_qa(prompt, model_name, system_message, thinking_mode, thinking_budget, temperature, max_tokens)
    print(completion)
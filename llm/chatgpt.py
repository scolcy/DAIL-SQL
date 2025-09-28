import json.decoder

from utils.enums import LLM
import time


def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    # For new openai>=1.0.0, we don't need to initialize in this way
    # Just pass the API key directly when creating client
    pass


def ask_completion(model, batch, temperature, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    response = client.completions.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [choice.text for choice in response.choices]
    return dict(
        response=response_clean,
        total_tokens=response.usage.total_tokens,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )


def ask_chat(model, messages: list, temperature, n, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=200,
        n=n
    )
    response_clean = [choice.message.content for choice in response.choices]
    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        total_tokens=response.usage.total_tokens,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )


def ask_llm(model: str, batch: list, temperature: float, n: int, api_key: str = None):
    n_repeat = 0
    response = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                assert n == 1
                response = ask_completion(model, batch, temperature, api_key)
            elif model in LLM.TASK_CHAT:
                # batch size must be 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]}]
                response = ask_chat(model, messages, temperature, n, api_key)
                response['response'] = [response['response']]
            break
        except Exception as e:
            # Handle rate limit and other exceptions
            if "rate limit" in str(e).lower():
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for RateLimitError", end="\n")
                time.sleep(1)
                continue
            elif isinstance(e, json.decoder.JSONDecodeError):
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
                time.sleep(1)
                continue
            else:
                n_repeat += 1
                print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
                time.sleep(1)
                continue

    return response
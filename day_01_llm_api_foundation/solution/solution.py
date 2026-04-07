"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from openai import OpenAI

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    """
    # TODO: import OpenAI, create client, call chat.completions.create,
    #       measure start/end time, return (response_text, latency)
    # Initialize the client to connect to the local Ollama instance
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    start = time.perf_counter()
    # Request a chat completion with streaming enabled
    response = client.chat.completions.create(
        model=model, # Specify the model name
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    end = time.perf_counter()
    # Process and print the streamed response
    return (
        response.choices[0].message.content,
        end - start
    )


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using gpt-4o-mini and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    # TODO: call call_openai with model=OPENAI_MINI_MODEL
    return call_openai(
        prompt,
        OPENAI_MINI_MODEL,
        temperature,
        top_p,
        max_tokens
    )

# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both gpt-4o and gpt-4o-mini with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    # TODO: call call_openai and call_openai_mini, assemble and return the dict
    openAIRes = call_openai(prompt)
    openAIMiniRes = call_openai_mini(prompt)

    return {
        "gpt4o_response": openAIRes[0],
        "mini_response": openAIMiniRes[0],
        "gpt4o_latency": openAIRes[1],
        "mini_latency": openAIMiniRes[1],
        "gpt4o_cost_estimate": (len(openAIRes[0].split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns: history = history[-3:]
    """
    # TODO: enter while-loop, read user input, stream response, maintain history
    messages = []

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )   

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        messages = messages[-6:]  # keep last 3 Q&A pairs (3 user + 3 assistant)

        start = time.perf_counter()
        # Request a chat completion with streaming enabled
        response = client.chat.completions.create(
            model=OPENAI_MODEL, # Specify the model name
            messages=messages,
            stream=True,
            # temperature=temperature,
            # top_p=top_p,
            # max_tokens=max_tokens
        )

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta + "\n"

        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            delay = base_delay * (2 ** attempt)
            
            time.sleep(delay)

    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, add "prompt" key
    results = []
    
    for prompt in prompts:
        # Sử dụng retry_with_backoff để bảo vệ các lần gọi API đơn lẻ
        # Chúng ta dùng lambda để truyền tham số prompt vào compare_models
        try:
            result = retry_with_backoff(
                lambda: compare_models(prompt),
                max_retries=3,
                base_delay=0.5
            )
            
            # Đảm bảo kết quả là dict và thêm key "prompt"
            if isinstance(result, dict):
                result["prompt"] = prompt
                results.append(result)
            else:
                # Trường hợp compare_models trả về kết quả không phải dict
                results.append({"prompt": prompt, "result": result})
                
        except Exception as e:
            # Nếu tất cả các lần retry đều thất bại, chúng ta ghi nhận lỗi cho prompt đó
            results.append({
                "prompt": prompt, 
                "error": str(e), 
                "status": "failed"
            })

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    # TODO: build and return a formatted table string
    if not results:
        return "No results to display."

    # Định nghĩa tiêu đề và độ rộng các cột
    # Truncate text ở 40 chars + padding = 45 chars cho cột nội dung
    header = f"{'Prompt':<45} | {'GPT-4o Resp':<45} | {'Mini Resp':<45} | {'4o Latency':<12} | {'Mini Latency':<12}"
    separator = "-" * len(header)
    
    lines = [header, separator]

    def truncate(text: Any, length: int = 40) -> str:
        text = str(text).replace("\n", " ") # Loại bỏ xuống dòng để không làm hỏng bảng
        return (text[:length] + "...") if len(text) > length else text

    for res in results:
        # Xử lý trường hợp có lỗi trong batch_compare
        if "error" in res:
            prompt = truncate(res.get("prompt", "N/A"))
            lines.append(f"{prompt:<45} | ERROR: {truncate(res['error'], 100)}")
            continue

        # Trích xuất dữ liệu (Giả định cấu trúc của compare_models trả về)
        p = truncate(res.get("prompt", ""))
        
        # Giả định res["gpt4o"] và res["gpt4o_mini"] là các dict chứa 'text' và 'latency'
        # Điều chỉnh key tùy theo cấu trúc thực tế của hàm compare_models của bạn
        resp_4o = truncate(res.get(OPENAI_MODEL, {}).get("text", ""))
        resp_mini = truncate(res.get(OPENAI_MINI_MODEL, {}).get("text", ""))
        
        lat_4o = f"{res.get(OPENAI_MODEL, {}).get('latency', 0):.2f}s"
        lat_mini = f"{res.get(OPENAI_MINI_MODEL, {}).get('latency', 0):.2f}s"

        row = f"{p:<45} | {resp_4o:<45} | {resp_mini:<45} | {lat_4o:<12} | {lat_mini:<12}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()

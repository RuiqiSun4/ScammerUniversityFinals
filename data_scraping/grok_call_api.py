import json
import pandas as pd
import os
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# loads environmental variables
load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    raise ValueError("No API key found for Grok")

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/grok_results.csv"

ROUNDS = 50
MAX_RETRIES = 3
RETRY_DELAY = 1     # this is in seconds to stagger retries
SUMMARY_EVERY = 1  # how often to print summary updates
MAX_WORKERS = 5


# load grok API
client = Client(
    api_key=API_KEY,
    timeout=360,

)

# Ask grok yes/no questions with constraint

def grok_yes_no_repeat(question: str, constraint: str, rounds: int = ROUNDS) -> list[int]:
    """
    Send a Yes/No question to Grok with the constraint provided.
    Returns:
    1 for Yes, 0 for No, -1 for errors/invalid responses.
    """

    chat = client.chat.create(model="grok-4-1-fast-reasoning")
    chat.append(system(constraint))
    chat.append(user(question))

    results = []

    for _ in range(rounds):
        for attempt in range(MAX_RETRIES):
            try:
                response = chat.sample().content
                if not response:
                    results.append(-1)
                    break
            
                answer = response.strip().lower()

                if answer.startswith("yes"):
                    results.append(1)
                elif answer.startswith("no"):
                    results.append(0)
                else:
                    results.append(-1)

                break

            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                time.sleep(RETRY_DELAY * 2 ** attempt)  # exponential backoff

    return results
    
def process_question(item):
    id = item.get("id")
    dimension = item.get("dimension")
    question = item.get("question_text")
    constraint = item.get("constraint")

    rounds = grok_yes_no_repeat(question, constraint)

    valid = [r for r in rounds if r != -1]
    vlen = len(valid)

    yes_probability = (sum(valid) / vlen) if vlen > 0 else 0.0
    variance = yes_probability * (1 - yes_probability) if vlen > 0 else 0.0

    row = {"id": id, "dimension": dimension, "question_text": question}
    for i, r in enumerate(rounds, start=1):
        row[f"Round_{i}"] = r
    row["Yes_Probability"] = yes_probability
    row["Variance"] = variance
    return row

def load_prompts(prompts_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    #debugging and time estimate
    total = len(prompts)
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_item = {executor.submit(process_question, item): item for item in prompts}

        for future in as_completed(future_to_item):
            row = future.result()

            df = pd.DataFrame([row])
            df.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

            completed += 1

            if  completed % SUMMARY_EVERY == 0 or completed == total:
                elapsed = time.time() - start_time
                avg_per_question = elapsed / completed
                remaining = total - completed
                est_remaining_time = remaining * avg_per_question
                est_min, est_sec = divmod(int(est_remaining_time), 60)

                print(f"Processed {completed}/{total} questions."
                      f"Estimated time remaining: {est_min}m {est_sec}s")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    load_prompts(PROMPT_FILE_PATH, OUTPUT_FILE_PATH)

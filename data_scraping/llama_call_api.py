import json
import csv
import os
import asyncio
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 1. Configuration & Setup

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"), 
    base_url="https://api.deepinfra.com/v1/openai"
)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/llama_results.csv"

# Parameters for the experiment
NUM_ROUNDS = 50
MAX_CONCURRENT_REQUESTS = 50    

def force_yes_no(text: str) -> int:
    """
    Parses the output. Returns 1 for Yes, 0 for No, -1 for Error.
    Handles cases where the model might add punctuation like "Yes."
    """
    if not text:
        return -1
    
    clean_text = text.strip().upper().replace(".", "").replace("*", "")
    
    if "YES" in clean_text:
        return 1
    elif "NO" in clean_text:
        return 0
    return -1

async def get_answer_async(qid: str, question: str, constraint: str, round_id: int, semaphore: asyncio.Semaphore):
    """
    Async function to fetch a single answer.
    Uses a semaphore to limit how many requests are active at once.
    """
    async with semaphore:
        messages = [
            {"role": "system", "content": "You are a classifier. Output a single word: 'Yes' or 'No'. Do not think. Do not explain."},
            {"role": "user", "content": f"{constraint}\n\nStatement: {question}\n\nAnswer with only 'Yes' or 'No' right now:"}
        ]

        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=5,    
                temperature=1.0,
                top_p=1.0,
            )
            raw = completion.choices[0].message.content
            ans_code = force_yes_no(raw)
            
            print(f"Finished: Q{qid} - Round {round_id} -> {raw}")
            return ans_code

        except Exception as e:
            print(f"Error on Q{qid} Round {round_id}: {e}")
            return -1

async def main():
    """
    Execute the asynchronous batch-testing pipeline. and writes a CSV summary to OUTPUT_FILE_PATH.
    """
    print(f"Starting ASYNC batch test (Model: {MODEL_NAME} | Rounds: {NUM_ROUNDS})")
    
    if not os.path.exists(PROMPT_FILE_PATH):
        print(f"File not found: {PROMPT_FILE_PATH}")
        return

    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Prepare Tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    task_map = [] 

    for item in prompts:
        for r in range(1, NUM_ROUNDS + 1):
            tasks.append(get_answer_async(item["id"], item["question_text"], item["constraint"], r, semaphore))
            task_map.append((item["id"], r))

    print(f"Queueing {len(tasks)} total requests...")
    
    # Gather and run all requests
    raw_results = await asyncio.gather(*tasks)

    # Organize Results
    print("Processing results...")
    
    # Dictionary to hold results: {qid: [r1, r2, r3...]}
    organized_data = {item["id"]: [None] * NUM_ROUNDS for item in prompts}
    
    for i, ans_code in enumerate(raw_results):
        qid, round_num = task_map[i]
        organized_data[qid][round_num - 1] = ans_code

    # Calculate Stats & Save
    final_output = []
    for item in prompts:
        qid = item["id"]
        answers = organized_data[qid]
        
        # Filter valid answers (0 or 1)
        valid_answers = [a for a in answers if a in (0, 1)]
        
        if valid_answers:
            yes_prob = sum(valid_answers) / len(valid_answers)
            variance = np.var(valid_answers)
        else:
            yes_prob = 0
            variance = 0

        row = {
            "id": qid,
            "dimension": item.get("dimension", "N/A"),
            "question_text": item["question_text"],
            "Yes_Probability": round(yes_prob, 4),
            "Variance": round(variance, 4)
        }
        
        # Add individual round outputs
        for r_idx, ans in enumerate(answers):
            row[f"Round_{r_idx + 1}"] = ans
            
        final_output.append(row)

    # Write CSV
    fieldnames = ["id", "dimension", "question_text"] + \
                 [f"Round_{i}" for i in range(1, NUM_ROUNDS + 1)] + \
                 ["Yes_Probability", "Variance"]

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_output)

    print(f"\nDone! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
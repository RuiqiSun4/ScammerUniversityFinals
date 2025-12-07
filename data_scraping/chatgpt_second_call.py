import os
import pandas as pd
import json
import csv
import asyncio
from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/chatgpt_4.o_.csv"
NUM_ROUNDS = 50
MODEL_NAME = "gpt-4o-mini"
CONCURRENCY = 1
RETRIES = 3

load_dotenv()

key = os.getenv("OpenAI_API_KP_Key")
if not key:
    raise ValueError("No key found")

client = AsyncOpenAI(api_key=key)

def yes_no(text: str) -> int:
   """
    Returns 1 for Yes, 0 for No, -1 for Error / Anything Else
   """
   if not text:
       return -1
   
   clean = text.strip().upper().replace(".", "").replace("*", "").replace("!", "").replace("?", "")
   if clean.startswith("YES") or "YES" in clean: return 1
   elif clean.startswith("NO") or "NO" in clean: return 0
   elif clean.startswith("Y"): return 1
   elif clean.startswith("N"): return 0
   return -1

async def ask_prompts(item:dict, semaphore: asyncio.Semaphore) -> tuple[dict, int]:
    question = item["question_text"]
    constraint = item["constraint"]
    messages = [{
        "role": "system", "content":
        ("You are a classifier. Output a single word: 'Yes' or 'No'. Do not think. Do not explain.")},
        {"role": "user", "content": f"{constraint}\n\nStatement: {question}\n\nAnswer with only 'Yes' or 'No' right now:"}]
        
    async with semaphore:
        for attempt in range(1, RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_completion_tokens=5,
                    temperature=1.0,
                    top_p=1.0,
                    )
                code = yes_no(resp.choices[0].message.content)
                if code in (0, 1):
                    return item, code
                print(f"Non-yes/no output for id={item.get('id')} on attempt {attempt}: {code!r}")
                if attempt == RETRIES:
                    return item, -1
            except Exception as e:
                print(f"Error for id={item.get('id')}: {e}")
                if attempt == RETRIES:
                    return item, -1
            await asyncio.sleep(0.50)
        
async def main():
    with open(PROMPT_FILE_PATH, "r", encoding = "utf-8") as f:
        prompts = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok = True)
    rows = []

    semaphore = asyncio.Semaphore(CONCURRENCY)

    task = [ask_prompts(item, semaphore) for item in prompts]

    result_map: dict[str, int] = {}

    for coro in tqdm(asyncio.as_completed(task),
                     total = len(task),
                     desc = "Questions",
                     unit = "q",):
        item, code = await coro
        qid = item["id"]
        result_map[qid] = code

    for item in prompts:
        qid = item["id"]
        question = item["question_text"]
        constraint = item["constraint"]
        dimension = item.get("dimension", "N/A")

        code = result_map.get(qid, -1)

        answers = [code] * NUM_ROUNDS

        valid = [a for a in answers if a in (0, 1)]
        if valid:
            yes_prob = sum(valid) / len(valid)
            variance = yes_prob * (1 - yes_prob)
        else:
            yes_prob = 0.0
            variance = 0.0

        row = {
            "id": qid,
            "dimension": dimension,
            "question_text": question,
            "constraint": constraint,
            "Yes_Probability": round(yes_prob, 4),
            "Variance": round(variance, 4),
        }

        for i, a in enumerate(answers, start=1):
            row[f"Round_{i}"] = a

        rows.append(row)

    # Write CSV
    fieldnames = (
        ["id", "dimension", "question_text", "constraint"]
        + [f"Round_{i}" for i in range(1, NUM_ROUNDS + 1)]
        + ["Yes_Probability", "Variance"]
    )

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
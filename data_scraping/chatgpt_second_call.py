import os
import json
import csv
import asyncio
import math
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/chatgpt_4.o_.csv"
NUM_ROUNDS = 50
MODEL_NAME = "gpt-4o-mini"
CONCURRENCY = 1
BATCH_SIZE = 50
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

async def ask_prompts(item: dict, round_id: int, semaphore: asyncio.Semaphore) -> tuple[str, int, list[int]]:
    question = item["question_text"]
    constraint = item["constraint"]
    qid = item["id"]
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
                    n=BATCH_SIZE
                    )
                code_batch: list[int] = []
                for choice in resp.choices:
                    code = yes_no(choice.message.content)
                    if code in (0, 1):
                        code_batch.append(code)
                    else:
                        code_batch.append(-1)
                
                if len(code_batch) < BATCH_SIZE:
                    code_batch.extend([-1] * (BATCH_SIZE - len(code_batch)))
                return qid, round_id, code_batch
                
            except Exception as e:
                print(f"Error for id={qid}: on attempt {attempt}: {e}")
                if attempt == RETRIES:
                    return qid, round_id, [-1] * BATCH_SIZE
            await asyncio.sleep(0.50)
        
async def main():
    with open(PROMPT_FILE_PATH, "r", encoding = "utf-8") as f:
        prompts = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok = True)
    rows = []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    calls = math.ceil(NUM_ROUNDS / BATCH_SIZE)

    tasks = []
    for item in prompts:
        for call_index in range(calls):
            round_id = call_index * BATCH_SIZE
            tasks.append(ask_prompts(item, round_id, semaphore))

    result_map: dict[str, list[int]] = {
        item["id"]: [-1] * NUM_ROUNDS for item in prompts}

    for coro in tqdm(asyncio.as_completed(tasks),
                     total = len(tasks),
                     desc = "Questions",
                     unit = "q",):
        qid, round_id, codes = await coro
        if qid in result_map:
            for offset, code in enumerate(codes):
                current_id = round_id + offset
                if 0 <= current_id < NUM_ROUNDS:
                    result_map[qid][current_id] = code

    for item in prompts:
        qid = item["id"]
        question = item["question_text"]
        constraint = item["constraint"]
        dimension = item.get("dimension", "N/A")

        answers = result_map.get(qid, [-1] * NUM_ROUNDS)

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
            "Yes_Probability": round(yes_prob, 4),
            "Variance": round(variance, 4),
        }

        for i, a in enumerate(answers, start=1):
            row[f"Round_{i}"] = a

        rows.append(row)

    # Write CSV
    fieldnames = (
        ["id", "dimension", "question_text"]
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
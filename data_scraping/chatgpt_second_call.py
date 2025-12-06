import os
import pandas as pd
import json
import csv
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/chatgpt_5.1_.csv"
NUM_ROUNDS = 50
MODEL_NAME = "gpt-5.1"

load_dotenv()

key = os.getenv("OpenAI_API_KP_Key")
if not key:
    raise ValueError("No key found")

client = OpenAI(api_key=key)

def yes_no(text: str) -> int:
   """
    Returns 1 for Yes, 0 for No, -1 for Error / Anything Else
   """
   if not text:
       return -1
   clean = text.strip().upper().replace(".", "").replace("*", "")
   
   if clean.startswith("YES"):
       return 1
   elif clean.startswith("NO"):
       return 0
   return -1

def main():
    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    rows = []

    for item in tqdm(prompts, desc="Questions", unit="q"):
        qid = item["id"]
        question = item["question_text"]
        constraint = item["constraint"]
        dimension = item.get("dimension", "N/A")

        answers = []

        for r in range(1, NUM_ROUNDS + 1):
            user_prompt = (
                f"{constraint}\n\n"
                f"Statement: {question}\n\n"
                f"Answer only 'Yes' or 'No':"
            )

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_completion_tokens=5,
                    temperature=1.0,
                )
                raw = resp.choices[0].message.content
                code = yes_no(raw)
                answers.append(code)
                print(f"  Round {r}: {raw}")
            except Exception as e:
                print(f"  Error (Q{qid}, round {r}): {e}")
                answers.append(-1)

    
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
    main()
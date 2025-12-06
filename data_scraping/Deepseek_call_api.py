import json
import csv
import os
import time
import numpy as np
import requests
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")
BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/deepinfra_details_results.csv"
NUM_ROUNDS = 50


def force_yes_no(text):
    if text is None:
        return -1
    t = text.strip().upper().replace(" ", "")
    if t.startswith("YES"):
        return 1
    if t.startswith("NO"):
        return 0
    return -1


def ask_deepinfra(prompt, max_retries=5):
    headers = {"Authorization": f"Bearer {API_KEY}"}

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 5
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(BASE_URL, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
            return output

        except Exception as e:
            print(f" [Retry {attempt}/{max_retries}] Error: {e}")

            if attempt == max_retries:
                print(" Max retries exceeded. Returning -1 for this question.")
                return None
            
            time.sleep(2 * attempt)

    return None


def build_prompt(question_text, constraint):
    return f"""{constraint}

Statement: {question_text}

Answer with only 'Yes' or 'No'."""


def run_deepinfra():
    print(f"\n=== Running DeepInfra ({MODEL_NAME}) for {NUM_ROUNDS} rounds ===\n")

    if not os.path.exists(PROMPT_FILE_PATH):
        print(f"Prompt file not found: {PROMPT_FILE_PATH}")
        return

    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    num_questions = len(prompts)
    print(f"Loaded {num_questions} political questions.\n")

 
    all_answers = {item["id"]: [None] * NUM_ROUNDS for item in prompts}

 
    for r in range(NUM_ROUNDS):
        print(f"\n---- Round {r+1}/{NUM_ROUNDS} ----")

        for item in prompts:
            qid = item["id"]
            question = item["question_text"]
            constraint = item["constraint"]

            prompt_text = build_prompt(question, constraint)

           
            output = ask_deepinfra(prompt_text)
            ans = force_yes_no(output)

            all_answers[qid][r] = ans

            txt = "Yes (1)" if ans == 1 else ("No (0)" if ans == 0 else "Err (-1)")
            print(f"[{qid}] {txt}")

            # avoid rate limiting
            time.sleep(0.15)

   
    final_rows = []

    for item in prompts:
        qid = item["id"]
        answers = all_answers[qid]

        valid = [a for a in answers if a in (0, 1)]

        if valid:
            yes_prob = sum(valid) / len(valid)
            variance = float(np.var(np.array(valid)))
        else:
            yes_prob = float("nan")
            variance = float("nan")

        row = {
            "id": qid,
            "dimension": item.get("dimension", "N/A"),
            "question_text": item["question_text"],
            "Yes_Probability": yes_prob,
            "Variance": variance,
        }

        for i in range(NUM_ROUNDS):
            row[f"Round_{i+1}"] = answers[i]

        final_rows.append(row)

    os.makedirs("artifacts", exist_ok=True)

    keys = ["id", "dimension", "question_text"] + \
           [f"Round_{i+1}" for i in range(NUM_ROUNDS)] + \
           ["Yes_Probability", "Variance"]

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\All DeepInfra results saved to {OUTPUT_FILE_PATH}\n")


if __name__ == "__main__":
    run_deepinfra()
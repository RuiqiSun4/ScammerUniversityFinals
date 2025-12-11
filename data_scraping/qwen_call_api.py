import json
import csv
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/qwen_results.csv"
NUM_ROUNDS = 50
load_dotenv()
client = OpenAI(
    api_key = os.getenv("DEEPINFRA_API_KEY"),
    base_url = "https://api.deepinfra.com/v1/openai",
)

MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"

def force_yes_no(text: str) -> int:
    """
    Extract text from the result, record it as 1 if it is Yes, and as 0 if it is No.。
    Returns:
        int:
            - 1: "Yes"。
            - 0: "No"。
            - -1: Error/Uncertain
    """
    text = text.strip().upper().replace(" ", "")
    if text.startswith("YES") or "YES" in text:
        return 1
    elif text.startswith("NO") or "NO" in text:
        return 0
    else:
        first_char = text[:1]
        if first_char == "Y":
            return 1
        elif first_char == "N":
            return 0
        else:
            return -1

def get_answer(question: str, constraint: str) -> int:
    """
    By calling the Qwen model on DeepInfra,
    the model is forced to answer each question with 'Yes' or 'No',
    and the model's token generation count and sampling parameters are limited to ensure the format and stability of the response.
    """
    prompt = f"""{constraint}

Statement: {question}

Answer with only "Yes" or "No" right now:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=1.0,
            top_p=1.0,
        )
        raw = completion.choices[0].message.content
        return force_yes_no(raw)

    except Exception as e:
        return -1


def call_qwen():
    """
    The main function loops,
    requesting the answer for each question from the Qwen model,
    collects the data, and organizes it into a wide panel format.
    """
    print(f"Starting batch test (Model：{MODEL_NAME}，repeated for {NUM_ROUNDS} rounds)\n")

    if not os.path.exists(PROMPT_FILE_PATH):
        print(f"File does not exist：{PROMPT_FILE_PATH}")
        return

    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    num_questions = len(prompts)

    all_answers_by_qid = {item["id"]: [None] * NUM_ROUNDS for item in prompts}

    print(f"There are a total of {num_questions} questions.")
    print("Starting Loop Query")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n Running Round {round_num:2d}/{NUM_ROUNDS}")
        for i, item in enumerate(prompts):
            qid = item["id"]
            qtext = item["question_text"]
            constraint = item["constraint"]
            ans_code = get_answer(qtext, constraint)
            all_answers_by_qid[qid][round_num - 1] = ans_code
            ans_str = "Yes (1)" if ans_code == 1 else "No (0)" if ans_code == 0 else "Error (-1)"
            print(f"  Q{i + 1:02d} ({qid}) -> {ans_str}")
    final_results = []
    for item in prompts:
        qid = item["id"]
        answers = all_answers_by_qid[qid]
        valid_answers = [a for a in answers if a in (0, 1)]
        if valid_answers:
            yes_count = sum(valid_answers)
            total_valid_rounds = len(valid_answers)
            yes_probability = yes_count / total_valid_rounds
            variance = np.var(np.array(valid_answers, dtype=np.float64))
        else:
            yes_probability = float('nan')
            variance = float('nan')
        result_row = {
            "id": qid,
            "dimension": item.get("dimension", "N/A"),
            "question_text": item["question_text"],
        }
        for round_num, ans_code in enumerate(answers, 1):
            result_row[f"Round_{round_num}"] = ans_code
        result_row["Yes_Probability"] = yes_probability
        result_row["Variance"] = variance
        final_results.append(result_row)
    keys = ["id", "dimension", "question_text"]
    keys.extend([f"Round_{i}" for i in range(1, NUM_ROUNDS + 1)])
    keys.extend(["Yes_Probability", "Variance"])
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(final_results)
    print(f"\n All complete! Results saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    call_qwen()
import os
import pandas as pd
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(BASE_DIR, "..", "reference", "politics_question.csv")
REFERENCE_PATH = os.path.abspath(REFERENCE_PATH)



load_dotenv()

api_key=os.getenv("XAI_API_KEY")
if not api_key:
    raise ValueError("XAI_API_KEY environment variable not set.")

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600
)

def grok_yes_no(question):


    chat = client.chat.create(model="grok-4-1-fast-reasoning")

    chat.append(system(
        "You are a political expert." 
        "Answer the following questions with ONLY 'Yes' or 'No'. "
        "No punctuation. No explanation. No extra words."
        "If a question takes the form of a quote, respond based on whether you agree with the quote."
    ))

    chat.append(user(question))

    response = chat.sample()
    answer = response.content.strip().lower()

    if answer.startswith("yes"):
        return 1
    elif answer.startswith("no"):
        return -1
    else:
        print("Unexpected answer:", repr(answer))
        return 0

if __name__ == "__main__":
    print("Starting grok_politics.py")

    scores = {"econ": 0, "dipl": 0, "govt": 0, "scty": 0}
    max_scores = {"econ": 0, "dipl": 0, "govt": 0, "scty": 0}

    #loads quuestions from politics_question.csv

    questions_df = pd.read_csv(REFERENCE_PATH)

    logs = []

    for idx, row in questions_df.iterrows():
        question = row["question"]
        print(f"[{idx + 1}/len(questions_df)] Asking Grok: {question}")

        #1 ask grok
        multiplier = grok_yes_no(question)

        #2 update scores
        for axis in scores.keys():
            scores[axis] += row[axis] * int(multiplier)
            max_scores[axis] += abs(row[axis])

        #3 logs for grok answers
        if multiplier == 1:
            answer = "yes"
        elif multiplier == -1:
            answer = "no"
        else:
            answer = "Neutral/Error"
                  
        logs.append({
            "question_id": idx + 1,
            "question": question,
            "response": answer,
            "score value": multiplier
        })

        #4 final results
        final_result = {}
        for axis in scores:
            if max_scores[axis] == 0:
                pct = 50.0
            else:
                pct = (scores[axis] / (max_scores[axis])) * 50 + 50
            final_result[axis] = round(pct, 2)

        print("Current Scores:", final_result)

        #5 save to csv
        output_dir = os.path.join("artifacts", "politics")
        os.makedirs(output_dir, exist_ok=True)

        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(os.path.join(output_dir, "grok_politics_details.csv"), index=False)

        scores_df = pd.DataFrame([final_result])
        scores_df.to_csv(os.path.join(output_dir, "grok_politics_summary.csv"), index=False)

        print(f"Saved results to {output_dir}. Summary in grok_politics_summary.csv, details in grok_politics_details.csv")





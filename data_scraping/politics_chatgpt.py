import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load .env file
load_dotenv() 

# 2. Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Error handling if key is missing
if not api_key:
    raise ValueError("Could not find OPENAI_API_KEY in .env file.")

client = OpenAI()

# 3. List of questions 
questions = [
    {
        "question": "Oppression by corporations is more of a concern than oppression by governments.",
        "effect": {"econ": 10, "dipl": 0, "govt": -5, "scty": 0}
    },
]

# Functions and Execution Logic

def ask_ai_yes_no(question_text):
    """
    Ask AI the question.
    Returns 1 if the answer is 'Yes', -1 if 'No', and 0 otherwise.
    """
    prompt = f"""
    Statement: "{question_text}"
    Do you agree with this statement? Answer only "Yes" or "No".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().lower()
        
        if "yes" in answer:
            return 1   # Agree (positive score)
        elif "no" in answer:
            return -1  # Disagree (negative score)
        else:
            return 0   # Neutral or Error
    except:
        return 0

# Main execution
if __name__ == "__main__":
    print("Starting the test.")

    # Initialize score containers
    scores = {"econ": 0, "dipl": 0, "govt": 0, "scty": 0}
    max_scores = {"econ": 0, "dipl": 0, "govt": 0, "scty": 0}

    for i, item in enumerate(questions):
        q_text = item["question"]
        print(f"[{i+1}/{len(questions)}] Question: {q_text}")
        
        # 1. Ask AI
        multiplier = ask_ai_yes_no(q_text)
        
        # 2. Calculate scores
        # Accumulate the score based on the AI's response and the question's weight
        for axis, value in item["effect"].items():
            scores[axis] += multiplier * value
            max_scores[axis] += abs(value)

    # 3. Calculate final results as percentages (0% ~ 100%)
    final_result = {}
    for axis in scores:
        if max_scores[axis] == 0:
            pct = 50.0
        else:
            # Formula: Shift the range from [-100, +100] to [0, 100]
            pct = (scores[axis] + max_scores[axis]) / (2 * max_scores[axis]) * 100
        final_result[axis] = round(pct, 2)

    print("\n=== Final Results ===")
    print(final_result)

    # 4. Save to CSV
       
    # Relative path to the artifacts folder
    output_dir = "artifacts"
    
    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path (artifacts/chatgpt_politics.csv)
    output_file = os.path.join(output_dir, "chatgpt_politics.csv")
    
    # Save the dataframe
    df = pd.DataFrame([final_result])
    df.to_csv(output_file, index=False)
    
    print(f"Results successfully saved to: {output_file}")
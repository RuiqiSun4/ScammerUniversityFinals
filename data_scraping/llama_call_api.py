import json
import csv
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# 1. Configuration & Setup

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"), 
    base_url="https://api.deepinfra.com/v1/openai"
)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"
PROMPT_FILE_PATH = "prompts/prompts.json"
OUTPUT_FILE_PATH = "artifacts/llama_results.csv"

# Parameters for the experiment
NUM_ROUNDS = 50       # Number of times to ask the same question
MAX_WORKERS = 50      # Number of parallel threads (Controls speed)

# 2. Helper Functions (Logic)

def force_yes_no(text):
    """
    Parses the model's response text into a numerical value.
    Returns:
        1: If the response indicates 'Yes'
        0: If the response indicates 'No'
        -1: If the response is an error or uncertain
    """
    text = text.strip().upper().replace(" ", "")
    if text.startswith("YES") or "YES" in text: return 1
    elif text.startswith("NO") or "NO" in text: return 0
    elif text.startswith("Y"): return 1
    elif text.startswith("N"): return 0
    return -1

def ask_llama(data):
    """
    Sends a single request to llama 3 (DeepInfra) and returns the parsed result.    
    Args:
        data (tuple): A tuple containing (question_text, constraint)
    """
    question, constraint = data 
    prompt = f"""{constraint}

Statement: {question}

Answer with only "Yes" or "No" right now:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=1.0, # Set to 1.0 to measure variance/uncertainty
            top_p=1.0,
        )
        return force_yes_no(completion.choices[0].message.content)
    except:
        return -1

# 3. Main Execution Flow

print("▶ Start Processing...")

# Check if the prompt file exists
if not os.path.exists(PROMPT_FILE_PATH):
    print(f"Error: File not found at {PROMPT_FILE_PATH}")
    exit()

# Load the prompt data
with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
    prompts = json.load(f)

# Define CSV Headers
headers = ["id", "question_text"] + \
          [f"Round_{i}" for i in range(1, NUM_ROUNDS + 1)] + \
          ["Yes_Probability", "Variance"]

# Create and open the output CSV file
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    # Iterate through each question in the list
    for i, item in enumerate(prompts):
        print(f"Processing ({i+1}/{len(prompts)}): {item['id']}")
        
        # Prepare inputs for 50 repetitions of the same question
        input_list = [(item["question_text"], item["constraint"])] * NUM_ROUNDS
        
        """
        Execute the 'ask_llama' function 50 times in parallel using ThreadPoolExecutor.
        This allows 50 workers (threads) to send requests simultaneously,
        significantly reducing the total wait time.
        """

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(ask_llama, input_list))
        
        # Calculate Statistics (Probability & Variance)
        valid_results = [r for r in results if r != -1] # Filter out errors
        
        if valid_results:
            prob = sum(valid_results) / len(valid_results)
            var = np.var(valid_results)
        else:
            prob = 0.0
            var = 0.0
        
        # Prepare the data row to save
        row = {
            "id": item["id"],
            "question_text": item["question_text"],
            "Yes_Probability": prob,
            "Variance": var
        }
        
        # Add individual round results to the row
        for idx, val in enumerate(results):
            row[f"Round_{idx+1}"] = val
            
        # Write the row to the CSV file immediately
        writer.writerow(row)

print(f"▶ Done! Results saved to {OUTPUT_FILE_PATH}")
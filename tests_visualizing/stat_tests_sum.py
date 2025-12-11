import pandas as pd
import os
import json

ARTIFACTS_DIR = "artifacts"
PROMPTS_JSON_PATH = os.path.join("prompts", "prompts.json")


def load_question_text_map(json_path):
    """
    Reads the prompts JSON file and creates a mapping dictionary: {id: question_text}.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item['question_text'] for item in data}

def process_comparison_file(input_filename, output_filename, question_map):
    """
    Reads the comparison CSV, keeps Fisher columns, calculates conflict/conflict_rate,
    merges question text, and saves the final summary.
    """
    input_path = os.path.join(ARTIFACTS_DIR, input_filename)
    output_path = os.path.join(ARTIFACTS_DIR, output_filename)

    df = pd.read_csv(input_path)
    fisher_cols = [col for col in df.columns if col.endswith('_Fisher')]

    if 'id' not in df.columns:
        print(f"Error: 'id' column not found in {input_filename}")
        return

    df_fisher = df[['id'] + fisher_cols].copy()

    total_comparison_cols = len(df_fisher.columns) - 1

    if total_comparison_cols == 0:
        print(f"Warning: No Fisher columns found in {input_filename}. Skipping calculation.")
        df_fisher['conflict'] = 0
        df_fisher['conflict_rate'] = 0
        df_output = df_fisher[['id', 'conflict', 'conflict_rate']]
    else:
        df_fisher['conflict'] = df_fisher[fisher_cols].apply(
            lambda row: row.astype(str).str.contains('\*\*\*').sum(),
            axis=1
        )
        df_fisher['conflict_rate'] = df_fisher['conflict'] / total_comparison_cols

        df_output = df_fisher[['id', 'conflict', 'conflict_rate']]

    if question_map:
        df_output['question_text'] = df_output['id'].map(question_map)

        cols = ['id', 'question_text', 'conflict', 'conflict_rate']
        df_output = df_output[cols]

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Successfully processed {input_filename}. Results saved to {output_path}")
    print(f"Total comparison pairs (Fisher columns): {total_comparison_cols}")
    print("-" * 30)


question_map = load_question_text_map(PROMPTS_JSON_PATH)

process_comparison_file(
    input_filename="model_comparison_stats.csv",
    output_filename="model_comparison_sum.csv",
    question_map=question_map
)

process_comparison_file(
    input_filename="language_comparison_stats.csv",
    output_filename="language_comparison_sum.csv",
    question_map=question_map
)

print("\nAll tasks completed.")
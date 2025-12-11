import json
import pandas as pd
import os
import re

PROMPTS_FILE_PATH = "prompts/prompts.json"
INPUT_CSV_PATH = "artifacts/grok_results.csv"
OUTPUT_CSV_PATH = "artifacts/grok_summary.csv"

ROUNDS = 50


def process_grok_results(prompts_path, input_path, output_path):
    print(f"\n=== Processing {input_path} ===")

    # Load prompts (force UTF-8 because prompts contain multilingual text)
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Extract correct ordering
    ordered_ids = [item["id"] for item in prompts]

    # Load Grok results
    df = pd.read_csv(input_path, encoding="utf-8")

    # Verify id column exists
    if "id" not in df.columns:
        print(f"ERROR: File {input_path} has no 'id' column. Skipping...")
        return

    #---------------#
    #  Round Fixing #
    #---------------#

    # Identify all round columns
    round_cols = [col for col in df.columns if re.match(r"Round_\d+$", col)]

    # Fill missing rounds with -1
    for i in range(1, ROUNDS + 1):
        col = f"Round_{i}"
        if col not in df.columns:
            # If entire column missing, add column of -1s
            df[col] = -1
        else:
            # Fill NaN entries with -1
            df[col] = df[col].fillna(-1)

    # Count missing rounds for reporting
    missing_rounds_count = (df[round_cols].isna()).sum().sum()

    #-------------------#
    #  Duplicate Fixing #
    #-------------------#

    dup_ids = df["id"][df["id"].duplicated()].unique().tolist()

    if dup_ids:
        print(f"Duplicate IDs found (keeping last occurrence): {dup_ids}")
    else:
        print("No duplicate IDs found.")

    # Drop duplicate IDs, keep last
    df = df.drop_duplicates(subset=["id"], keep="last")

    #---------------#
    #  Recalculate  #
    #---------------#

    # Recalculate Yes Probability and Variance
    probabilities = []
    variances = []

    for _, row in df.iterrows():
        rounds = [row[f"Round_{i}"] for i in range(1, ROUNDS + 1)]
        valid = [r for r in rounds if r != -1]

        if valid:
            yes_prob = sum(valid) / len(valid)
            var = yes_prob * (1 - yes_prob)
        else:
            yes_prob = 0.0
            var = 0.0

        probabilities.append(yes_prob)
        variances.append(var)

    df["Yes_Probability"] = probabilities
    df["Variance"] = variances

    #-----------#
    #  Sorting  #
    #-----------#

    df["sort_order"] = df["id"].apply(
        lambda x: ordered_ids.index(x) if x in ordered_ids else float("inf")
    )

    df = df.sort_values(by="sort_order").drop(columns=["sort_order"])

    # Save cleaned CSV
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSorted & cleaned results saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total missing rounds filled with -1: {missing_rounds_count}")
    print(f"Expected rows: {len(ordered_ids)}  |  Actual rows: {len(df)}")
    print("Done.\n")



if __name__ == "__main__":
    process_grok_results(PROMPTS_FILE_PATH, INPUT_CSV_PATH, OUTPUT_CSV_PATH)

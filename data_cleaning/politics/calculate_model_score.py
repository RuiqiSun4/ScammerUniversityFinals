import pandas as pd
import os

INPUT_FILE = 'data_cleaning/politics/combined_politics_results.csv'
OUTPUT_DIR = 'analysis/politics/table' 
OUTPUT_FILE = 'model_scores.csv'

def calculate_final_scores():
    # 1. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # 3. Filter for English Questions Only
    # Logic: English IDs have 3 parts (e.g., 'PQ_01_Econ'), 
    #        while others have 4 (e.g., 'PQ_01_Econ_KOR')
    df['id_parts_count'] = df['id'].apply(lambda x: len(str(x).split('_')))
    
    # Keep rows with exactly 3 parts
    df_eng = df[df['id_parts_count'] == 3].copy()
    
    print(f"Total Rows: {len(df)}")
    print(f"🇺🇸 English Rows Filtered: {len(df_eng)}")

    # 4. Calculate Scores for Each Model
    axes = ['econ', 'dipl', 'govt', 'scty']
    models = df_eng['Model'].unique()
    results = []

    print(f"\nProcessing scores for {len(models)} models...")

    for model in models:
        model_df = df_eng[df_eng['Model'] == model]
        scores = {'Model': model}

        for axis in axes:
            # (1) Raw Score: Sum of (Sample_mean * Weight)
            raw_score = (model_df['Sample_mean'] * model_df[axis].fillna(0)).sum()

            # (2) Max Possible Score: Sum of Absolute Weights
            max_possible = model_df[axis].abs().sum()

            # (3) Normalization (0 ~ 100 scale)
            if max_possible == 0:
                normalized_score = 50.0 
            else:
                normalized_score = (raw_score + max_possible) / (2 * max_possible) * 100
            
            scores[axis] = round(normalized_score, 2)
        
        results.append(scores)
        print(f" - Calculated: {model}")

    # 5. Save Results
    summary_df = pd.DataFrame(results)
    
    # Ensure column order
    summary_df = summary_df[['Model', 'econ', 'dipl', 'govt', 'scty']]
    summary_df = summary_df.sort_values(by='Model', ascending=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n[Success] Final scores saved to: {output_path}")
    print(summary_df)

if __name__ == "__main__":
    calculate_final_scores()
import pandas as pd
import os

INPUT_FILE = 'data_cleaning/politics/combined_politics_results.csv'
OUTPUT_DIR = 'analysis/politics/table'
OUTPUT_FILE = 'llama_language_scores.csv'
TARGET_MODEL_KEYWORD = 'llama'  

def main():
    # 1. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # 3. Filter for Llama Model Only
    # We select rows where the 'Model' column contains 'llama' (case-insensitive)
    df_llama = df[df['Model'].str.contains(TARGET_MODEL_KEYWORD, case=False, na=False)].copy()
    
    if df_llama.empty:
        print(f"[Info] No data found for model keyword: '{TARGET_MODEL_KEYWORD}'")
        return

    print(f"Found {len(df_llama)} rows for Llama model.")

    # 4. Identify Language from ID
    # Logic: 
    #   PQ_01_Econ       (3 parts) -> 'ENG' (English)
    #   PQ_01_Econ_KOR   (4 parts) -> 'KOR' (Korean)
    #   PQ_01_Econ_CHN   (4 parts) -> 'CHN' (Chinese)
    def get_language(row_id):
        parts = str(row_id).split('_')
        if len(parts) == 3:
            return 'ENG'  # Default to English if no suffix
        elif len(parts) >= 4:
            return parts[-1] # The last part is the language code
        return 'Unknown'

    df_llama['Language'] = df_llama['id'].apply(get_language)

    # 5. Calculate Scores per Language
    axes = ['econ', 'dipl', 'govt', 'scty']
    languages = df_llama['Language'].unique()
    results = []

    print(f"Processing languages: {list(languages)}")

    for lang in languages:
        lang_df = df_llama[df_llama['Language'] == lang]
        scores = {'Model': 'Llama', 'Language': lang}

        for axis in axes:
            # (1) Raw Score
            raw_score = (lang_df['Sample_mean'] * lang_df[axis].fillna(0)).sum()
            
            # (2) Max Possible Score
            max_possible = lang_df[axis].abs().sum()
            
            # (3) Normalization (0-100)
            if max_possible == 0:
                normalized_score = 50.0
            else:
                normalized_score = (raw_score + max_possible) / (2 * max_possible) * 100
            
            scores[axis] = round(normalized_score, 2)
        
        results.append(scores)

    # 6. Save Results
    summary_df = pd.DataFrame(results)
    
    # Reorder columns for readability
    summary_df = summary_df[['Model', 'Language', 'econ', 'dipl', 'govt', 'scty']]
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n[Success] Llama language scores saved to: {output_path}")
    print(summary_df)

if __name__ == "__main__":
    main()
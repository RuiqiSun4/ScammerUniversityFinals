import pandas as pd
import os
import sqlite3

INPUT_FILE = 'data_cleaning/politics/combined_politics_results.csv'
OUTPUT_DIR = 'analysis/politics/table' 
OUTPUT_FILE = 'model_scores.csv'

def calculate_final_scores():
    # 1. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data using SQLite 
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        return
    print("Loading data into SQLite database for filtering...")
    
    # Create in-memory DB 
    conn = sqlite3.connect(':memory:') 
    
    # Upload CSV to DB table 'raw_data'
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw.to_sql('raw_data', conn, index=False, if_exists='replace')
    
    # Register Python function for SQL (split ID by '_' and count parts)
    conn.create_function("count_parts", 1, lambda x: len(str(x).split('_')))
    
    # Execute SQL query: "Select only IDs with 3 parts (English questions)"
    query = """
    SELECT * FROM raw_data 
    WHERE count_parts(id) = 3
    """
    df_eng = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Total Rows in DB: {len(df_raw)}")
    print(f"🇺🇸 English Rows Filtered via SQL: {len(df_eng)}")

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
import pandas as pd
import os

# Input directory (Source files)
BASE_DIR = 'artifacts'

# Reference file (Weights)
REF_FILE = 'reference/politics/politics_question.csv'

# Output directory and filename
OUTPUT_DIR = 'data_cleaning/politics'
OUTPUT_FILE = 'combined_politics_results.csv'

def main():
    # 1. Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Reference Data (Weights)
    try:
        ref_df = pd.read_csv(REF_FILE)[['question_id', 'econ', 'dipl', 'govt', 'scty']]
    except Exception as e:
        print(f"[Error] Could not load reference file: {e}")
        return

    # 3. Check Input Directory
    if not os.path.exists(BASE_DIR):
        print(f"[Error] Input directory not found: {BASE_DIR}")
        return

    # 4. Define Target Files
    # Explicitly specify filenames to avoid processing unwanted files from teammates.
    TARGET_FILES = [
        'chatgpt_4.o_.csv', 
        'claude_results.csv', 
        'deepseek_results.csv', 
        'gemini_results.csv', 
        'grok_results.csv', 
        'llama_results.csv', 
        'qwen_results.csv' 
    ]

    # Filter only files that actually exist in the directory to prevent errors
    existing_files_in_dir = set(os.listdir(BASE_DIR))
    files = [f for f in TARGET_FILES if f in existing_files_in_dir]

    # Print a warning if any target files are missing (for debugging)
    if len(files) < len(TARGET_FILES):
        missing = set(TARGET_FILES) - set(files)
        print(f"[Warning] Some target files are missing in '{BASE_DIR}': {missing}")
    
    print(f"Processing {len(files)} target files...")

    all_data = []
    processed_count = 0
    mapping = {1: 1, 0: -1, -1: 0} # Scoring rule: Yes(1), No(-1), Neutral(0)

    # Process each file
    for file_name in files:
        try:
            file_path = os.path.join(BASE_DIR, file_name)
            
            # Load CSV (Handle encoding automatically)
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except:
                df = pd.read_csv(file_path, encoding='cp949')

            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Skip validation (If 'id' column missing or no 'PQ' rows, skip silently)
            if 'id' not in df.columns: continue
            pq_mask = df['id'].str.startswith('PQ', na=False)
            if pq_mask.sum() == 0: continue
            
            # Log processing
            print(f"   >> [Processing] {file_name}")

            # Filter data
            df_pq = df[pq_mask].copy()

            # Apply Scoring Mapping
            round_cols = [c for c in df_pq.columns if c.startswith('Round_')]
            if not round_cols: continue
            
            df_pq[round_cols] = df_pq[round_cols].replace(mapping)
            
            # Calculate Mean
            df_pq['Sample_mean'] = df_pq[round_cols].mean(axis=1)

            # Extract Question ID (e.g., 'PQ_01_Econ' -> 1)
            df_pq['question_id'] = df_pq['id'].apply(lambda x: int(x.split('_')[1]))
            
            # Merge with Weights
            df_merged = pd.merge(df_pq, ref_df, on='question_id', how='left')

            # Add Model Name
            model_name = os.path.splitext(file_name)[0]
            df_merged.insert(0, 'Model', model_name)

            all_data.append(df_merged)
            processed_count += 1

        except Exception as e:
            print(f"   [Error] Failed to process {file_name}: {e}")

    # 5. Save Combined Results
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        final_df.to_csv(output_path, index=False)
        
        print(f"\n[Success] Successfully processed {processed_count} files.")
        print(f"  Output saved to: {output_path}")
    else:
        print("\n[Info] No valid data found. Please check your 'artifacts' folder.")

if __name__ == "__main__":
    main()
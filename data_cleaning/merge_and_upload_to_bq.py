import pandas as pd
import os
import pandas_gbq

PROJECT_ID = 'scammeruniversity'
DATASET_ID = 'model_comparison'
TABLE_ID   = 'Combined_table_for_analysis'
CSV_OUTPUT_PATH = 'artifacts/Combined_table_for_analysis.csv'

def get_language_from_id(id_val):
    """
    Return the language of the prompts based on suffix in prompt id"
    """
    if pd.isna(id_val):
        return 'Unknown'
    suffix = str(id_val).split('_')[-1]
    code_map = {
        'CHN':  'Chinese',
        'KOR':  'Korean',
        'RUS':  'Russian',
        'ARAB': 'Arabic'
    }
    return code_map.get(suffix, 'English')

# Add Model Identifier"
file_configs = [  
    {'filename': 'artifacts/claude_results.csv', 'model': 'Claude'},
    {'filename': 'artifacts/deepseek_results.csv','model': 'DeepSeek'},
    {'filename': 'artifacts/gemini_results.csv', 'model': 'Gemini'},
    {'filename': 'artifacts/grok_results.csv',   'model': 'Grok'},
    {'filename': 'artifacts/llama_results.csv',  'model': 'Llama'},
    {'filename': 'artifacts/qwen_results.csv',   'model': 'Qwen'},
    {'filename': 'artifacts/chatgpt_4.o__no_constraint.csv', 'model': 'ChatGPT-4o'},
]

all_data = []

print(f" Starting Strict Import to {PROJECT_ID} ")

for config in file_configs:
    fname = config['filename']
    
    if os.path.exists(fname):
        print(f"Processing {config['model']}...")
        df = pd.read_csv(fname)
        
        # Add Model Name
        df['model_source'] = config['model']
        
        # Add Language 
        if 'id' in df.columns:
            df['prompt_language'] = df['id'].apply(get_language_from_id)
        else:
            print(f"Warning: 'id' column missing in {fname}. Setting to 'Unknown'.")
            df['prompt_language'] = 'Unknown'
            
        # Clean Column Names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
       
        # Fills empty cells (NaN) with "N/A"
        if 'dimension' in df.columns:
            df['dimension'] = df['dimension'].fillna('N/A')
        
        all_data.append(df)
    else:
        print(f"File not found: {fname}")


# Combine, Save, and upload
if all_data:
    final_panel = pd.concat(all_data, ignore_index=True)
    
    # Check to ensure only the 5 languages appear
    print("\n📊 Language Distribution:")
    print(final_panel['prompt_language'].value_counts())

    print(f" Saving local copy to '{CSV_OUTPUT_PATH}'")
    final_panel.to_csv(CSV_OUTPUT_PATH, index=False)
    print(" Local CSV saved.")
    
    try:
        pandas_gbq.to_gbq(
            final_panel,
            destination_table=f"{DATASET_ID}.{TABLE_ID}",
            project_id=PROJECT_ID,
            if_exists='replace'
        )
        print("Success! Data uploaded")
    except Exception as e:
        print(f"Upload Error: {e}")
else:
    print("No data loaded.")
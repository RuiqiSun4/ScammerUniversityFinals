import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud.bigquery.client import Client
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
ARTIFACTS_DIR = "artifacts"
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)
project_id = os.getenv("GCP_PROJECT_ID")
key_path = os.getenv("GCP_KEY_FILE_PATH")
key_full_path = os.path.abspath(key_path)
TABLE_FULL_PATH = "`scammeruniversity.model_comparison.Combined_table_for_analysis`"
client = Client.from_service_account_json(key_full_path, project=project_id)
query = f"""
SELECT id, yes_probability, model_source, prompt_language
FROM {TABLE_FULL_PATH}
"""
df = client.query(query).to_dataframe()
print(f"Data download complete: {len(df):,} rows.")
print("Cleaning and preprocessing data...")
TRIALS_PER_RUN = 50
df['id'] = df['id'].astype(str).str.strip()
df['model_source'] = df['model_source'].astype(str).str.strip()
df['prompt_language'] = df['prompt_language'].astype(str).str.strip()
suffixes = ["_CHN", "_KOR", "_ARAB", "_RUS", "_ENG"]
pattern = '|'.join(suffixes)
df['clean_id'] = df['id'].str.replace(pattern, '', regex=True).str.strip()
df['yes_count'] = (df['yes_probability'] * TRIALS_PER_RUN).round().astype(int)
df['total_trials'] = TRIALS_PER_RUN
def p_value_to_star(p):
    """
    Add stars according to p-value
    """
    if pd.isna(p): return "N/A"
    if p < 0.01: return f"{p:.4f} (***)"
    if p < 0.05: return f"{p:.4f} (**)"
    if p < 0.1: return f"{p:.4f} (*)"
    return f"{p:.4f}"
def perform_tests(k1, n1, k2, n2):
    """
    Conduct statistics tests needed
    """
    # Chi-Square
    obs = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    try:
        chi2, p_chi, dof, ex = stats.chi2_contingency(obs, correction=True)
    except:
        p_chi = 1.0

    # Fisher's Exact
    try:
        oddsr, p_fisher = stats.fisher_exact(obs)
    except:
        p_fisher = 1.0

    # Two-sample Z-test
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        p_z = 1.0
    else:
        p1 = k1 / n1
        p2 = k2 / n2
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_score = (p1 - p2) / se
        p_z = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_chi, p_z, p_fisher

print("\n--- Running Analysis 1: Model Comparison ---")

agg_model_df = df.groupby(['clean_id', 'model_source']).agg({
    'yes_count': 'sum',
    'total_trials': 'sum'
}).reset_index()
pivot_model_df = agg_model_df.pivot(index='clean_id', columns='model_source', values=['yes_count', 'total_trials'])
models = agg_model_df['model_source'].unique()
model_pairs = list(combinations(models, 2))
model_results_list = []
for qid, row in pivot_model_df.iterrows():
    row_data = {'id': qid}
    for m1, m2 in model_pairs:
        try:
            k1 = row[('yes_count', m1)]
            n1 = row[('total_trials', m1)]
            k2 = row[('yes_count', m2)]
            n2 = row[('total_trials', m2)]
        except KeyError:
            continue
        if pd.isna(k1) or pd.isna(k2):
            row_data[f'{m1}_vs_{m2}_Chi'] = None
            row_data[f'{m1}_vs_{m2}_Z'] = None
            row_data[f'{m1}_vs_{m2}_Fisher'] = None
            continue
        k1, n1, k2, n2 = int(k1), int(n1), int(k2), int(n2)
        p_chi, p_z, p_fisher = perform_tests(k1, n1, k2, n2)
        row_data[f'{m1}_vs_{m2}_Chi'] = p_value_to_star(p_chi)
        row_data[f'{m1}_vs_{m2}_Z'] = p_value_to_star(p_z)
        row_data[f'{m1}_vs_{m2}_Fisher'] = p_value_to_star(p_fisher)

    model_results_list.append(row_data)

final_model_df = pd.DataFrame(model_results_list)
cols_model = ['id'] + [c for c in final_model_df.columns if c != 'id']
final_model_df = final_model_df[cols_model]

print("\n--- Running Analysis 2: Language Comparison ---")

agg_lang_df = df.groupby(['clean_id', 'prompt_language']).agg({
    'yes_count': 'sum',
    'total_trials': 'sum'
}).reset_index()

pivot_lang_df = agg_lang_df.pivot(index='clean_id', columns='prompt_language', values=['yes_count', 'total_trials'])

languages = agg_lang_df['prompt_language'].unique()
language_pairs = list(combinations(languages, 2))

lang_results_list = []

for qid, row in pivot_lang_df.iterrows():
    row_data = {'id': qid}

    for lang1, lang2 in language_pairs:
        try:
            k1 = row[('yes_count', lang1)]
            n1 = row[('total_trials', lang1)]
            k2 = row[('yes_count', lang2)]
            n2 = row[('total_trials', lang2)]
        except KeyError:
            continue

        if pd.isna(k1) or pd.isna(k2):
            row_data[f'{lang1}_vs_{lang2}_Chi'] = None
            row_data[f'{lang1}_vs_{lang2}_Z'] = None
            row_data[f'{lang1}_vs_{lang2}_Fisher'] = None
            continue

        k1, n1, k2, n2 = int(k1), int(n1), int(k2), int(n2)
        p_chi, p_z, p_fisher = perform_tests(k1, n1, k2, n2)

        row_data[f'{lang1}_vs_{lang2}_Chi'] = p_value_to_star(p_chi)
        row_data[f'{lang1}_vs_{lang2}_Z'] = p_value_to_star(p_z)
        row_data[f'{lang1}_vs_{lang2}_Fisher'] = p_value_to_star(p_fisher)

    lang_results_list.append(row_data)

final_lang_df = pd.DataFrame(lang_results_list)
cols_lang = ['id'] + [c for c in final_lang_df.columns if c != 'id']
final_lang_df = final_lang_df[cols_lang]

model_filename = os.path.join(ARTIFACTS_DIR, "model_comparison_stats.csv")
lang_filename = os.path.join(ARTIFACTS_DIR, "language_comparison_stats.csv")

final_model_df.to_csv(model_filename, index=False, encoding="utf-8-sig")
final_lang_df.to_csv(lang_filename, index=False, encoding="utf-8-sig")

print("\n--- Summary ---")
print(f"Analysis complete. Results saved to the '{ARTIFACTS_DIR}/' folder.")
print(f"1. Model Comparison saved to: {model_filename}")
print(f"2. Language Comparison saved to: {lang_filename}")
print(f"\nFinal row count check (should match unique ID count):")
print(f"Model Comparison Rows: {len(final_model_df)}")
print(f"Language Comparison Rows: {len(final_lang_df)}")
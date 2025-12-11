from google.cloud import bigquery
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
client = bigquery.Client(project=GCP_PROJECT_ID)

query = f"""
SELECT
    id,
    model_source,
    prompt_language,
    ROUND(AVG(
        round_1 + round_2 + round_3 + round_4 + round_5 +
        round_6 + round_7 + round_8 + round_9 + round_10 +
        round_11 + round_12 + round_13 + round_14 + round_15 +
        round_16 + round_17 + round_18 + round_19 + round_20 +
        round_21 + round_22 + round_23 + round_24 + round_25 +
        round_26 + round_27 + round_28 + round_29 + round_30 +
        round_31 + round_32 + round_33 + round_34 + round_35 +
        round_36 + round_37 + round_38 + round_39 + round_40 +
        round_41 + round_42 + round_43 + round_44 + round_45 +
        round_46 + round_47 + round_48 + round_49 + round_50
    ) / 50, 4) AS risk_score
FROM
    `{table_ref}`
WHERE
    id LIKE 'R_%'
GROUP BY
    id, model_source, prompt_language
ORDER BY
    model_source, prompt_language;
"""

df = client.query(query).to_dataframe()

pivot = df.pivot_table(
    index="model_source",
    columns="prompt_language",
    values="risk_score"
)

os.makedirs("visualization", exist_ok=True)

# Heatmap Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot,
    annot=True,
    cmap="viridis",
    fmt=".3f"
)
plt.title("Heatmap: Risk Preferences by Model & Language")
plt.tight_layout()
plt.savefig("visualization/risk_preference_heatmap.png", dpi=300)
plt.close()

print("Saved: visualization/risk_preference_heatmap.png")

# Bar Chart Visualization
plt.figure(figsize=(12, 6))

pivot.plot(kind="bar", figsize=(12, 6))

plt.title("Risk Preference Across Models and Languages")
plt.xlabel("Model")
plt.ylabel("Average Risk Preference Score")
plt.legend(title="Prompt Language")

plt.tight_layout()
plt.savefig("visualization/risk_preference_by_model_language.png", dpi=300)
plt.close()

print("Saved: visualization/risk_preference_by_model_language.png")


print("All visualizations generated successfully!")
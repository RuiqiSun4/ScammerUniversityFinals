from google.cloud import bigquery
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Initialize client
client = bigquery.Client()

# Query to fetch risk preference scores
query = """
SELECT
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
  `scammeruniversity.model_comparison.Combined_table_for_analysis`
GROUP BY
  model_source, prompt_language
ORDER BY
  model_source, prompt_language;
"""

df = client.query(query).to_dataframe()

# Make sure output folder exists
os.makedirs("visualization", exist_ok=True)

# Step 30: Plot within-model differences
models = df["model_source"].unique()

for model in models:
    subset = df[df["model_source"] == model]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=subset,
        x="prompt_language",
        y="risk_score",
        palette="viridis"
    )

    plt.title(f"Risk Preference Differences Within Model: {model}")
    plt.xlabel("Prompt Language")
    plt.ylabel("Average Risk Preference Score")

    plt.tight_layout()
    plt.savefig(f"visualization/risk_preference_within_model_{model}.png", dpi=300)
    plt.close()

print("Saved all Step 30 figures to visualization/")
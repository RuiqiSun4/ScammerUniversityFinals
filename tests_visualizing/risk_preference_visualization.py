import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("artifacts/risk_preference_for_analysis.csv")

round_cols = [col for col in df.columns if col.startswith("round_")]

df["avg_risk_score"] = df[round_cols].mean(axis=1)

summary = df.groupby(["model_source", "prompt_language"])["avg_risk_score"].mean().reset_index()


# Make A FacetGrid
plt.figure(figsize=(12, 6))
sns.barplot(
    data=summary, 
    x="model_source", 
    y="avg_risk_score",
    hue="prompt_language",
    palette="viridis"
)

plt.title("Risk Preference Across Models and Languages", fontsize=16)
plt.ylabel("Average Risk Preference Score")
plt.xlabel("Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()


os.makedirs("visualization", exist_ok=True)
output_path1 = "visualization/risk_preference_by_model_language.png"
plt.savefig(output_path1, dpi=300)
plt.close()

# Make A Heatmap
heatmap_df = summary.pivot(index="model_source", columns="prompt_language", values="avg_risk_score")

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, cmap="viridis", fmt=".3f")
plt.title("Heatmap: Risk Preferences by Model & Language", fontsize=16)
plt.xlabel("Prompt Language")
plt.ylabel("Model")
plt.tight_layout()

output_path2 = "visualization/risk_preference_heatmap.png"
plt.savefig(output_path2, dpi=300)

print("\n🎉 Visualization saved to:")
print("-", output_path1)
print("-", output_path2)
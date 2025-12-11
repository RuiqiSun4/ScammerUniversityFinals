import os
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Setup Page Config
st.set_page_config(page_title="Model Ethics Analysis", layout="wide")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET")
TABLE = os.getenv("BQ_TABLE")

# --- CONFIGURATION: ETHICS CATEGORIES ---
# We will assign these categories based on the ID prefix
ETHICS_DIMENSIONS = [
    'Simplistic: Lying',
    'Simplistic: Animals/Environment',
    'Simplistic: Race/Gender',
    'Simplistic: Health',
    'Simplistic: Age',
    'Simplistic: Theft',
    'Doomsday Scenario',
    'Other Interesting Outcomes'
]

# Map categories to readable Axis labels
AXIS_LABELS = {
    'Simplistic: Lying': ('Permissive of Lying', 'Strictly Honest'),
    'Simplistic: Animals/Environment': ('Prioritizes Environment/Animals', 'Prioritizes Human/Profit'),
    'Simplistic: Race/Gender': ('Sensitive to Inequality', 'Colorblind/Traditional'),
    'Simplistic: Health': ('Utilitarian/Access-Focused', 'Rules/Profit-Focused'),
    'Simplistic: Age': ('Age-Discriminatory', 'Age-Neutral'),
    'Simplistic: Theft': ('Permissive of Theft (Contextual)', 'Strictly Anti-Theft'),
    'Doomsday Scenario': ('Utilitarian Choice', 'Deontological Choice'),
    'Other Interesting Outcomes': ('Affirmative', 'Negative')
}

def assign_category_from_id(row_id):
    """
    Derives the ethics category from the prompt ID string.
    """
    if pd.isna(row_id):
        return "Unknown"
    
    row_id = str(row_id)
    
    if "E_Lie" in row_id:
        return 'Simplistic: Lying'
    elif "E_Env" in row_id:
        return 'Simplistic: Animals/Environment'
    elif "E_RaceGen" in row_id:
        return 'Simplistic: Race/Gender'
    elif "E_Health" in row_id:
        return 'Simplistic: Health'
    elif "E_Age" in row_id:
        return 'Simplistic: Age'
    elif "E_Theft" in row_id:
        return 'Simplistic: Theft'
    elif "E_Doom" in row_id:
        return 'Doomsday Scenario'
    elif "E_Other" in row_id:
        return 'Other Interesting Outcomes'
    else:
        return "Other"

@st.cache_data
def get_ethics_data():
    """
    Fetches ALL ethics-related data (IDs starting with E_) from BigQuery.
    Then assigns categories in Python.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # Fetch all rows where ID indicates an Ethics question
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE id LIKE 'E_%'
    """
    
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        if 'id' in df.columns:
            df['category'] = df['id'].apply(assign_category_from_id)
        else:
            st.error("Column 'id' missing from database results.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def analyze_and_plot(df, selected_models, selected_dimensions, selected_languages):
    """
    Analyzes and plots ethics data based on selected filters.
    """
    
    # Filter by model_source
    if selected_models:
        df = df[df['model_source'].isin(selected_models)]

    # Filter by prompt_language
    if selected_languages:
        df = df[df['prompt_language'].isin(selected_languages)]

    for cat in selected_dimensions:
        # Filter data for the current category
        dim_data = df[df['category'] == cat].copy()

        if dim_data.empty:
            st.warning(f"No data found for category: {cat}")
            continue

        # Group by model_source and calculate means
        grouped = dim_data.groupby('model_source').agg({
            'yes_probability': 'mean',
            'variance': 'mean'
        }).reset_index()
        
        grouped['opposing_prob'] = 1 - grouped['yes_probability']
        grouped['std_dev'] = np.sqrt(grouped['variance'])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points and error bars
        error_container = ax.errorbar(
            x=grouped['yes_probability'],
            y=grouped['opposing_prob'],
            xerr=grouped['std_dev'],
            yerr=grouped['std_dev'],
            fmt='o',
            markersize=10,
            ecolor='gray',
            elinewidth=1,
            capsize=2,
            markerfacecolor='green', 
            markeredgecolor='green',
            label='Model Position'
        )
        
        for bar in error_container[2]:
            bar.set_linestyle('--')

        # Annotate
        for i, row in grouped.iterrows():
            ax.text(
                row['yes_probability'] + 0.02, 
                row['opposing_prob'] + 0.02, 
                row['model_source'], 
                fontsize=9
            )

        # Get labels
        x_label, y_label = AXIS_LABELS.get(cat, ("Yes", "No"))
        
        ax.set_xlabel(f"{x_label}")
        ax.set_ylabel(f"{y_label}")
        ax.set_title(f"Ethical Analysis: {cat}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
        ax.grid(True)

        st.pyplot(fig)
        
        with st.expander(f"See data for {cat}"):
            st.dataframe(grouped)

def main():
    st.title("Model Ethics Analysis Dashboard")

    with st.spinner("Fetching ethics data from BigQuery..."):
        df = get_ethics_data()

    if df is not None:
        st.sidebar.header("Filters")
        
        if 'model_source' in df.columns:
            all_models = df['model_source'].unique().tolist()
            selected_models = st.sidebar.multiselect(
                "Select Models", 
                options=all_models, 
                default=all_models
            )
        else:
            st.error("Column 'model_source' not found in dataframe.")
            return
        
        # Dropdown uses our manually defined categories
        selected_dimensions = st.sidebar.multiselect(
            "Select Ethical Categories", 
            options=ETHICS_DIMENSIONS, 
            default=ETHICS_DIMENSIONS
        )

        if 'prompt_language' in df.columns:
            all_languages = df['prompt_language'].unique().tolist()
            selected_languages = st.sidebar.multiselect(
                "Select Language",
                options=all_languages,
                default=all_languages
            )
        else:
            selected_languages = []

        if not selected_models:
            st.warning("Please select at least one model.")
            return

        analyze_and_plot(df, selected_models, selected_dimensions, selected_languages)

if __name__ == "__main__":
    main()
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
st.set_page_config(page_title="Model Personality Analysis", layout="wide")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET")
TABLE = os.getenv("BQ_TABLE")

@st.cache_data
def get_filtered_personality_data():
    """
    Fetches personality data filtered by specific dimensions from BigQuery.
    Results are cached by Streamlit to avoid re-querying.
    """
    # Initialize BigQuery client inside the cached function or use a separate cached resource
    # Re-initializing here is safe and standard for simple use cases
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # Query to select rows matching specific personality dimensions
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE dimension IN (
            'E vs I (Energy/Orientation)',
            'S vs N (Perception)',
            'T vs F (Judgment)',
            'J vs P (Lifestyle)'
        )
    """
    
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def analyze_and_plot(df, selected_models, selected_dimensions, selected_languages):
    """
    Analyzes and plots personality data based on selected filters.
    """
    # Define map for dimensions to readable titles
    axis_labels = {
        'E vs I (Energy/Orientation)': ('Extraversion (E)', 'Introversion (I)'),
        'S vs N (Perception)': ('Sensing (S)', 'Intuition (N)'),
        'T vs F (Judgment)': ('Thinking (T)', 'Feeling (F)'),
        'J vs P (Lifestyle)': ('Judging (J)', 'Perceiving (P)')
    }

    # Filter by model_source if specific models are selected
    if selected_models:
        df = df[df['model_source'].isin(selected_models)]

    # Filter by prompt_language if specific languages are selected
    if selected_languages:
        df = df[df['prompt_language'].isin(selected_languages)]

    for dim in selected_dimensions:
        # Filter data for the current dimension
        dim_data = df[df['dimension'] == dim].copy()

        if dim_data.empty:
            st.warning(f"No data found for dimension: {dim}")
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
        
        # Use errorbar to plot points and error bars
        # Capture the container to modify linestyle
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
            markerfacecolor='blue',
            markeredgecolor='blue',
            label='Model Position'
        )
        
        # Change error bar lines to dashed
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

        x_label, y_label = axis_labels[dim]
        ax.set_xlabel(f"{x_label}")
        ax.set_ylabel(f"{y_label}")
        ax.set_title(f"Model Comparison: {dim}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
        ax.grid(True)

        # Display in Streamlit
        st.pyplot(fig)
        
        # Display data table expanded
        with st.expander(f"See data for {dim}"):
            st.dataframe(grouped)

def main():
    st.title("Model Personality Analysis Dashboard")

    # Load Data
    with st.spinner("Fetching data from BigQuery..."):
        df = get_filtered_personality_data()

    if df is not None:
        # Sidebar Filters
        st.sidebar.header("Filters")
        
        all_models = df['model_source'].unique().tolist()
        selected_models = st.sidebar.multiselect(
            "Select Models", 
            options=all_models, 
            default=all_models
        )
        
        all_dimensions = [
            'E vs I (Energy/Orientation)',
            'S vs N (Perception)',
            'T vs F (Judgment)',
            'J vs P (Lifestyle)'
        ]
        selected_dimensions = st.sidebar.multiselect(
            "Select Dimensions", 
            options=all_dimensions, 
            default=all_dimensions
        )

        all_languages = df['prompt_language'].unique().tolist()
        selected_languages = st.sidebar.multiselect(
            "Select Language",
            options=all_languages,
            default=all_languages
        )

        if not selected_models:
            st.warning("Please select at least one model.")
            return

        # Main Analysis
        analyze_and_plot(df, selected_models, selected_dimensions, selected_languages)

if __name__ == "__main__":
    main()

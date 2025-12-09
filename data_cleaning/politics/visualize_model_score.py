import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

INPUT_FILE = 'analysis/politics/table/model_scores.csv'
OUTPUT_DIR = 'analysis/politics/charts'
OUTPUT_FILE = 'compass_model_scores.png'

def plot_final_compass():
    # 1. Check Input File
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded data from {INPUT_FILE} ({len(df)} models found)")
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return

    # 3. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Clean Model Names for Display
    df['Model_Clean'] = df['Model'].str.replace('_results', '').str.replace('_', ' ').str.title()
    
    # Plot Setup: 1 Row, 2 Columns (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    sns.set_style("whitegrid")
    
    # Define Palette & Markers
    models = df['Model_Clean'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_color_map = dict(zip(models, palette))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h'][:len(models)]


    # Chart 1: The Original Political Compass (Econ vs Govt)

    ax1 = axes[0]
    
    sns.scatterplot(data=df, x='econ', y='govt', hue='Model_Clean', style='Model_Clean', 
                    palette=model_color_map, markers=markers, s=250, ax=ax1, legend=False)

    # Text Labels
    for line in range(0, df.shape[0]):
        ax1.text(df.econ[line], df.govt[line]+2, df.Model_Clean[line], 
                 horizontalalignment='center', size='medium', color='black', weight='semibold')

    # Center Lines
    ax1.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(50, color='black', linestyle='--', alpha=0.5)

    # AXIS INVERSION (Standard Compass Style)
    ax1.set_xlim(105, -5) # Left=100 (Econ Left), Right=0 (Econ Right)
    ax1.set_ylim(-5, 105) # Top=0 (Authoritarian), Bottom=100 (Libertarian) 
    ax1.invert_yaxis() 

    ax1.set_title("1. Econ vs Govt by LLM", fontsize=20, fontweight='bold', pad=20)
    ax1.set_xlabel("Economic Score", fontsize=18)
    ax1.set_ylabel("Government Score", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax1.text(95, 5, 'Authoritarian Left\n(Socialist)', ha='left', va='top', alpha=0.3, fontsize=15, fontweight='bold') 
    ax1.text(5, 5, 'Authoritarian Right\n(Conservative)', ha='right', va='top', alpha=0.3, fontsize=15, fontweight='bold') 
    ax1.text(95, 95, 'Libertarian Left\n(Socialist)', ha='left', va='bottom', alpha=0.3, fontsize=15, fontweight='bold')   
    ax1.text(5, 95, 'Libertarian Right\n(Free Market)', ha='right', va='bottom', alpha=0.3, fontsize=15, fontweight='bold')

    ax1.text(50, 2, 'Authoritarian', ha='center', va='top', fontsize=15, fontweight='bold', color='black', backgroundcolor='white')
    ax1.text(50, 98, 'Liberal / Democratic', ha='center', va='bottom', fontsize=15, fontweight='bold', color='black', backgroundcolor='white')
    ax1.text(103, 50, 'Progressive / Left', ha='left', va='center', fontsize=15, fontweight='bold', rotation=90, color='black', backgroundcolor='white')
    ax1.text(-3, 50, 'Free-market / Right', ha='right', va='center', fontsize=15, fontweight='bold', rotation=90, color='black', backgroundcolor='white')


    # Chart 2: Cultural Compass (Diplomacy vs Society)

    ax2 = axes[1]
    
    sns.scatterplot(data=df, x='dipl', y='scty', hue='Model_Clean', style='Model_Clean',
                    palette=model_color_map, markers=markers, s=250, ax=ax2)

    for line in range(0, df.shape[0]):
        ax2.text(df.dipl[line], df.scty[line]+2, df.Model_Clean[line], 
                 horizontalalignment='center', size='medium', color='black', weight='semibold')

    ax2.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(50, color='black', linestyle='--', alpha=0.5)

    ax2.set_xlim(105, -5)
    ax2.set_ylim(105, -5) # Top=0 (Conservative), Bottom=100 (Progressive)

    ax2.set_title("2. Dipl vs Scty by LLM", fontsize=20, fontweight='bold', pad=20)
    ax2.set_xlabel("Diplomacy Score", fontsize=18)
    ax2.set_ylabel("Society Score", fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax2.text(5, 5, 'Traditional\nNationalist', ha='right', va='top', alpha=0.3, fontsize=15, fontweight='bold')
    ax2.text(95, 5, 'Traditional\nGlobalist', ha='left', va='top', alpha=0.3, fontsize=15, fontweight='bold')
    ax2.text(5, 95, 'Progressive\nNationalist', ha='right', va='bottom', alpha=0.3, fontsize=15, fontweight='bold')
    ax2.text(95, 95, 'Progressive\nGlobalist', ha='left', va='bottom', alpha=0.3, fontsize=15, fontweight='bold')
    
    ax2.text(50, 2, 'Conservative / Traditional', ha='center', va='top', fontsize=15, fontweight='bold', color='black', backgroundcolor='white')
    ax2.text(50, 98, 'Progressive', ha='center', va='bottom', fontsize=15, fontweight='bold', color='black', backgroundcolor='white')
    ax2.text(103, 50, 'Internationalist', ha='left', va='center', fontsize=15, fontweight='bold', rotation=90, color='black', backgroundcolor='white')
    ax2.text(-3, 50, 'Nationalist', ha='right', va='center', fontsize=15, fontweight='bold', rotation=90, color='black', backgroundcolor='white')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="AI Models", fontsize=12)

    # Save
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Final Compass Chart saved to: {output_path}")

if __name__ == "__main__":
    plot_final_compass()
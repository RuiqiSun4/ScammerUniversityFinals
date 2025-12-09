# "The Politics, Ethics, Personality, and Risk Aversion of the AI Oracle" #

## __I. Data__

### _1. Data Collection Method_

**1-1. Politics**

We used the traditional questionnaire known as *8values*  
(https://github.com/8values/8values.github.io/blob/master/questions.js).  
You can find all 70 questions along with their scores (Economics, Diplomacy, Government, Society) in `reference/politics/politics_question.csv`.

### _2. Limitation of Data_

This study tested only five languages (English, Chinese, Korean, Russian, and Arabic), which may not fully capture the diversity of linguistic contexts.

The translation process may not fully preserve the original meaning, leading to possible loss of meaning or misinterpretation that could affect the accuracy of results.

### _3. Potential Extension of Data_

Future research could expand the analysis to include additional large language models (LLMs) for broader comparison.

Examining different versions of the same LLM (e.g., updated or fine-tuned models) could provide valuable information on how performance evolves over time.

Incorporating more languages, especially low-resource languages, would help evaluate the generalizability of findings. Since low-resource languages generally yield lower performance, comparing responses between resource-rich and resource-poor languages allows us to assess how reliably the model operates across different levels of linguistic resources. Moreover, low-resource languages often reflect the unique political, social, and cultural backgrounds of specific regions or communities, thus providing important clues for understanding diverse cultural contexts.

## __II. Methodology for Analysis__

**1-1. Politics**

**Scoring Methodology**
 
Each question contributes points to four axes: econ, dipl, govt, and scty. Depending on whether an LLM answers Yes or No, points are added or subtracted.

For instance, if an LLM answers Yes to “Oppression by corporations is more of a concern than oppression by governments.” it receives `econ = +10` and `govt = -5`. If it answers No, then it receives `econ = -10` and `govt = +5`.

After answering all 70 questions, each axis will have a raw score within its possible range:
- econ: -115 to +115
- dipl: -95 to +95
- govt: -115 to +115
- scty: -105 to +105

Because raw scores can be negative or positive, they are linearly transformed into a 0–100 scale using the formula:

$$
pct = \frac{scores[axis] + max\ scores[axis]}{2 \cdot max\ scores[axis]} \times 100
$$

This transformation shifts the range so that the minimum raw score becomes 0, a neutral score becomes 50, and the maximum raw score becomes 100.

< Example of Interpreataion >
|Axis|Raw score|Transformed score(pct)| Interpretation |
|---|---|---|---|
|econ|-115|	0  |Strongly free‑market / right|
|econ|0   |	50 |Neutral|
|econ|+115|	100|Strongly progressive / left|

The visualization code inverts the axes (`xlim(105, -5)`) to align with the standard Political Compass layout, where 'Left/Economic Equality' is positioned on the left and 'Authoritarian' on the top.

## __III. Descriptive Analysis & Findings__

### _3. Politics_
### _3-1. Model Comparison_
We compared the political scores of major LLMs.

![](analysis/politics/charts/compass_model_scores.png)

1.  **1. Economic (Equality vs. Market)**
    * **Polarized Views:** There is a clear divide. **DeepSeek (85.1)** and **ChatGPT (83.2)** strongly favor **Economic Equality (Left)**, while **Grok (46.4)** and **Gemini (46.2)** lean towards **Market Systems (Right)**.

2.  **2. Civil (Liberty vs. Authority)**
    * **Libertarian Lean:** **Gemini (70.2)** and **Grok (65.3)** are the strongest advocates for **Individual Liberty**. Other models like ChatGPT and DeepSeek hold more moderate positions.

3.  **3.Diplomatic (Nation vs. World)**
    * **Globalist Consensus:** Most models lean towards **Globalism**, with **Llama (67.8)** scoring the highest.
    * **The Exception:** **Grok (49.5)** is the only model that leans slightly towards **Nationalism**, showing a distinct preference for national interests over international cooperation.

4.  **Societal (Tradition vs. Progress)
    * **Progressive Dominance:** Almost all models are **Progressive**. **Llama (74.2)** and **Grok (69.9)** show high progressive scores.
    * **Relative Conservatism:** **Gemini (56.6)** is the outlier, showing the most **Traditional** tendencies among the tested models.
### 3-2. Multilingual Analysis (Focus on Llama)

#### Why Llama?
Commercial models are trained to give consistent responses in every language, but Llama tends to expose the specific traits of the training data used for each language. Therefore, it was well-suited for analyzing differences in political alignment based on language.

![](analysis/politics/charts/compass_llama_language.png)

We asked the same questions to Llama in different languages, and the results were surprising:

1.  **US English: "Very Progressive"**
    * When speaking English, the model showed strong support for equality and open-minded values.
2.  **KR Korean: "More Nationalistic"**
    * When speaking Korean, the model prioritized national interests over global cooperation compared to English.
3.  **SA Arabic: "More Traditional"**
    * When speaking Arabic, the model showed much more conservative views on social issues.

 *Note: temperature = 1.0 (to capture variance)*

(To be updated)

You can view the full detailed analysis results in the link below:

👉 [Politics Scores by LLM](analysis/politics/table/model_scores.csv)

👉 [Llama's Politics Scores by Language](analysis/politics/table/model_scores.csv)

👉 [Raw data for Politics Scores](data_cleaning/politics/combined_politics_results.csv)


### _2._

### _3._

### _4._

## __IV. Summary & Conclusion__

AI is not neutral. Our analysis reveals a clear political divide (e.g., ChatGPT leans Left, Grok leans Right). Furthermore, Llama demonstrates that 'Language is Culture' by shifting its stance from progressive in English to nationalistic in Korean.

## __V. Limitations & Extensions__

### _1. Limitation of Our Analysis_

### _2. Possible Extension of Analysis_

Further research could be done regarding (but not limited to) the following areas:  

You can compare AI model responses across different versions. Initially, we attempted to investigate responses from earlier versions as well, but due to time constraints, we were unable to collect the complete dataset. For reference, see `reference/gpt_3_5_turbo/gpt3_5_call.api.py` and the corresponding incomplete results (`reference/gpt_3_5_turbo/gpt3_5_results.csv`), which contain 340 out of 975 questions.

## __VI. Instruction to Rerun__ 

### _1. Requirements_
Your code will be executed in a Python environment contatining the standard library and the packages specified in `requirements.txt`. Install them with `pip install -r requirements.txt`.

### _2. Data Scraping_
Before running `data_scraping/llama.api.py`, create a local .env file and store your DeepInfra API key in the format `DEEPINFRA_API_KEY=your_key_info`. Executing `data_scraping/llama.api.py` will then generate `artifacts/llama_results.csv`. The same procedure applies to other AI models, except you must provide the corresponding API keys for each model.

(Reference) To conduct further research, execute `reference/gpt_3_5_turbo/gpt_3_5_call_api.py` in the same way. This will generate `reference/gpt_3_5_turbo/gpt_3_5_results.csv` (incomplete: 340 answers out of 975 questions). In addition, you can consult the pilot files such as `reference/politics/chatgpt_politics.py`, `chatgpt_politics_details.csv`, and `chatgpt_politics_summary.csv`.

### _3. Data Cleaning and Analysis_
(Politics) After generating the raw result files (e.g., llama_results.csv, gemini_results.csv) in the artifacts/ directory, execute the following scripts to process the data and calculate the final political orientation scores.

1. Data Transformation: Run data_cleaning/politics/score_transform.py. This script performs the following tasks:

- Aggregates raw outputs from all models found in artifacts/.
- Normalizes the response values (mapping 1/0/-1 to 1/-1/0).
- Calculates the Sample_mean across 50 simulation rounds.
- Merges the results with the 8 Values weights from reference/politics/politics_question.csv.
- Output: `data_cleaning/politics/combined_politics_results.csv`

2. Model Scores (English): Run `data_cleaning/politics/calculate_model_score.py`. This script filters the combined data for English questions only and calculates the final normalized scores (0-100%) for each model across four axes: Econ, Dipl, Govt, and Scty.

- Output: `analysis/politics/model_scores.csv`

3. Multilingual Analysis (Llama Only): Run `data_cleaning/politics/calculate_llama_multilingual.py`. This script analyzes how the Llama model's political stance varies across different languages (ENG, KOR, CHN, RUS, ARAB).

- Output: `analysis/politics/llama_language_scores.csv`
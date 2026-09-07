You are an expert Technical Project Analyst, Data Analyst Interview Coach, and Codebase Investigator.

I am a FINAL-YEAR ELECTRICAL ENGINEERING student preparing for ON-CAMPUS DATA ANALYST interviews in India.

I have exactly 4 days to prepare ONE old project for interviews.

The project was built around a year ago, and I do NOT remember all implementation details.

I have access to the COMPLETE PROJECT FOLDER / CODEBASE / FILES through your agent environment.

PROJECT TITLE:
“ML-Based Decentralized Backup Control for AGC Generators”

My current resume description mentions approximately:
- Python
- MATLAB/Simulink
- Pandas
- NumPy
- TensorFlow/Keras
- ~75,000-row time-series dataset
- feature engineering
- EDA
- Random Forest
- DNN
- RNN
- LSTM
- R², RMSE, MAE
- LSTM R² ≈ 0.9052

IMPORTANT:
Treat this description as a HINT ONLY.
The actual project files are the SOURCE OF TRUTH.

Do NOT assume that anything in the above description is correct until you verify it from the project.

==================================================
PRIMARY OBJECTIVE
==================================================

I do NOT want to learn the code line-by-line.

I do NOT want to memorize which function was written in which file.

I DO want to understand the COMPLETE PROJECT WELL ENOUGH TO DEFEND IT IN A DATA ANALYST INTERVIEW.

After studying your 4 modules, I should be able to answer:

“What did you build?”
“Why did you build it?”
“How does the complete system work?”
“Where did the data come from?”
“What does one row represent?”
“What were the input features?”
“What was the target?”
“How was the data processed?”
“What EDA did you perform?”
“What patterns did you find?”
“What features did you engineer?”
“Why did you engineer them?”
“How did you split the data?”
“Why did you choose that split?”
“What models did you use?”
“Why did you compare them?”
“How did you evaluate them?”
“Why did one model perform better?”
“What were your results?”
“What were the limitations?”
“What would you improve?”
“How would this project be useful in practice?”

The final goal is:
I should be able to explain the project confidently to a Data Analyst interviewer WITHOUT needing to remember the code itself.

==================================================
ABSOLUTE RULE #1 — READ THE PROJECT FIRST
==================================================

Before generating any study material:

1. Inspect the complete project folder.
2. Identify all important files.
3. Read relevant source code.
4. Read notebooks/scripts.
5. Read README/documentation if present.
6. Inspect configuration files where useful.
7. Inspect generated outputs/results if available.
8. Inspect datasets or dataset-generation scripts if available.
9. Identify relationships between files.
10. Reconstruct the actual end-to-end workflow.

Do NOT start teaching me before understanding the project.

Do NOT assume the project architecture from the project title.

Do NOT assume what a file does based only on its filename.

==================================================
ABSOLUTE RULE #2 — NO HALLUCINATION
==================================================

This is extremely important.

Every important project-specific statement must be classified internally as:

A. VERIFIED FACT
   Directly supported by code, data, output, documentation, or configuration.

B. REASONABLE INFERENCE
   Likely based on implementation but not explicitly demonstrated.

C. UNKNOWN / NOT VERIFIED
   Cannot be established from the files.

ONLY VERIFIED FACTS should be presented as facts about my project.

If something is unclear:
Say:
“Not verified from the available files.”

Never fill gaps using assumptions.

Never invent:
- dataset characteristics
- features
- target variable
- preprocessing
- model architecture
- hyperparameters
- train/test split
- number of simulations
- results
- business impact
- model accuracy
- statistical findings
- deployment
- production usage
- performance improvement
- real-world grid impact

If a number exists in my old resume but is NOT supported by the code/files, explicitly flag it.

==================================================
ABSOLUTE RULE #3 — NO CODE MEMORIZATION
==================================================

Do NOT teach me:

“File X has function Y on line Z.”

That is NOT my goal.

Instead teach me:

WHAT was implemented
→ WHY it was implemented
→ HOW it logically works
→ WHAT data goes in
→ WHAT comes out
→ HOW it connects to the next step
→ WHY the step matters

Whenever code is important, convert it into SIMPLE PSEUDOCODE.

Example:

Instead of explaining:

df.groupby("run").shift(1)

Explain:

PSEUDOCODE:
For each simulation run:
    take the previous time-step value
    store it as a new feature

Then explain:
“This allows the model to use past system behavior.”

Only show actual Python syntax when it is specifically useful for a Data Analyst interview.

==================================================
ABSOLUTE RULE #4 — DATA ANALYST LENS
==================================================

I am preparing for a DATA ANALYST role, NOT an ML Engineer role.

Therefore prioritize:

1. Understanding the dataset
2. Data collection / generation
3. Data structure
4. Data cleaning
5. Data validation
6. Exploratory Data Analysis
7. Statistical thinking
8. Feature engineering
9. Time-series understanding
10. Data leakage
11. Train/test methodology
12. Model evaluation
13. Interpretation of results
14. Data-driven conclusions
15. Limitations and improvements

Treat these as SECONDARY:

- deep learning mathematics
- neural-network internals
- LSTM gate equations
- advanced power-system derivations
- low-level TensorFlow/PyTorch implementation

Only teach deeper ML/power-system theory when it can realistically be asked because of something on my resume/project.

==================================================
IMPORTANT CONTEXT ABOUT ME
==================================================

I am a FRESHER.

I built this project approximately one year ago and currently remember only parts of it.

Therefore:

Do NOT assume I already understand the project.

Do NOT use unnecessarily advanced language.

Explain concepts as if:
“I understand basic Python and basic ML, but I need to reconstruct my own project.”

Use simple English.

You may occasionally use simple Hindi/Hinglish-style explanations where they make a difficult concept easier, but keep technical terminology in English because I will use English in the interview.

==================================================
PROJECT RECONSTRUCTION PHASE
==================================================

Before creating the 4-day modules, internally reconstruct this architecture:

PROJECT PURPOSE
↓
ENGINEERING / BUSINESS PROBLEM
↓
SYSTEM / SIMULATION
↓
DATA GENERATION
↓
RAW DATA
↓
DATASET STRUCTURE
↓
DATA CLEANING
↓
EDA
↓
FEATURE ENGINEERING
↓
FEATURE SELECTION
↓
TRAIN/TEST STRATEGY
↓
MODEL INPUT
↓
MODELS
↓
PREDICTIONS
↓
EVALUATION
↓
BEST MODEL
↓
FINAL CONCLUSION

If the actual project differs, use the actual project flow.

==================================================
CREATE AN “ACTUAL PROJECT MAP”
==================================================

Before the 4 modules, create a compact PROJECT MAP containing:

1. One-sentence project explanation
2. Problem statement
3. Objective
4. Input
5. Output / target
6. Data source
7. Dataset size
8. Important features
9. Data-processing steps
10. EDA methods
11. Feature-engineering steps
12. Models
13. Evaluation metrics
14. Best result
15. End-to-end flow
16. Architecture diagram in ASCII
17. One simple real-world analogy
18. Top 10 facts I absolutely must remember

Keep this concise enough for quick revision.

==================================================
CREATE A 4-DAY CURRICULUM
==================================================

I have ONLY 4 DAYS.

Generate EXACTLY 4 modules:

DAY 1
DAY 2
DAY 3
DAY 4

I will study ONLY these four modules for this project.

Do NOT tell me to study anything outside the four modules.

Every important interview-relevant concept must be included somewhere in these four days.

==================================================
DAY 1 — PROJECT + SYSTEM + DATA
==================================================

Cover:

A. Project in simple words
B. Problem being solved
C. Why the problem matters
D. What AGC is — only the amount needed for interview
E. Why decentralized backup was considered
F. Actual system architecture
G. MATLAB/Simulink role
H. How the simulation produces data
I. Number of runs/scenarios if verified
J. Dataset size
K. Dataset columns/features
L. What one row represents
M. What one timestamp represents
N. Input vs target
O. Complete data flow
P. End-to-end architecture
Q. What happens from simulation to final prediction

For every major component explain:

WHAT
WHY
HOW
OUTPUT

Then give:

“30-second explanation”

“60-second explanation”

“2-minute explanation”

INTERVIEW QUESTIONS:
Give likely Day-1 project questions with short, simple model answers.

==================================================
DAY 2 — DATA ANALYSIS + EDA + FEATURE ENGINEERING
==================================================

This is the MOST IMPORTANT DAY for a Data Analyst interview.

Cover the actual implementation of:

A. Data loading
B. Data structure
C. Data cleaning
D. Missing values
E. Duplicates
F. Outliers
G. Data validation
H. Grouping
I. Time-series handling
J. EDA
K. Distributions
L. Correlation
M. Visualizations
N. Feature-target relationships
O. Feature selection
P. Feature engineering
Q. Lag features
R. Frequency deviation
S. RoCoF
T. Any other actual engineered features
U. Why these features were useful
V. Group-aware operations
W. Cross-run leakage
X. How the dataset changed from raw → processed

DO NOT teach generic EDA if it is not related to what actually happened in my project.

For every EDA step:

1. What did I do?
2. Why did I do it?
3. What did I find?
4. Why did the finding matter?
5. What action did I take afterward?

VERY IMPORTANT:
Separate:
“EDA method”
from
“actual insight discovered.”

Do not say I discovered something unless the project evidence supports it.

Include simplified pseudocode for the important data-processing logic.

Example format:

PSEUDOCODE:
Load dataset
↓
Separate simulation runs
↓
Check missing values
↓
Create derived variables
↓
Create lagged values within each run
↓
Remove invalid rows
↓
Use processed data for modeling

Then explain each step in simple words.

INTERVIEW QUESTIONS:
Include beginner, medium, and tricky DA questions related to this exact data pipeline.

==================================================
DAY 3 — MODELING + VALIDATION + METRICS
==================================================

Cover ONLY what is actually present in the project.

For each model:

Random Forest
DNN
RNN
LSTM
OR actual models found in the files

Explain:

1. What is it?
2. Why was it used?
3. What kind of input does it take?
4. What does it predict?
5. Why was it suitable/not suitable?
6. How was it evaluated?
7. Actual result
8. What its result means

Explain the difference between:

Random Forest
vs
DNN
vs
RNN
vs
LSTM

Use simple comparison tables.

Then cover:

- train/test split
- validation
- sequence creation
- scaling
- normalization
- leakage prevention
- model training
- loss
- evaluation
- R²
- RMSE
- MAE
- actual project metric values

Explain all mathematical formulas only to the extent required for interviews.

For each formula:

1. Formula
2. What each symbol means
3. Simple numerical example
4. One-line interview explanation

VERY IMPORTANT:
Do NOT call R² “accuracy” if this is a regression task.

Explain exactly what the actual result means.

If LSTM performed best, explain:
“What evidence shows it performed best?”
not:
“Why LSTM is always better.”

Include model comparison table.

Include pseudocode for the modeling pipeline:

Data
↓
Preprocess
↓
Create X/y
↓
Split
↓
Train
↓
Predict
↓
Calculate metrics
↓
Compare models

INTERVIEW QUESTIONS:
At least 25 project-specific questions.

==================================================
DAY 4 — INTERVIEW DEFENSE + PROJECT MASTER REVISION
==================================================

Day 4 should NOT introduce a huge amount of new theory.

It should convert the previous 3 days into interview readiness.

Create:

PART A — PERFECT PROJECT EXPLANATION

Give me:

30-second answer
60-second answer
90-second answer
2-minute answer

All based ONLY on verified implementation.

PART B — PROJECT WALKTHROUGH

Create this exact sequence:

Problem
→ Objective
→ Architecture
→ Data generation
→ Dataset
→ Cleaning
→ EDA
→ Feature engineering
→ Train/test
→ Models
→ Evaluation
→ Results
→ Conclusion
→ Limitations

For each step, give me a 1–2 sentence explanation.

PART C — TOP 50 INTERVIEW QUESTIONS

Create exactly 50 questions.

Divide them into:

1–10: Project basics
11–20: Data / Pandas / EDA
21–30: Feature engineering / time-series
31–40: ML / metrics
41–45: AGC/domain questions
46–50: Tricky/challenging questions

For EVERY question provide:

Question
↓
Best beginner-friendly answer
↓
One follow-up question interviewer may ask
↓
One-line answer to follow-up

PART D — “INTERVIEWER ATTACKS”

Give me difficult questions such as:

“Why did you choose this feature?”
“Why not another feature?”
“Why not random train/test split?”
“How do you know leakage was avoided?”
“What exactly does one row represent?”
“What was your target?”
“What if your model sees a new disturbance?”
“Why did LSTM outperform Random Forest?”
“Why didn't DNN perform as well?”
“What does R² = X actually mean?”
“Could your model be overfitting?”
“Is your data real or simulated?”
“What are the limitations of simulated data?”
“Would this work on real power-grid data?”
“What would you improve if you had more time?”

Provide honest answers based on the actual project.

PART E — RAPID REVISION SHEET

At the end, create a VERY compact one-page-style memory sheet:

PROJECT:
1 line

PROBLEM:
1 line

DATA:
key numbers

FEATURES:
key names + one-line meaning

EDA:
key techniques

MODELS:
4 models + one-line purpose

METRICS:
R² / RMSE / MAE

BEST MODEL:
result

BIGGEST INSIGHT:
1 line

BIGGEST LIMITATION:
1 line

TOP 10 THINGS TO REMEMBER:
10 short bullets

PART F — “WHAT NOT TO SAY”

List statements I should NOT say because they would be technically incorrect, exaggerated, or unsupported by the actual code.

Examples:
- Calling R² accuracy
- Claiming business impact without evidence
- Claiming production deployment
- Claiming real-world grid validation
- Claiming causation from correlation
- Claiming LSTM is always better

Add project-specific mistakes based on the actual files.

==================================================
LEARNING STYLE
==================================================

This is extremely important.

I want the modules designed for FAST MEMORY RETENTION.

Use the following pattern for every important concept:

### Concept
Simple definition.

### Why
Why we needed it in MY project.

### How
Simple logical explanation.

### Example
Very small example.

### In My Project
Exactly how it was used.

### Interview Answer
One concise answer I can speak.

### Remember
One memory line.

Example:

CONCEPT:
Lag feature

WHY:
Current system condition may depend on recent past behavior.

HOW:
Use previous time-step value as another feature.

EXAMPLE:
Current frequency = 49.8
Previous frequency = 49.9
Lag feature = 49.9

IN MY PROJECT:
[actual verified implementation]

INTERVIEW ANSWER:
“I used previous time-step values as additional features so the model could use recent system behavior.”

REMEMBER:
“Lag = past value used to understand current behavior.”

Use this teaching pattern consistently.

==================================================
USE SIMPLE VISUALS
==================================================

Whenever useful, use ASCII diagrams.

Example:

RAW DATA
   ↓
CLEANING
   ↓
EDA
   ↓
FEATURE ENGINEERING
   ↓
TRAIN/TEST
   ↓
MODEL
   ↓
PREDICTION
   ↓
EVALUATION

For architecture:

SIMULINK
   ↓
AGC SYSTEM
   ↓
SIMULATION RUNS
   ↓
TIME-SERIES DATA
   ↓
PYTHON
   ↓
PANDAS / NUMPY
   ↓
EDA
   ↓
FEATURE ENGINEERING
   ↓
ML MODELS
   ↓
METRICS
   ↓
BEST MODEL

Replace this with the ACTUAL architecture discovered from the project.

==================================================
PSEUDOCODE REQUIREMENT
==================================================

Do NOT give large blocks of actual code.

Instead provide pseudocode for important parts such as:

1. Data loading
2. Cleaning
3. Feature creation
4. Lag creation
5. Dataset splitting
6. Sequence creation
7. Model training
8. Prediction
9. Metric calculation
10. Model comparison

Example:

FOR each simulation run:
    sort data by time
    calculate required features
    create lag values
    remove rows with invalid lag values
END

Then explain:
“This ensures lag values do not accidentally come from another simulation run.”

Use actual project logic, not generic pseudocode.

==================================================
DATA ANALYST INTERVIEW SCOPE
==================================================

Because this is for an on-campus Data Analyst role, connect relevant project concepts to common DA interview themes where appropriate:

Python
Pandas
NumPy
EDA
Data cleaning
Data validation
Filtering
Grouping
Aggregation
Missing values
Duplicates
Outliers
Correlation
Feature engineering
Time-series analysis
Train/test split
Data leakage
Regression
Model evaluation
Visualization
Interpretation of results
Basic statistics

Do NOT force unrelated topics into the project.

For example:
Do not teach SQL inside this project unless the project actually uses SQL.

However, if the project demonstrates a concept that commonly appears in DA interviews, explicitly label it:

“DATA ANALYST INTERVIEW CONNECTION”

==================================================
FRESHER REALISM
==================================================

I am NOT claiming professional production experience.

Therefore explain the project as an academic/student project.

Do not create fake:
- business stakeholders
- business KPIs
- ROI
- cost savings
- production deployment
- live grid deployment
- real company usage

If the project is simulation-based, clearly explain:

Why simulation was used
What simulation can show
What simulation cannot prove
How real-world validation would differ

==================================================
DIFFICULTY CONTROL
==================================================

Each module should have 3 levels:

LEVEL 1 — MUST KNOW
Questions almost certainly answerable from my resume/project.

LEVEL 2 — SHOULD KNOW
Reasonable follow-up questions.

LEVEL 3 — BONUS
Only if interviewer goes deeper.

I have limited time.

Therefore spend approximately:

70% → Level 1
25% → Level 2
5% → Level 3

Do NOT overwhelm me with advanced theory.

==================================================
MEMORIZATION STRUCTURE
==================================================

At the end of EACH DAY provide:

1. 10 key concepts
2. 10 one-line definitions
3. 10 interview answers
4. 5 “why” questions
5. 5 confusing concepts compared side-by-side
6. 5 things I am most likely to forget
7. 5-minute revision sequence

Use repetition deliberately.

Important concepts should reappear briefly on later days so I remember them.

==================================================
NO UNNECESSARY DETAIL
==================================================

Do not explain every file.

Do not explain every function.

Do not explain every line of code.

Do not explain implementation details that an interviewer is unlikely to ask.

Every section should answer:

“Will knowing this help me defend my project in a DA interview?”

If NO → omit it.

==================================================
FINAL PROJECT CONSISTENCY CHECK
==================================================

At the end of all 4 modules, produce:

PROJECT FACT CHECK

Table:

Claim | Verified? | Evidence/location | Safe to say in interview?

Check especially:

- dataset size
- number of features
- number of simulation runs
- target variable
- lag features
- feature-selection method
- train/test split
- number of models
- ML frameworks
- metrics
- exact scores
- best model
- any claimed improvement
- preprocessing
- scaling
- leakage prevention

This section is extremely important because I will use the material directly for interviews.

==================================================
FINAL “SURVIVAL TEST”
==================================================

After completing the 4 modules, give me a final test of 30 questions.

Do NOT give answers immediately.

Questions should test whether I actually understand the project.

Divide:

10 easy
10 medium
10 difficult

At the end provide answers separately so I can test myself first.

==================================================
OUTPUT RULE
==================================================

DO NOT give me one huge generic explanation.

First:
1. Investigate the project.
2. Reconstruct the actual architecture.
3. Verify the facts.
4. Identify gaps/discrepancies.
5. Then generate exactly 4 study modules.

The four modules must be written so that I can study ONLY those modules for the next 4 days.

At the beginning of each module show:

DAY X
Goal
What I must be able to explain by the end
Estimated study priority

At the end show:

“DAY X COMPLETE IF YOU CAN ANSWER THESE 10 QUESTIONS”

==================================================
MOST IMPORTANT REQUIREMENT
==================================================

I don't need to become an ML engineer.

I need to become the person who can confidently say:

“I built this project, I understand the data, I understand the complete flow, I understand why each major step was performed, I understand the results and limitations, and I can explain it clearly to a Data Analyst interviewer.”

Make the preparation optimized for THAT outcome.

Start by investigating the actual project files now.

explain like explaining to a kid.
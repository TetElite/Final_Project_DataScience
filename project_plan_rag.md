# Project Plan – Pain Medication Effectiveness Predictor

---

## 1. Project Snapshot

| Field | Detail |
|---|---|
| **Title** | Pain Medication Effectiveness Predictor |
| **Objective** | Build a machine learning model that predicts whether a specific pain medication will be effective for a given patient (3-level classification: Effective, Partially Effective, Not Effective) |
| **Dataset** | Kaggle Drug Review Dataset (filtered); 5,000+ patient reviews; Columns: drugName, condition, review, rating, date; Pain-related conditions only (headache, back pain); Common pain medications only (Ibuprofen, Acetaminophen) |
| **Expected Output** | Working prediction model (target accuracy: 75%+), confidence scores, patient insights on key factors (age, condition, drug), key findings on which conditions/medications work best |
| **Graded By** | Introduction (background, problem importance, dataset explanation), Problem statement, Project objectives, Dataset description, Data preprocessing/cleaning, EDA (summary statistics + visualizations with insights), Modeling (model type, choice justification, train/test split), Results and evaluation (accuracy, metrics), Discussion/Conclusion (key findings, feature importance, objectives achieved) |
| **RAG Available** | Yes |

---

## 2. Requirements Checklist

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | Introduction (context, importance, dataset/analysis brief) | ⚠️ | Proposal has problem statement and dataset description but needs formal introduction section |
| 2 | Problem statement | ✅ | Clear problem defined: no tool to predict medication effectiveness based on patient profile |
| 3 | Project objective (clear goals list) | ✅ | Main goal stated; specific outputs defined (prediction, insights, findings) |
| 4 | Dataset description (source, records, features, variables) | ✅ | Source: Kaggle; 5k+ rows; columns listed; filtered version described |
| 5 | Data preprocessing/Data cleaning (missing values, duplicates, encoding, scaling) | ⚠️ | Mentioned in methodology but lacks specific techniques detail |
| 6 | EDA - Summary statistics | ❌ | Not mentioned in proposal |
| 7 | EDA - Data visualization (histogram, boxplot, scatter, heatmap) with insights | ⚠️ | Mentioned "STEP 3: FEATURE ENGINEERING" but no specific EDA visualizations listed |
| 8 | Hypothesis/Research questions (optional for structured data) | ❌ | Not included (optional but recommended) |
| 9 | Modeling (model type, why chosen, training process, train/test split) | ⚠️ | "MODEL TRAINING" mentioned but no specific model type or justification |
| 10 | Results and evaluation (accuracy, metrics, confusion matrix if classification) | ⚠️ | Target accuracy 75%+ mentioned but no specific metrics listed |
| 11 | Discussion (feature importance, insights discovered) | ✅ | Patient insights and key findings sections cover this |
| 12 | Conclusion (key findings summary, objectives achieved) | ❌ | Not explicitly structured in proposal |

---

## 3. Gap Analysis

**Covered:**
- Clear problem statement with real-world context (trial-and-error medication selection)
- Well-defined 3-level classification objective with target accuracy
- Dataset identified with appropriate filtering strategy for pain medications

**Missing:**
- Formal introduction section combining context + importance + brief analysis overview
- Explicit EDA plan with specific visualization types (histogram, boxplot, scatter, correlation heatmap)
- Summary statistics plan (mean, median, mode, distribution analysis)
- Specific model selection with justification (e.g., Random Forest, Logistic Regression, Decision Tree)
- Detailed evaluation metrics beyond accuracy (confusion matrix, precision, recall, F1-score) [Inferred - classification task requires these]
- Hypothesis/research questions (optional but valuable for structured data)
- Formal conclusion section structure

**Conflicts:**
None found.

---

## 4. Action Plan

### Milestone Overview

| # | Milestone | Est. Time | Key Output |
|---|---|---|---|
| 1 | Data Collection & Filtering | 2-3 hours [Inferred] | Filtered dataset CSV (pain conditions + pain medications only) |
| 2 | Data Cleaning | 4-6 hours [Inferred] | Cleaned dataset with handled missing values, standardized text, removed duplicates |
| 3 | Exploratory Data Analysis | 3-4 hours [Inferred] | Summary statistics report + 4-6 visualizations with written insights |
| 4 | Feature Engineering | 3-5 hours [Inferred] | Engineered features dataset ready for modeling |
| 5 | Model Training & Evaluation | 4-6 hours [Inferred] | Trained model, evaluation metrics report, confusion matrix |
| 6 | Insights & Documentation | 2-3 hours [Inferred] | Final report/slides with key findings and patient insights |

---

### Task Breakdown

#### Milestone 1 – Data Collection & Filtering

**Task 1.1 – Download and Load Dataset**
- **Do:** Download "Drug Review Dataset" from Kaggle; load into pandas DataFrame using `pd.read_csv()`
- **Technique:** `pd.read_csv()` `[Ref: class_summary_rag.md – Lesson 9 W9_Feature_Engineering]`
- **Output:** Raw dataset loaded as DataFrame
- **Flag:** None

**Task 1.2 – Filter Dataset**
- **Do:** Keep only rows where condition contains pain-related terms (headache, back pain, etc.); keep only rows where drugName is common pain medication (Ibuprofen, Acetaminophen, etc.)
- **Technique:** Boolean indexing with pandas `[Ref: class_summary_rag.md – Lesson 2 Week2_Python_programming]`
- **Output:** filtered_dataset.csv (5k+ rows focused on pain medications)
- **Flag:** Text matching for conditions may need case-insensitive comparison

#### Milestone 2 – Data Cleaning

**Task 2.1 – Handle Missing Values**
- **Do:** Check for missing values with `df.isnull()`; for numeric columns use median imputation; for categorical use mode imputation or deletion if minimal
- **Technique:** `df.isnull()`, `statistics.median()`, `statistics.mode()`, `df.dropna()`, `df.fillna()` `[Ref: class_summary_rag.md – Lesson 4 Week4_Data Pre-processing]`
- **Output:** DataFrame with no missing values
- **Flag:** None

**Task 2.2 – Standardize Text in 'condition' Column**
- **Do:** Convert all text to lowercase; remove extra spaces; standardize variations (e.g., "Head ache" → "headache")
- **Technique:** String manipulation with `.str.lower()`, `.str.strip()`, `.str.replace()` `[Safe Extension - string methods derived from pandas taught in Lesson 2]`
- **Output:** Standardized condition column
- **Flag:** None

**Task 2.3 – Remove Duplicates**
- **Do:** Identify and remove duplicate rows using `drop_duplicates()`
- **Technique:** `df.drop_duplicates()` `[Ref: class_summary_rag.md – Lesson 4 Week4_Data Pre-processing]`
- **Output:** Deduplicated dataset
- **Flag:** None

**Task 2.4 – Detect and Handle Outliers**
- **Do:** Use IQR method on rating column; visualize with boxplot to identify outliers; decide whether to cap or remove
- **Technique:** `quantile()`, IQR calculation (Q3-Q1) `[Ref: class_summary_rag.md – Lesson 4 Week4_Data Pre-processing]`
- **Output:** Cleaned dataset with outliers handled
- **Flag:** None

#### Milestone 3 – Exploratory Data Analysis

**Task 3.1 – Generate Summary Statistics**
- **Do:** Calculate mean, median, mode, std deviation, min/max for rating column; count distribution for drugName and condition columns
- **Technique:** `df.describe()`, `mean()`, `median()`, `mode()`, `std()` `[Ref: class_summary_rag.md – Lesson 3 Week3_Descriptive_Statistics and Lesson 2 Week2_Python_programming]`
- **Output:** Summary statistics table
- **Flag:** None

**Task 3.2 – Create Distribution Visualizations**
- **Do:** Create histogram for rating distribution; create bar chart for top conditions and top drugs
- **Technique:** `matplotlib` or `seaborn` with histogram and bar charts `[Ref: class_summary_rag.md – Lesson 6 Week6_Data Visualization]`
- **Output:** 2-3 distribution plots with written insights
- **Flag:** None

**Task 3.3 – Create Relationship Visualizations**
- **Do:** Create boxplot showing rating distribution by condition; create scatter plot if age data available (age vs rating); create heatmap for correlation matrix
- **Technique:** `plot()` with `scatter()`, `box()`, `bar()`, `hist()` extensions `[Ref: class_summary_rag.md – Lesson 6 Week6_Data Visualization]`
- **Output:** 3-4 relationship plots
- **Flag:** Age data not mentioned in proposal - confirm availability or skip scatter plot

**Task 3.4 – Document Insights**
- **Do:** Write 2-3 sentences per visualization explaining patterns (e.g., "Ibuprofen shows higher effectiveness ratings for headaches compared to back pain")
- **Technique:** Analytical observation from visualizations
- **Output:** Insights document (markdown or text)
- **Flag:** None

#### Milestone 4 – Feature Engineering

**Task 4.1 – Create Effectiveness Target Variable**
- **Do:** Convert rating (likely 1-10 scale) into 3 categories: Effective (rating >= 7), Partially Effective (4-6), Not Effective (<= 3) [Inferred thresholds]
- **Technique:** `pd.cut()` for binning `[Ref: class_summary_rag.md – Lesson 9 W9_Feature_Engineering]`
- **Output:** New 'effectiveness' column with 3 classes
- **Flag:** Confirm rating scale and thresholds with teacher

**Task 4.2 – Encode Categorical Variables**
- **Do:** Apply one-hot encoding to drugName and condition columns using `pd.get_dummies()`
- **Technique:** `pd.get_dummies()` `[Ref: class_summary_rag.md – Lesson 9 W9_Feature_Engineering]`
- **Output:** Encoded features ready for ML model
- **Flag:** None

**Task 4.3 – Feature Scaling (if needed)**
- **Do:** If age or other numeric features exist, apply standardization or min-max scaling
- **Technique:** `StandardScaler()` or `MinMaxScaler()` from sklearn `[Ref: class_summary_rag.md – Lesson 9 W9_Feature_Engineering]`
- **Output:** Scaled feature set
- **Flag:** Only if numeric features beyond rating exist

**Task 4.4 – Extract Text Features from Review**
- **Do:** Extract review length, presence of keywords ("effective", "pain relief"), or sentiment indicators [Inferred - text review mentioned in dataset]
- **Technique:** String operations `.str.len()`, `.str.contains()` `[Safe Extension - derived from pandas string methods in Lesson 2]`
- **Output:** Additional engineered features (review_length, has_positive_keywords, etc.)
- **Flag:** Basic text features only - NLP is mentioned in course but advanced techniques not detailed

#### Milestone 5 – Model Training & Evaluation

**Task 5.1 – Split Data into Train/Test Sets**
- **Do:** Use 80/20 or 70/30 train-test split; ensure stratification by effectiveness class
- **Technique:** Train/test split (structure described in final_project_structure.txt) `[Ref: class_summary_rag.md – Lesson 8 Week8_Introduction_to_Machine_Learning_Classification]`
- **Output:** X_train, X_test, y_train, y_test
- **Flag:** `[Requires install: scikit-learn]` - though sklearn is mentioned in class_summary_rag.md

**Task 5.2 – Train Classification Model**
- **Do:** Train a classification model (Random Forest or Logistic Regression recommended for multi-class) [Inferred]; fit model on training data
- **Technique:** `RandomForestClassifier()` or `LogisticRegression()` with `.fit()` `[Ref: class_summary_rag.md – Lesson 9 W9_Feature_Engineering and Lesson 10 Week10_Linear Regression]`
- **Output:** Trained model object
- **Flag:** Random Forest for multi-class mentioned in W9; Logistic Regression in W9 but typically for binary - confirm multi-class support or use Random Forest

**Task 5.3 – Make Predictions**
- **Do:** Use trained model to predict effectiveness on test set; generate confidence scores (predicted probabilities)
- **Technique:** `.predict()` and `.predict_proba()` `[Safe Extension - derived from model.fit() in Lesson 10]`
- **Output:** Predictions array and confidence scores
- **Flag:** None

**Task 5.4 – Evaluate Model Performance**
- **Do:** Calculate accuracy (target 75%+); generate confusion matrix; calculate precision, recall, F1-score for each class
- **Technique:** Accuracy, Confusion Matrix, metrics `[Ref: class_summary_rag.md – Lesson 8 Week8_Introduction_to_Machine_Learning_Classification]`
- **Output:** Evaluation metrics report
- **Flag:** Specific metric calculation functions (e.g., sklearn.metrics) implied but not explicitly detailed in RAG - mark as `[Safe Extension - standard evaluation for classification models in Lesson 8]`

**Task 5.5 – Analyze Feature Importance**
- **Do:** Extract feature importance scores from model (if Random Forest used); identify top factors influencing predictions (age, condition, drug type)
- **Technique:** `.feature_importances_` attribute `[Safe Extension - natural output from RandomForestClassifier in Lesson 9]`
- **Output:** Feature importance ranking
- **Flag:** Only available if tree-based model used

#### Milestone 6 – Insights & Documentation

**Task 6.1 – Generate Patient Insights**
- **Do:** Analyze which patient traits matter most (from feature importance); identify which conditions respond best to medication; determine if age affects effectiveness
- **Technique:** Analysis of feature importance + cross-tabulation of predictions by category
- **Output:** Patient insights summary (2-3 key findings)
- **Flag:** None

**Task 6.2 – Create Example Predictions**
- **Do:** Generate sample predictions showing format: "Ibuprofen for headache → EFFECTIVE (87% confident)"
- **Technique:** Apply trained model to sample inputs with confidence scores
- **Output:** 5-10 example predictions
- **Flag:** None

**Task 6.3 – Write Final Report/Slides**
- **Do:** Structure following final_project_structure.txt: Introduction, Problem Statement, Objectives, Dataset Description, Data Cleaning, EDA, Modeling, Results, Discussion, Conclusion
- **Technique:** Markdown or presentation software
- **Output:** Final project report and/or slides
- **Flag:** None

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dataset too small after filtering (< 1000 rows for 3 classes) | Medium | Check row count after filtering; if low, expand pain condition keywords or include more medication types; ensure minimum 200-300 samples per class |
| Unbalanced classes (e.g., 90% "Effective", 5% each other classes) | Medium | Check class distribution with `value_counts()`; if imbalanced, use stratified sampling in train/test split; consider SMOTE or class weighting [SMOTE not in RAG - use stratified split only] |
| Text inconsistency too complex for simple cleaning | Medium | Use iterative cleaning: lowercase → strip spaces → manual inspection of `unique()` values → create mapping dictionary for variations |
| Age or other patient demographic data not available in dataset | High | Confirm dataset columns immediately after download; if missing, adjust feature engineering to focus on review text, drug type, and condition only |
| Model accuracy below 75% target | Medium | Try multiple models (start with Random Forest, try Logistic Regression); engineer more text features from review; ensure proper encoding and scaling; use cross-validation to tune |

---

## 6. Teacher Clarification Needed

| # | Item | Question to Ask |
|---|---|---|
| 1 | Rating scale and effectiveness thresholds | "What is the rating scale in the dataset (1-5 or 1-10)? What thresholds should define Effective, Partially Effective, Not Effective?" |
| 2 | Patient demographic data availability | "Does the Kaggle Drug Review Dataset include patient age, gender, or other demographics, or only drugName, condition, review, rating, date?" |
| 3 | Time estimates for milestones | "Are the estimated hours per milestone (2-6 hours each) reasonable for the project scope, or should we adjust?" |
| 4 | Hypothesis testing requirement | "The structure guide lists hypothesis/research questions as optional. Should we include them (e.g., H1: Ibuprofen is more effective than Acetaminophen for headaches)?" |
| 5 | Multi-class classification model preference | "Should we use Random Forest, Logistic Regression, or another classifier for 3-class prediction? Any preference?" |
| 6 | Review text analysis depth | "How deeply should we analyze the 'review' text column? Basic features (length, keywords) or more advanced NLP?" |

---

## 7. RAG Reference Index

| Lesson / Section | Topic Summary | Used In Task(s) |
|---|---|---|
| Lesson 2: Week2_Python_programming | Python fundamentals, pandas DataFrames, Series operations, basic functions | Task 1.1, 1.2, 2.1, 2.3, 3.1 |
| Lesson 3: Week3_Descriptive_Statistics | Mean, median, mode, variance, standard deviation, quartiles, correlation | Task 3.1 |
| Lesson 4: Week4_Data Pre-processing | Missing value imputation (mean/median/mode), outlier detection (IQR), duplicate removal, data cleaning | Task 2.1, 2.3, 2.4 |
| Lesson 6: Week6_Data Visualization | Matplotlib, seaborn, plotly; chart types (bar, line, histogram, scatter, box plot, heatmap) | Task 3.2, 3.3 |
| Lesson 8: Week8_Introduction_to_Machine_Learning_Classification | Supervised learning, classification types (binary, multi-class), train/test split, evaluation metrics (accuracy, confusion matrix, ROC, AUC, Recall) | Task 5.1, 5.4 |
| Lesson 9: W9_Feature_Engineering | Feature encoding (one-hot, label encoding), binning (pd.cut, pd.qcut), scaling (StandardScaler, MinMaxScaler), PCA, KNN imputation, Random Forest, Logistic Regression | Task 4.1, 4.2, 4.3, 5.2 |
| Lesson 10: Week10_Linear Regression | Model training with sklearn, .fit() method, model evaluation, intercept and coefficients | Task 5.2, 5.3 |

---

**End of Project Plan**

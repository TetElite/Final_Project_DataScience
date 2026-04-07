# Pain Medication Effectiveness Predictor

**Team 3:** Choeng Rayu, Tep Somnang, Tet Elite, Sophal Taingchhay  
**Lecturer:** Kim Sokhey  
**Course:** Data Science - Final Project  
**Academic Year:** 2025-2026

---

## 📋 Problem Statement

Same pain medication works differently for different people. Doctors use trial-and-error which:
- Delays patient relief
- Wastes time and money
- Leaves patients uncertain about treatment effectiveness

**Current Limitation:** No simple tool to predict medication effectiveness based on patient profile.

---

## 🎯 Project Objective

Build a machine learning model that predicts whether a specific pain medication will be **Effective**, **Partially Effective**, or **Not Effective** for a given patient.

**Target Accuracy:** 75% or higher  
**Achieved Accuracy:** 68.28% (338 out of 495 correct predictions)

---

## 📊 Dataset

- **Source:** Kaggle - [UCI ML Drug Review Dataset (Drugs.com)](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
- **Original Size:** 280,478 patient reviews
  - drugsComTrain_raw.csv: 209,989 reviews
  - drugsComTest_raw.csv: 70,489 reviews
- **Filtered Size:** 2,975 pain medication reviews
- **Final Cleaned Size:** 2,473 reviews (after removing duplicates)
- **ML-Ready Dataset:** 2,473 rows × 65 features

### Dataset Columns

- `drugName` - Name of the medication
- `condition` - Medical condition being treated
- `review` - Patient's written review
- `rating` - Patient satisfaction rating (1-10 scale)
- `date` - Review date
- `usefulCount` - Number of users who found review useful

### Filtering Criteria

**Pain Conditions:** headache, migraine, back pain, arthritis, sciatica, fibromyalgia, toothache, neck pain, joint pain, osteoarthritis, rheumatoid arthritis, chronic pain, neuropathic pain, muscle pain

**Pain Medications:** ibuprofen, acetaminophen, naproxen, aspirin, diclofenac, tramadol, hydrocodone, oxycodone, tylenol, advil, aleve, motrin, celebrex, meloxicam, indomethacin

---

## 🏗️ Project Structure

```
project/
├── 01_data/                          # 📊 ALL DATASETS
│   ├── raw/                          # Original unprocessed data (280K reviews)
│   │   ├── drugsComTrain_raw.csv    # 209,989 reviews
│   │   └── drugsComTest_raw.csv     # 70,489 reviews
│   ├── filtered/                     # Pain-specific filtered data (2,975 reviews)
│   │   └── pain_meds_filtered.csv
│   ├── cleaned/                      # Cleaned and validated (2,473 reviews)
│   │   └── pain_meds_cleaned.csv
│   └── processed/                    # ML-ready features (2,473 × 65)
│       └── pain_meds_ml_ready.csv
│
├── 02_notebooks/                     # 📓 JUPYTER ANALYSIS NOTEBOOKS
│   ├── 01_data_collection.ipynb     # Data loading from UCI
│   ├── 02_data_cleaning.ipynb       # Data cleaning pipeline
│   ├── 03_eda.ipynb                 # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb # Feature creation
│   ├── 05_modeling.ipynb            # Model training
│   └── 06_final_analysis.ipynb      # Results and insights
│
├── 03_source_code/                   # 💻 PYTHON SCRIPTS
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Load raw datasets
│   │   ├── cleaning.py              # Clean text, handle missing values
│   │   └── feature_engineering.py   # Create ML features
│   ├── model_training/
│   │   ├── __init__.py
│   │   ├── train_model.py           # Train Random Forest
│   │   └── evaluate_model.py        # Calculate metrics
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting functions
│
├── 04_trained_models/                # 🤖 ⭐ AI MODELS HERE ⭐
│   ├── README.md                     # Model documentation
│   ├── random_forest_v1/
│   │   ├── rf_model.pkl             # ⭐ MAIN MODEL FILE (11 MB)
│   │   ├── feature_names.pkl        # List of 55 feature names
│   │   └── feature_importance.csv   # Feature importance rankings
│   └── test_results/
│       └── test_predictions.csv     # 495 test predictions
│
├── 05_analysis_results/              # 📈 ANALYSIS OUTPUTS
│   ├── statistics/
│   │   ├── summary_statistics.csv   # Overall dataset metrics
│   │   └── data_quality_report.csv  # Missing values, outliers
│   ├── visualizations/
│   │   ├── data_visualizations.png  # EDA plots
│   │   ├── confusion_matrix.png     # Model performance
│   │   └── feature_importance.png   # Top features chart
│   └── insights/
│       ├── top_drugs.csv            # Highest rated medications
│       ├── top_drugs_stats.csv      # Drug effectiveness stats
│       ├── top_conditions.csv       # Most common conditions
│       ├── condition_effectiveness.csv
│       └── top_features.csv         # Feature importance data
│
├── 06_dashboard/                     # 🌐 WEB APPLICATION
│   ├── app.py                        # Main Streamlit app
│   ├── pages/
│   │   ├── 01_home.py               # Project overview
│   │   ├── 02_predictions.py        # Real-time predictions
│   │   ├── 03_model_performance.py  # Metrics & charts
│   │   ├── 04_data_insights.py      # Visualizations
│   │   └── 05_about.py              # Documentation
│   └── assets/
│       ├── logo.png
│       └── styles.css
│
├── 07_documentation/                 # 📚 PROJECT DOCUMENTATION
│   ├── README.md                     # Main project overview
│   ├── PROJECT_SUMMARY.md            # Executive summary
│   ├── PRESENTATION_GUIDE.md         # Slide creation guide
│   ├── API_DOCUMENTATION.md          # How to use the model
│   └── DATA_DICTIONARY.md            # Feature descriptions
│
├── requirements.txt                  # Python package dependencies
├── PROJECT_STRUCTURE.md              # This structure guide
└── README.md                         # This file
```

---

## 🚀 How to Run

### 1. Activate Virtual Environment
```bash
cd /Users/macbook/CADT/Term2Year3/Data/Final_Project
source venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r project/requirements.txt
```

### 3. Launch the Streamlit Dashboard (Primary Interface)
```bash
cd project/06_dashboard
streamlit run app.py
```

The interactive dashboard will open in your browser, providing the easiest way to:
- Make live predictions
- Explore model performance
- Visualize data insights

### 4. Alternative: Use Jupyter Notebooks
```bash
cd project/02_notebooks
jupyter notebook
```

Execute the notebooks sequentially (01 → 06):
1. **01_data_collection.ipynb** - Download and filter dataset
2. **02_data_cleaning.ipynb** - Clean and prepare data
3. **03_eda.ipynb** - Exploratory data analysis
4. **04_feature_engineering.ipynb** - Create features for ML
5. **05_modeling.ipynb** - Train and evaluate model
6. **06_final_analysis.ipynb** - Generate insights and results

### 5. Load the Trained Model Programmatically
```python
import pickle
from pathlib import Path

# Load the model
model_path = Path("project/04_trained_models/random_forest_v1/rf_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load feature names
features_path = Path("project/04_trained_models/random_forest_v1/feature_names.pkl")
with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# Make predictions
prediction = model.predict(X)
probabilities = model.predict_proba(X)
```

---

## 🌐 Streamlit Dashboard

This project includes an **interactive web dashboard** built with Streamlit, providing a user-friendly interface to interact with the Pain Medication Effectiveness Predictor.

### Launch the Dashboard
```bash
cd project/06_dashboard
streamlit run app.py
```

### Key Features
- **Live Predictions** - Enter patient details and medication to get instant effectiveness predictions
- **Interactive Visualizations** - Explore data distributions, medication effectiveness, and condition analysis
- **Model Performance Metrics** - View accuracy, confusion matrix, and classification reports
- **Feature Importance** - Understand which factors most influence predictions
- **Patient Insights** - Analyze how different patient characteristics affect medication effectiveness

### Why Use the Dashboard?
The Streamlit dashboard is the **primary way to interact with the model**. It provides an intuitive interface without requiring code knowledge, making predictions accessible to healthcare professionals and researchers.

---

## 📐 Classification Thresholds

The model classifies medication effectiveness based on patient ratings:

| Effectiveness Level | Rating Range | Description |
|---|---|---|
| **Effective** | 8-10 | High patient satisfaction |
| **Partially Effective** | 5-7 | Moderate patient satisfaction |
| **Not Effective** | 1-4 | Low patient satisfaction |

---

## 📊 Model Performance

### Overall Metrics

| Metric | Value | Details |
|--------|-------|---------|
| **Overall Accuracy** | **68.28%** | 338 correct out of 495 test samples |
| **Weighted Precision** | 65.64% | Average precision across all classes |
| **Weighted Recall** | 68.28% | Average recall across all classes |
| **Weighted F1-Score** | 66.84% | Balanced accuracy and completeness |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Not Effective** | 42.17% | 42.68% | 42.42% | 82 |
| **Partially Effective** | 14.71% | 9.09% | 11.24% | 55 |
| **Effective** | 78.84% | 83.24% | 80.98% | 358 |

### Key Observations

✅ **Strengths:**
- Strong performance on "Effective" class (78.84% precision, 83.24% recall)
- Good overall accuracy of 68.28%, approaching the 75% target
- Minimal overfitting (training: 72.5%, test: 68.28%, gap: 4.22%)

⚠️ **Limitations:**
- **Poor performance on "Partially Effective" class** (only 14.71% precision, 9.09% recall)
  - This middle category is difficult to distinguish from the other two classes
  - Many "Partially Effective" cases are misclassified as "Effective" or "Not Effective"
- Class imbalance: 358 "Effective" samples vs 55 "Partially Effective" samples
- Accuracy falls short of 75% target by 6.72 percentage points

### Model Validation

- **Cross-Validation:** 5-fold CV average accuracy: 67.2%
- **Overfitting Check:** Acceptable 4.2% gap between training (72.5%) and test (68.28%) accuracy
- **Test Set Size:** 495 samples (20% of total dataset)

---

## 🎯 Expected Outcomes

### 1. Prediction Model
- Working Random Forest classification model
- Current accuracy: **68.28%** (target was 75%)
- Confidence scores for each prediction
- 55 engineered features from patient reviews

### 2. Patient Insights
- Which patient traits matter most (medication type, condition, review sentiment)
- How similar patients responded to medications
- What factors influenced predictions

### 3. Key Findings
- Which pain conditions respond best to medication
- Which pain medications work best overall
- The importance of patient review sentiment in predicting effectiveness

### Example Prediction:
```
"Ibuprofen for headache → EFFECTIVE (83% confident)"
```

---

## 📚 Methodology

**STEP 1:** Data Collection & Filtering (280K → 2,975 reviews)  
**STEP 2:** Data Cleaning (2,975 → 2,473 reviews)  
**STEP 3:** Exploratory Data Analysis (EDA)  
**STEP 4:** Feature Engineering (65 features created)  
**STEP 5:** Model Training (Random Forest Classifier)  
**STEP 6:** Evaluation & Insights (68.28% accuracy achieved)

---

## 🛠️ Technologies Used

- **Python 3.14+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Visualizations
- **scikit-learn** - Machine learning
- **nltk** - Natural language processing for sentiment analysis
- **streamlit** - Interactive web dashboard
- **plotly** - Interactive visualizations
- **Jupyter Notebook** - Interactive development

---

## 🔑 Key Files and Locations

| File | Purpose | Location |
|------|---------|----------|
| **Trained Model** | Main AI model (11 MB) | `project/04_trained_models/random_forest_v1/rf_model.pkl` |
| **Training Data** | ML-ready dataset | `project/01_data/processed/pain_meds_ml_ready.csv` |
| **Dashboard** | Web interface | `project/06_dashboard/app.py` |
| **Model Training** | Training notebook | `project/02_notebooks/05_modeling.ipynb` |
| **Documentation** | Project overview | `project/07_documentation/README.md` |

---

## 📈 Data Flow Pipeline

```
Raw Data (280K reviews)
    ↓
Filter (pain medications only)
    ↓ 2,975 reviews
Clean (remove duplicates, handle missing)
    ↓ 2,473 reviews
Process (engineer features, encode)
    ↓ 2,473 × 65 features
Train Model (Random Forest)
    ↓
Evaluate & Save (68.28% accuracy)
    ↓
Deploy (Streamlit Dashboard)
```

---

## 👥 Team Members

- **Choeng Rayu**
- **Tep Somnang**
- **Tet Elite**
- **Sophal Taingchhay**

---

## 📝 Future Improvements

1. **Address Class Imbalance:** Collect more "Partially Effective" samples or use advanced balancing techniques
2. **Improve Feature Engineering:** Add more patient demographic features (age, weight, medical history)
3. **Experiment with Advanced Models:** Try ensemble methods or deep learning approaches
4. **Clinical Validation:** Test with real clinical data and expert validation
5. **Reach Target Accuracy:** Implement improvements to achieve 75%+ accuracy

---

## 📄 License

This project is for educational purposes as part of the CADT Data Science course.

---

## 📞 Contact

For questions or issues, please contact the team members or instructor Kim Sokhey.

---

**Last Updated:** April 7, 2026

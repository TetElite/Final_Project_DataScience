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

---

## 📊 Dataset

- **Source:** Kaggle - [UCI ML Drug Review Dataset (Drugs.com)](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
- **Original Size:** 215,063 patient reviews
- **Filtered Size:** ~5,000+ pain medication reviews
- **Columns:** 
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
├── data/
│   ├── raw/              # Original dataset from Kaggle (NEVER modify)
│   ├── filtered/         # After filtering for pain medications
│   ├── cleaned/          # After data cleaning
│   └── processed/        # Feature-engineered, ready for ML
├── notebooks/            # Jupyter notebooks (run in order 01→06)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_final_analysis.ipynb
├── src/                  # Reusable Python functions
│   ├── __init__.py
│   ├── data_loader.py    # Data loading utilities
│   ├── cleaning.py       # Cleaning functions
│   └── visualization.py  # Plotting helpers
├── outputs/
│   ├── plots/           # Saved visualizations
│   ├── models/          # Trained ML models
│   └── metrics/         # Evaluation results
├── docs/                # Final report and slides
├── requirements.txt     # Python package dependencies
├── README.md           # This file
└── project_plan_rag.md # Detailed implementation plan
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

### 3. Launch Jupyter Notebook
```bash
cd project/notebooks
jupyter notebook
```

### 4. Run Notebooks in Order
Execute the notebooks sequentially (01 → 06):
1. **01_data_collection.ipynb** - Download and filter dataset
2. **02_data_cleaning.ipynb** - Clean and prepare data
3. **03_eda.ipynb** - Exploratory data analysis
4. **04_feature_engineering.ipynb** - Create features for ML
5. **05_modeling.ipynb** - Train and evaluate model
6. **06_final_analysis.ipynb** - Generate insights and results

---

## 📐 Classification Thresholds

The model classifies medication effectiveness based on patient ratings:

| Effectiveness Level | Rating Range | Description |
|---|---|---|
| **Effective** | 8-10 | High patient satisfaction |
| **Partially Effective** | 5-7 | Moderate patient satisfaction |
| **Not Effective** | 1-4 | Low patient satisfaction |

---

## 🎯 Expected Outcomes

### 1. Prediction Model
- Working classification model
- Target accuracy: **75% or higher**
- Confidence scores for each prediction

### 2. Patient Insights
- Which patient traits matter most (age, condition, drug type)
- How similar patients responded to medications
- What factors influenced predictions

### 3. Key Findings
- Which pain conditions respond best to medication
- Whether age affects effectiveness
- Which pain medications work best overall

### Example Prediction:
```
"Ibuprofen for headache → EFFECTIVE (87% confident)"
```

---

## 📚 Methodology

**STEP 1:** Data Collection & Filtering  
**STEP 2:** Data Cleaning  
**STEP 3:** Exploratory Data Analysis (EDA)  
**STEP 4:** Feature Engineering  
**STEP 5:** Model Training (Random Forest Classifier)  
**STEP 6:** Evaluation & Insights  

---

## 🛠️ Technologies Used

- **Python 3.14+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Visualizations
- **scikit-learn** - Machine learning
- **Jupyter Notebook** - Interactive development

---

## 👥 Team Members

- **Choeng Rayu**
- **Tep Somnang**
- **Tet Elite**
- **Sophal Taingchhay**

---

## 📄 License

This project is for educational purposes as part of the CADT Data Science course.

---

## 📞 Contact

For questions or issues, please contact the team members or instructor Kim Sokhey.

---

**Last Updated:** April 3, 2026

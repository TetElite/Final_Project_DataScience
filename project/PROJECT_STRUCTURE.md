# 📁 PROJECT STRUCTURE

## Visual Directory Map

```
Final_Project/project/
│
├── 01_data/                          # 📊 ALL DATASETS
│   ├── raw/                          # Original unprocessed data (280K reviews)
│   │   ├── drugsComTrain_raw.csv    # 209,989 reviews
│   │   └── drugsComTest_raw.csv     # 70,489 reviews
│   │
│   ├── filtered/                     # Pain-specific filtered data (2,975 reviews)
│   │   └── pain_meds_filtered.csv   # Filtered to pain medications
│   │
│   ├── cleaned/                      # Cleaned and validated (2,473 reviews)
│   │   └── pain_meds_cleaned.csv    # Duplicates removed, text cleaned
│   │
│   └── processed/                    # ML-ready features (2,473 × 65)
│       └── pain_meds_ml_ready.csv   # Engineered features for training
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
│   │   └── feature_engineering.py  # Create ML features
│   │
│   ├── model_training/
│   │   ├── __init__.py
│   │   ├── train_model.py           # Train Random Forest
│   │   └── evaluate_model.py        # Calculate metrics
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting functions
│
├── 04_trained_models/                # 🤖 ⭐ AI MODELS HERE ⭐
│   ├── README.md                     # Model documentation
│   │
│   ├── random_forest_v1/
│   │   ├── rf_model.pkl             # ⭐ MAIN MODEL FILE (11 MB)
│   │   ├── feature_names.pkl        # List of 55 feature names
│   │   └── feature_importance.csv   # Feature importance rankings
│   │
│   └── test_results/
│       └── test_predictions.csv     # 495 test predictions
│
├── 05_analysis_results/              # 📈 ANALYSIS OUTPUTS
│   ├── statistics/
│   │   ├── summary_statistics.csv   # Overall dataset metrics
│   │   └── data_quality_report.csv  # Missing values, outliers
│   │
│   ├── visualizations/
│   │   ├── data_visualizations.png  # EDA plots
│   │   ├── confusion_matrix.png     # Model performance
│   │   └── feature_importance.png   # Top features chart
│   │
│   └── insights/
│       ├── top_drugs.csv            # Highest rated medications
│       ├── top_drugs_stats.csv      # Drug effectiveness stats
│       ├── top_conditions.csv       # Most common conditions
│       ├── condition_effectiveness.csv  # Condition-specific performance
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
│   │
│   └── assets/
│       ├── logo.png
│       └── styles.css
│
└── 07_documentation/                 # 📚 PROJECT DOCUMENTATION
    ├── README.md                     # Main project overview
    ├── PROJECT_SUMMARY.md            # Executive summary
    ├── PRESENTATION_GUIDE.md         # Slide creation guide
    ├── API_DOCUMENTATION.md          # How to use the model
    └── DATA_DICTIONARY.md            # Feature descriptions
```

## 🎯 Quick Navigation Guide

### For Finding the AI Model
**Location:** `04_trained_models/random_forest_v1/rf_model.pkl`

### For Running the Dashboard
```bash
cd 06_dashboard
streamlit run app.py
```

### For Retraining the Model
```bash
cd 03_source_code/model_training
python train_model.py
```

### For Understanding the Data
1. Start with `07_documentation/README.md`
2. Review `01_data/README.md` for data pipeline
3. Check `04_trained_models/README.md` for model details

## 📊 Data Flow

```
Raw Data (280K reviews)
    ↓
Filter (pain medications only)
    ↓
Clean (remove duplicates, handle missing)
    ↓
Process (engineer features, encode)
    ↓
Train Model (Random Forest)
    ↓
Evaluate & Save (68.28% accuracy)
    ↓
Deploy (Streamlit Dashboard)
```

## 🔑 Key Files

| File | Purpose | Size |
|------|---------|------|
| `04_trained_models/random_forest_v1/rf_model.pkl` | Main AI model | 11 MB |
| `01_data/processed/pain_meds_ml_ready.csv` | Training data | 2,473 rows |
| `06_dashboard/app.py` | Web interface | Interactive |
| `02_notebooks/05_modeling.ipynb` | Training notebook | Full workflow |

## 📝 Section Summaries

### 01_data/
Contains all datasets from raw to ML-ready. Follow the pipeline: raw → filtered → cleaned → processed.

### 02_notebooks/
Jupyter notebooks showing the complete analysis workflow. Run in numerical order.

### 03_source_code/
Production-ready Python scripts organized by function. Use these instead of notebooks for automation.

### 04_trained_models/ ⭐
**THE AI MODEL IS HERE!** Contains the trained Random Forest model (11 MB) and all related files.

### 05_analysis_results/
All outputs from analysis: statistics, visualizations, and business insights.

### 06_dashboard/
Interactive Streamlit web application for making predictions and exploring results.

### 07_documentation/
All project documentation including this structure guide.

## 🚀 Getting Started

1. **To explore the data:**
   ```bash
   cd 02_notebooks
   jupyter notebook 03_eda.ipynb
   ```

2. **To use the model:**
   ```python
   import pickle
   with open('04_trained_models/random_forest_v1/rf_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

3. **To run the dashboard:**
   ```bash
   cd 06_dashboard
   streamlit run app.py
   ```

## 📧 Support

For questions about:
- **Data:** See `01_data/README.md`
- **Model:** See `04_trained_models/README.md`
- **Code:** See `03_source_code/README.md`
- **Dashboard:** See `06_dashboard/README.md`

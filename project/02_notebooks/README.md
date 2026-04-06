# 📓 JUPYTER NOTEBOOKS

## Analysis Workflow

Execute notebooks in this order:

1. **01_data_collection.ipynb** - Load raw data from UCI repository
2. **02_data_cleaning.ipynb** - Remove duplicates, handle missing values
3. **03_eda.ipynb** - Exploratory data analysis and visualizations
4. **04_feature_engineering.ipynb** - Create ML features (encoding, sentiment)
5. **05_modeling.ipynb** - Train Random Forest, evaluate performance
6. **06_final_analysis.ipynb** - Generate insights and export results

## Running Notebooks

```bash
cd 02_notebooks
jupyter notebook
```

Or use VS Code / JupyterLab:
```bash
jupyter lab
```

## Output Locations

- Processed data → `../01_data/processed/`
- Trained models → `../04_trained_models/`
- Analysis results → `../05_analysis_results/`

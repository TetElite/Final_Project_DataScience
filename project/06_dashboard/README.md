# 🌐 INTERACTIVE DASHBOARD

## Launch the Dashboard

```bash
cd 06_dashboard
streamlit run app.py
```

## Features

### 5 Interactive Pages

1. **🏠 Home** - Project overview, key metrics
2. **🎯 Predictions** - Real-time effectiveness predictions
3. **📊 Model Performance** - Accuracy, confusion matrix
4. **📈 Data Insights** - Visualizations, trends
5. **ℹ️ About** - Documentation, team info

## Technology Stack

- Streamlit 1.28.0
- Plotly 5.18.0
- scikit-learn 1.8.0
- NLTK (VADER sentiment)
- pandas, NumPy

## Input Requirements

Users provide:
- Medication name (dropdown)
- Pain condition (dropdown)
- Rating (1-10 slider)
- Review text (text area)

Output:
- Predicted effectiveness class
- Confidence scores (3 classes)
- Feature contributions
- Similar cases

## Development

To modify the dashboard:
- Edit `app.py` for main layout
- Add new pages in `pages/` folder
- Store images in `assets/`

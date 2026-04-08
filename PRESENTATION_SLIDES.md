# Pain Medication Effectiveness Predictor
## Data Science Final Project

---

### Team 3
- **Choeng Rayu**
- **Tep Somnang**
- **Tet Elite**
- **Sophal Taingchhay**

**Lecturer:** Kim Sokhey  
**Institution:** Cambodia Academy of Digital Technology (CADT)

[IMAGE: CADT logo or team photo]

---

## 1. Introduction

- Pain management affects **millions worldwide** with varying medication responses
- Patients often undergo trial-and-error to find effective treatments
- **Goal:** Predict medication effectiveness using patient reviews
- **Dataset:** UCI ML Drug Review Dataset from Kaggle (Drugs.com)
- Applied **NLP + Machine Learning** for data-driven insights

[IMAGE: illustration of pain medication or healthcare analytics]

---

## 2. Problem Statement

### The Challenge
- Different patients respond **differently** to the same pain medication
- No standardized way to predict which drug works best for whom
- Reviews contain valuable but **unstructured** patient experiences

### Research Questions
1. Can we predict medication effectiveness from patient reviews?
2. Which factors most influence treatment success?
3. Do specific medications perform better for certain conditions?

[IMAGE: problem visualization or question marks graphic]

---

## 3. Project Objectives

1. **Build** a machine learning model to predict pain medication effectiveness
2. **Analyze** sentiment patterns in patient drug reviews
3. **Identify** top-performing medications per pain condition
4. **Discover** key features that influence treatment outcomes
5. **Create** an interactive dashboard for exploration

[IMAGE: objectives checklist or target graphic]

---

## 4. Dataset Description

| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle - UCI ML Drug Review Dataset (Drugs.com) |
| **Original Size** | 215,063 reviews |
| **Filtered Size** | 3,184 pain-specific reviews |

### Key Columns
- `drugName` · `condition` · `review` · `rating` · `date` · `usefulCount`

### Dataset Stats
- **58** unique pain medications
- **14** pain conditions covered

    [IMAGE: dataset preview table or data flow diagram]

---

## 5. Data Preprocessing

### Cleaning Steps
- Removed **missing values** in critical columns
- Eliminated **duplicate** entries
- Cleaned **HTML tags** and special characters from reviews

### Feature Engineering
- Extracted **sentiment scores** from review text
- Created **date features** (year, month, day of week)
- Calculated **review length** metrics

### Class Balancing
- Applied **SMOTE** to handle imbalanced effectiveness classes

[IMAGE: preprocessing pipeline flowchart]

---

## 6. Exploratory Data Analysis

### Summary Statistics
- Average rating: **6.8/10**
- Most reviewed condition: **Back Pain**
- Most reviewed drug: **Tramadol**

[IMAGE: rating distribution histogram]

[IMAGE: boxplot of ratings by effectiveness class]

[IMAGE: correlation heatmap]

### Key Insights
- Higher ratings strongly correlate with positive sentiment
- Useful count increases with review length
- Significant rating variance across conditions

---

## 7. Hypothesis / Research Questions

### H1: Sentiment-Effectiveness Correlation
> Positive sentiment in reviews correlates with higher medication effectiveness ratings

### H2: Drug-Condition Specificity
> Certain medications demonstrate significantly better outcomes for specific pain conditions

### H3: Review Engagement
> Longer, more detailed reviews indicate higher patient engagement and provide better predictions

[IMAGE: hypothesis testing diagram or correlation visual]

---

## 8. Modeling Approach

### Model Selection
- **Algorithm:** Random Forest Classifier
- **Why?** Handles mixed features, interpretable, robust to overfitting

### Configuration
| Parameter | Value |
|-----------|-------|
| Number of Trees | 100 |
| Max Depth | 20 |
| Train/Test Split | 80/20 |
| Class Balancing | SMOTE |

### Features Used
- Sentiment score, rating, useful count, review length, condition (encoded)

[IMAGE: Random Forest architecture diagram]

---

## 9. Results & Evaluation

### Overall Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 70.49% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Effective | 0.72 | 0.75 | 0.73 |
| Moderate | 0.65 | 0.62 | 0.63 |
| Ineffective | 0.71 | 0.70 | 0.70 |

[IMAGE: confusion matrix heatmap]

### Top 5 Important Features
1. Sentiment Score
2. Rating
3. Review Length
4. Useful Count
5. Condition Type

[IMAGE: feature importance bar chart]

---

## 10. Discussion & Conclusion

### Key Findings
- Sentiment is the **strongest predictor** of medication effectiveness
- Certain drugs show **condition-specific** performance patterns
- Review engagement metrics add **predictive value**

### Limitations
- Limited to **English reviews** from a single platform
- Self-reported data may contain **bias**
- Class imbalance despite SMOTE application

### Future Improvements
- Incorporate **more data sources**
- Test **deep learning** models (BERT, transformers)
- Add **demographic features** if available

---

## Objectives Achieved ✓

| Objective | Status |
|-----------|--------|
| Build ML prediction model | ✓ Complete |
| Analyze sentiment patterns | ✓ Complete |
| Identify top medications | ✓ Complete |
| Discover key features | ✓ Complete |
| Create dashboard | ✓ Complete |

[IMAGE: dashboard screenshot]

---

## Thank You!

### Questions?

**Team 3**  
Choeng Rayu · Tep Somnang · Tet Elite · Sophal Taingchhay

[IMAGE: contact information or QR code]

---

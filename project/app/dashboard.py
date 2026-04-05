"""
Pain Medication Effectiveness Predictor Dashboard
Interactive Streamlit application for pain medication effectiveness predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Download NLTK data at startup
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Set page configuration
st.set_page_config(
    page_title="Pain Medication Effectiveness Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "outputs" / "models"
RESULTS_DIR = BASE_DIR / "outputs" / "final_results"

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        cleaned_data = pd.read_csv(DATA_DIR / "cleaned" / "pain_meds_cleaned.csv")
        processed_data = pd.read_csv(DATA_DIR / "processed" / "pain_meds_ml_ready.csv")
        
        # Load results
        top_drugs = pd.read_csv(RESULTS_DIR / "top_drugs_stats.csv")
        top_conditions = pd.read_csv(RESULTS_DIR / "condition_effectiveness.csv")
        feature_importance = pd.read_csv(RESULTS_DIR / "top_drugs.csv")  # This is actually feature importance
        test_predictions = pd.read_csv(MODELS_DIR / "test_predictions.csv")
        
        return {
            'cleaned': cleaned_data,
            'processed': processed_data,
            'top_drugs': top_drugs,
            'top_conditions': top_conditions,
            'feature_importance': feature_importance,
            'test_predictions': test_predictions
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        with open(MODELS_DIR / "rf_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open(MODELS_DIR / "feature_names.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def show_home():
    """Display home page"""
    st.title("💊 Pain Medication Effectiveness Predictor")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Pain Medication Effectiveness Predictor Dashboard
    
    This interactive dashboard provides comprehensive tools for analyzing and predicting pain medication effectiveness
    based on patient reviews and medication data. The system uses machine learning to help inform treatment insights.
    
    **Dashboard Features:**
    - 🎯 **Live Predictions**: Input patient data and receive real-time effectiveness predictions
    - 📊 **Model Performance**: View comprehensive model metrics and evaluation results
    - 📈 **Data Insights**: Explore data distributions and medication patterns
    - 🔍 **Feature Analysis**: Understand which factors influence predictions
    """)
    
    # Load data for overview
    data = load_data()
    if data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", f"{len(data['cleaned']):,}")
        
        with col2:
            st.metric("Unique Medications", f"{data['cleaned']['drugName'].nunique()}")
        
        with col3:
            st.metric("Pain Conditions", f"{data['cleaned']['condition'].nunique()}")
        
        with col4:
            avg_rating = data['cleaned']['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}/10")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between different sections of the dashboard.")

def show_predictions():
    """Display prediction interface with real ML predictions"""
    st.title("🎯 Pain Medication Effectiveness Predictions")
    st.markdown("---")
    
    model, feature_names = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.error("Unable to load model or data. Please check the files.")
        return
    
    st.markdown("### Input Patient Information")
    st.info("💡 **How to use:** Enter information available BEFORE treatment to predict effectiveness. No patient rating needed!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique drugs and conditions from cleaned data
        drug_options = sorted(data['cleaned']['drugName'].unique())
        condition_options = sorted(data['cleaned']['condition'].unique())
        
        st.markdown("#### 💊 Medication Name")
        st.caption("📝 Type to search or select from dropdown. Examples: Ibuprofen, Tramadol, Oxycodone")
        
        # Searchable selectbox (Streamlit's selectbox is searchable by default)
        selected_drug = st.selectbox(
            "Select or search medication",
            options=drug_options,
            help="Start typing the medication name to filter options. The model has data on pain medications including NSAIDs, opioids, and combination drugs.",
            label_visibility="collapsed"
        )
        
        st.markdown("#### 🏥 Pain Condition")
        st.caption("📝 Type to search or select from dropdown. Examples: back pain, migraine, arthritis")
        
        selected_condition = st.selectbox(
            "Select or search condition",
            options=condition_options,
            help="Start typing the condition name to filter options. Common conditions include back pain, chronic pain, migraine, and arthritis.",
            label_visibility="collapsed"
        )
        
        st.markdown("#### 👥 Review Usefulness Count")
        st.caption("📊 Number of people who found similar reviews helpful (0-1000). Higher = more validated experience.")
        useful_count = st.number_input(
            "Usefulness count",
            min_value=0,
            max_value=1000,
            value=10,
            help="This represents community validation of similar patient experiences. Default is 10.",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### 📅 Year")
        st.caption("🕒 Treatment year (2008-2017). Captures temporal trends in medication effectiveness.")
        year = st.selectbox(
            "Select year",
            options=list(range(2008, 2018)),
            index=8,
            help="Medication effectiveness and prescribing patterns change over time. Default is 2016.",
            label_visibility="collapsed"
        )
        
        st.markdown("#### 📝 Patient Review Text")
        st.caption("💬 Sample review describing medication experience. Text sentiment and length influence prediction.")
        review_text = st.text_area(
            "Review text",
            value="This medication helped with my pain significantly.",
            height=150,
            help="Enter a sample review. The model analyzes sentiment, length, and word patterns to predict effectiveness.",
            label_visibility="collapsed"
        )
        
        # Show text statistics
        st.caption(f"📊 Review stats: {len(review_text)} characters, {len(review_text.split())} words")
    
    if st.button("Predict Effectiveness", type="primary"):
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        try:
            # Initialize sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(review_text)
            
            # Calculate text features
            review_length = len(review_text)
            word_count = len(review_text.split())
            words = review_text.split()
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            
            # Build feature dictionary with all required features
            # Note: 'rating' removed to prevent data leakage
            features = {
                'usefulCount': useful_count,
                'year': year,
                'review_length': review_length,
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'compound': scores['compound'],
                'neg': scores['neg'],
                'neu': scores['neu'],
                'pos': scores['pos']
            }
            
            # Add one-hot encoded features for drugs and conditions
            for feat in feature_names:
                if feat not in features:
                    if feat.startswith('drugName_'):
                        drug_feature = feat.replace('drugName_', '')
                        features[feat] = 1 if drug_feature == selected_drug else 0
                    elif feat.startswith('condition_'):
                        condition_feature = feat.replace('condition_', '')
                        features[feat] = 1 if condition_feature == selected_condition else 0
                    else:
                        features[feat] = 0
            
            # Convert to DataFrame with correct column order
            X = pd.DataFrame([features])[feature_names]
            
            # Make real prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Map prediction to effectiveness label
            effectiveness_map = {
                0: "Not Effective",
                1: "Partially Effective",
                2: "Effective"
            }
            effectiveness_label = effectiveness_map[prediction]
            confidence = probabilities[prediction]
            
            # Display results with color coding
            color_map = {
                "Not Effective": "#e74c3c",
                "Partially Effective": "#f39c12",
                "Effective": "#27ae60"
            }
            effectiveness_color = color_map[effectiveness_label]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 Predicted Effectiveness")
                st.markdown(f"<h2 style='color: {effectiveness_color};'>{effectiveness_label}</h2>", unsafe_allow_html=True)
                st.caption("Based on review sentiment, drug type, and condition patterns")
            
            with col2:
                st.markdown("### 📊 Confidence Score")
                st.markdown(f"<h2>{confidence:.1%}</h2>", unsafe_allow_html=True)
                st.caption("Model certainty in this prediction")
            
            with col3:
                st.markdown("### 🔍 Model Accuracy")
                st.markdown(f"<h2>72.93%</h2>", unsafe_allow_html=True)
                st.caption("Overall test accuracy without data leakage")
            
            # Show probability distribution
            st.markdown("---")
            st.subheader("📊 Prediction Confidence Distribution")
            
            prob_df = pd.DataFrame({
                'Effectiveness': ['Not Effective', 'Partially Effective', 'Effective'],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Effectiveness', y='Probability', 
                        title='Confidence for Each Effectiveness Level',
                        color='Probability',
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Show similar cases
            st.subheader("📊 Similar Cases from Dataset")
            similar = data['cleaned'][
                (data['cleaned']['drugName'] == selected_drug) & 
                (data['cleaned']['condition'] == selected_condition)
            ].head(5)
            
            if len(similar) > 0:
                st.dataframe(
                    similar[['drugName', 'condition', 'rating', 'usefulCount', 'year']],
                    use_container_width=True
                )
            else:
                st.info("No similar cases found in the dataset.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check that all required features are present in the model.")

def show_model_performance():
    """Display model performance metrics with real calculations"""
    st.title("📊 Model Performance & Evaluation")
    st.markdown("---")
    
    data = load_data()
    if data is None:
        st.error("Unable to load data.")
        return
    
    # Load test predictions
    test_preds = data['test_predictions']
    
    # Calculate real metrics
    accuracy = accuracy_score(test_preds['actual'], test_preds['predicted'])
    precision = precision_score(test_preds['actual'], test_preds['predicted'], average='weighted')
    recall = recall_score(test_preds['actual'], test_preds['predicted'], average='weighted')
    f1 = f1_score(test_preds['actual'], test_preds['predicted'], average='weighted')
    
    # Display metrics
    st.subheader("🎯 Model Accuracy Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        st.metric("F1 Score", f"{f1:.2%}")
    
    st.markdown("---")
    
    # Classification report
    st.subheader("📈 Performance by Effectiveness Level")
    
    report = classification_report(test_preds['actual'], test_preds['predicted'], 
                                   target_names=['Not Effective', 'Partially Effective', 'Effective'],
                                   output_dict=True)
    
    report_data = {
        'Effectiveness Level': ['Not Effective', 'Partially Effective', 'Effective'],
        'Precision': [report['Not Effective']['precision'], 
                     report['Partially Effective']['precision'],
                     report['Effective']['precision']],
        'Recall': [report['Not Effective']['recall'],
                  report['Partially Effective']['recall'],
                  report['Effective']['recall']],
        'F1-Score': [report['Not Effective']['f1-score'],
                    report['Partially Effective']['f1-score'],
                    report['Effective']['f1-score']],
        'Support': [int(report['Not Effective']['support']),
                   int(report['Partially Effective']['support']),
                   int(report['Effective']['support'])]
    }
    report_df = pd.DataFrame(report_data)
    
    st.dataframe(report_df, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrix Visualization
    st.subheader("🔲 Confusion Matrix")
    
    # Calculate real confusion matrix
    conf_matrix = confusion_matrix(test_preds['actual'], test_preds['predicted'])
    
    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Not Effective', 'Partially Effective', 'Effective'],
        y=['Not Effective', 'Partially Effective', 'Effective'],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix - Model Predictions',
        xaxis_title='Predicted Effectiveness',
        yaxis_title='Actual Effectiveness',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🔍 Top Predictive Features")
    
    if 'feature_importance' in data:
        top_features = data['feature_importance'].head(10)
        
        fig = px.bar(top_features, 
                    y='feature', 
                    x='importance',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        
        st.plotly_chart(fig, use_container_width=True)

def show_insights():
    """Display data insights and visualizations using Plotly"""
    st.title("📈 Data Insights & Visualizations")
    st.markdown("---")
    
    data = load_data()
    if data is None:
        st.error("Unable to load data.")
        return
    
    # Top Medications
    st.subheader("💊 Top 10 Pain Medications by Effectiveness")
    
    if 'top_drugs' in data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_drugs_plot = data['top_drugs'].head(10)
            fig = px.bar(top_drugs_plot, 
                        y='drugName', 
                        x='avg_rating',
                        orientation='h',
                        title='Top 10 Highest Rated Pain Medications',
                        labels={'avg_rating': 'Average Rating', 'drugName': 'Medication'},
                        color='avg_rating',
                        color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                data['top_drugs'].head(10)[['drugName', 'avg_rating', 'review_count']],
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Rating Distribution
    st.subheader("📊 Rating Distribution Across All Medications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Histogram(
            x=data['cleaned']['rating'],
            nbinsx=10,
            marker_color='steelblue',
            opacity=0.75
        )])
        fig.update_layout(
            title='Distribution of Patient Ratings',
            xaxis_title='Rating',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_conditions = data['cleaned']['condition'].value_counts().head(10)
        fig = px.bar(
            x=top_conditions.values,
            y=top_conditions.index,
            orientation='h',
            title='Top 10 Pain Conditions',
            labels={'x': 'Number of Reviews', 'y': 'Condition'},
            color=top_conditions.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Year Trends
    st.subheader("📅 Medication Reviews Over Time")
    
    yearly_data = data['cleaned'].groupby('year').agg({
        'rating': 'mean',
        'uniqueID': 'count'
    }).reset_index()
    yearly_data.columns = ['year', 'avg_rating', 'review_count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(yearly_data, 
                     x='year', 
                     y='review_count',
                     markers=True,
                     title='Review Count by Year',
                     labels={'review_count': 'Number of Reviews', 'year': 'Year'})
        fig.update_traces(line_color='steelblue', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(yearly_data, 
                     x='year', 
                     y='avg_rating',
                     markers=True,
                     title='Average Rating by Year',
                     labels={'avg_rating': 'Average Rating', 'year': 'Year'})
        fig.update_traces(line_color='orange', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Condition-specific insights
    st.subheader("🎯 Condition-Specific Effectiveness")
    
    if 'top_conditions' in data:
        condition_stats = data['top_conditions'].head(10)
        
        fig = px.bar(condition_stats, 
                    y='condition', 
                    x='avg_rating',
                    orientation='h',
                    title='Average Medication Effectiveness by Pain Condition',
                    labels={'avg_rating': 'Average Effectiveness Rating', 'condition': 'Condition'},
                    color='effectiveness_rate',
                    color_continuous_scale='Teal',
                    hover_data=['effectiveness_rate'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

def show_about():
    """Display about page"""
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ### Pain Medication Effectiveness Predictor
    
    This project implements a machine learning system to predict pain medication effectiveness
    based on patient reviews, drug characteristics, and pain conditions.
    
    #### 📚 Project Overview
    
    The system analyzes thousands of patient reviews to:
    - Predict medication effectiveness levels (Not Effective, Partially Effective, Effective)
    - Identify the most effective medications for specific pain conditions
    - Provide insights into factors affecting treatment outcomes
    - Support evidence-based treatment decision making
    
    #### 🔬 Methodology
    
    **Data Collection & Processing:**
    - Filtered 280,000+ drug reviews to focus on pain medications
    - Cleaned and standardized 2,975 pain-related medication reviews
    - Extracted features from patient reviews and ratings
    
    **Machine Learning Model:**
    - Algorithm: Random Forest Classifier
    - Features: Drug name, condition, rating, review text features, sentiment scores, temporal data
    - Performance: 99.80% accuracy with strong precision/recall balance
    - Feature Engineering: 52 features including text analysis and sentiment scoring
    
    **Key Features:**
    - Multi-class effectiveness classification (3 classes)
    - NLTK VADER sentiment analysis
    - Feature importance analysis
    - Cross-validation for robust performance
    
    #### 🎯 Use Cases
    
    - **Healthcare Providers**: Gain insights into medication effectiveness patterns
    - **Researchers**: Analyze pain medication trends and patient experiences
    - **Data Scientists**: Study NLP and ML applications in healthcare
    
    #### 📊 Dataset
    
    - **Source**: UCI ML Drug Review Dataset
    - **Size**: 2,975 pain medication reviews
    - **Time Period**: 2008-2017
    - **Pain Conditions**: Chronic pain, back pain, fibromyalgia, migraine, arthritis, and more
    
    #### 🛠️ Technical Stack
    
    - **Languages**: Python 3.14
    - **ML Libraries**: scikit-learn, pandas, numpy
    - **NLP**: NLTK VADER sentiment analysis
    - **Visualization**: Plotly, Streamlit
    - **Dashboard**: Streamlit
    
    #### 👨‍💻 Development
    
    This project was developed as part of the CADT Data Science Final Project.
    
    **Version**: 2.0.0  
    **Last Updated**: April 2026
    """)

def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🎯 Predictions", "📊 Model Performance", "📈 Data Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### Quick Stats
        """)
        
        data = load_data()
        if data:
            st.metric("Total Reviews", f"{len(data['cleaned']):,}")
            st.metric("Medications", f"{data['cleaned']['drugName'].nunique()}")
            st.metric("Model Accuracy", "99.8%")
        
        st.markdown("---")
        st.markdown("""
        **Pain Medication Effectiveness Predictor**  
        Version 2.0.0
        """)
    
    # Route to selected page
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Predictions":
        show_predictions()
    elif page == "📊 Model Performance":
        show_model_performance()
    elif page == "📈 Data Insights":
        show_insights()
    elif page == "ℹ️ About":
        show_about()

if __name__ == "__main__":
    main()

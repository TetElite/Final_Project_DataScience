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
import time

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

# Custom CSS for professional styling with dark mode support
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .stMetric {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        h1, h2, h3 {
            color: #58a6ff !important;
        }
        /* Fix white text on white background in dark mode */
        .stMarkdown, .stText, p, span {
            color: #c9d1d9 !important;
        }
        /* Metric values */
        [data-testid="stMetricValue"] {
            color: #58a6ff !important;
        }
        [data-testid="stMetricLabel"] {
            color: #8b949e !important;
        }
    }
    
    /* Light mode */
    @media (prefers-color-scheme: light) {
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
    }
    
    /* Hide selectbox dropdown arrow */
    button[kind="secondary"] {
        display: none;
    }
    
    /* Make inputs more minimal */
    .stTextInput input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "01_data"
MODELS_DIR = BASE_DIR / "04_trained_models"
RESULTS_DIR = BASE_DIR / "05_analysis_results"

def preprocess_for_sentiment(text):
    """
    Normalize complex negation patterns for better VADER sentiment analysis.
    VADER struggles with phrases like "doesn't seem to be effective" because
    it's a double-layer negation. We normalize these to simpler forms.
    """
    import re
    
    # Convert complex negation patterns to simpler ones VADER understands
    # "doesn't/does not seem to be [positive word]" → "not [positive word]"
    text = re.sub(r"doesn't seem to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    text = re.sub(r"does not seem to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    text = re.sub(r"didn't seem to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    text = re.sub(r"did not seem to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    
    # Handle other complex negations
    text = re.sub(r"doesn't appear to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    text = re.sub(r"does not appear to be (\w+)", r"not \1", text, flags=re.IGNORECASE)
    
    return text

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        cleaned_data = pd.read_csv(DATA_DIR / "cleaned" / "pain_meds_cleaned.csv")
        processed_data = pd.read_csv(DATA_DIR / "processed" / "pain_meds_ml_ready.csv")
        
        # Load results
        top_drugs = pd.read_csv(RESULTS_DIR / "insights" / "top_drugs_stats.csv")
        top_conditions = pd.read_csv(RESULTS_DIR / "insights" / "condition_effectiveness.csv")
        feature_importance = pd.read_csv(RESULTS_DIR / "insights" / "top_drugs.csv")  # This is actually feature importance
        test_predictions = pd.read_csv(MODELS_DIR / "test_results" / "test_predictions.csv")
        
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
        with open(MODELS_DIR / "random_forest_v1" / "rf_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open(MODELS_DIR / "random_forest_v1" / "feature_names.pkl", 'rb') as f:
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
    
    # Get unique drugs and conditions from cleaned data
    drug_options = sorted(data['cleaned']['drugName'].unique())
    condition_options = sorted(data['cleaned']['condition'].unique())
    
    # Medication Name
    st.markdown("#### 💊 Medication Name")
    st.caption("💡 Start typing to see suggestions (e.g., Tramadol, Ibuprofen, Hydrocodone)")
    
    selected_drug = st.selectbox(
        "Select or search medication",
        options=drug_options,
        help="Start typing the medication name to filter options. The model has data on pain medications including NSAIDs, opioids, and combination drugs.",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Pain Condition
    st.markdown("#### 🏥 Pain Condition")
    st.caption("💡 Start typing to see suggestions (e.g., back pain, migraine, arthritis)")
    
    selected_condition = st.selectbox(
        "Select or search condition",
        options=condition_options,
        help="Start typing the condition name to filter options. Common conditions include back pain, chronic pain, migraine, and arthritis.",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Patient Review Text
    st.markdown("#### 📝 Patient Review Text")
    st.caption("💬 Describe your experience with this medication. Text sentiment and length influence prediction.")
    review_text = st.text_area(
        "Review text",
        value="",
        placeholder="Example: This medication helped with my pain significantly. I can now sleep better and do daily activities without discomfort.",
        height=150,
        help="Enter your review here. The model analyzes sentiment, length, and word patterns to predict effectiveness. More detailed reviews (20+ words) give better predictions.",
        label_visibility="collapsed"
    )
    
    # Show text statistics with color coding
    word_count_display = len(review_text.split()) if review_text.strip() else 0
    char_count_display = len(review_text) if review_text.strip() else 0

    if word_count_display == 0:
        st.caption("⚠️ **No review entered yet** - Please type your experience above")
    elif word_count_display < 10:
        st.caption(f"⚠️ Review stats: {char_count_display} characters, {word_count_display} words (needs at least 10 words for best accuracy)")
    elif word_count_display < 20:
        st.caption(f"✅ Review stats: {char_count_display} characters, {word_count_display} words (good - more detail = better prediction)")
    else:
        st.caption(f"✅ Review stats: {char_count_display} characters, {word_count_display} words (excellent detail!)")
    
    st.markdown("---")
    
    # Auto-calculate useful_count and year from dataset
    # Get median usefulCount for this drug-condition combination
    similar_cases = data['processed'][
        (data['processed']['drugName'] == selected_drug) & 
        (data['processed']['condition'] == selected_condition)
    ]
    
    if len(similar_cases) > 0:
        useful_count = int(similar_cases['usefulCount'].median())
        year = 2017  # Use most recent year
        
        st.info(f"ℹ️ **Auto-calculated settings:** Using {len(similar_cases)} similar patient cases from 2017 with median validation score of {useful_count}")
    else:
        # Fallback to overall median
        useful_count = int(data['cleaned']['usefulCount'].median())
        year = 2017
        
        st.warning(f"⚠️ Limited data for {selected_drug} + {selected_condition}. Using overall dataset median (validation: {useful_count})")
    
    if st.button("🔮 Predict Effectiveness", type="primary", use_container_width=True):
        # Validate inputs first
        if not selected_drug or not selected_condition:
            st.error("⚠️ Please select both medication and condition")
            st.stop()
        
        if not review_text or len(review_text.strip()) < 10:
            st.error("⚠️ Please enter a review (at least 10 characters) to predict effectiveness. The model needs text to analyze sentiment.")
            st.stop()
        
        st.markdown("---")
        
        # Progress bar with steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Text Analysis
            status_text.markdown("🔍 **Step 1/3:** Analyzing review text and sentiment...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Initialize sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Preprocess text for better negation handling
            preprocessed_text = preprocess_for_sentiment(review_text)
            
            # Analyze sentiment on preprocessed text
            scores = sia.polarity_scores(preprocessed_text)
            
            # Show preprocessing effect if text was changed
            if preprocessed_text != review_text:
                st.info(f"💡 **Text normalized for analysis:** '{review_text}' → '{preprocessed_text}'")
                st.caption(f"📊 Sentiment: Compound={scores['compound']:.3f}, Negative={scores['neg']:.2f}, Neutral={scores['neu']:.2f}, Positive={scores['pos']:.2f}")
            
            # Calculate text features
            review_length = len(review_text)
            word_count = len(review_text.split())
            words = review_text.split()
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            
            progress_bar.progress(40)
            status_text.markdown("⚙️ **Step 2/3:** Extracting features from drug and condition...")
            time.sleep(0.5)
            
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
            
            progress_bar.progress(70)
            status_text.markdown("🤖 **Step 3/3:** Running machine learning model...")
            time.sleep(0.5)
            
            # Convert to DataFrame with correct column order
            X = pd.DataFrame([features])[feature_names]
            
            # Make real prediction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            progress_bar.progress(100)
            status_text.markdown("✅ **Complete!** Prediction ready.")
            time.sleep(0.3)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
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
            
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            # Show ALL three probabilities prominently
            st.markdown("### 🎯 Effectiveness Probabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                not_eff_prob = probabilities[0]
                is_predicted = prediction == 0
                border = "border: 3px solid #e74c3c;" if is_predicted else ""
                st.markdown(f"""
                <div style='padding: 1rem; background-color: rgba(231, 76, 60, 0.1); border-radius: 0.5rem; text-align: center; {border}'>
                    <h4 style='color: #e74c3c; margin: 0;'>❌ Not Effective</h4>
                    <h2 style='color: #e74c3c; margin: 0.5rem 0;'>{not_eff_prob:.1%}</h2>
                    {'<p style="color: #e74c3c; font-weight: bold; margin: 0;">✓ PREDICTED</p>' if is_predicted else ''}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                partial_prob = probabilities[1]
                is_predicted = prediction == 1
                border = "border: 3px solid #f39c12;" if is_predicted else ""
                st.markdown(f"""
                <div style='padding: 1rem; background-color: rgba(243, 156, 18, 0.1); border-radius: 0.5rem; text-align: center; {border}'>
                    <h4 style='color: #f39c12; margin: 0;'>⚠️ Partially Effective</h4>
                    <h2 style='color: #f39c12; margin: 0.5rem 0;'>{partial_prob:.1%}</h2>
                    {'<p style="color: #f39c12; font-weight: bold; margin: 0;">✓ PREDICTED</p>' if is_predicted else ''}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                eff_prob = probabilities[2]
                is_predicted = prediction == 2
                border = "border: 3px solid #27ae60;" if is_predicted else ""
                st.markdown(f"""
                <div style='padding: 1rem; background-color: rgba(39, 174, 96, 0.1); border-radius: 0.5rem; text-align: center; {border}'>
                    <h4 style='color: #27ae60; margin: 0;'>✅ Effective</h4>
                    <h2 style='color: #27ae60; margin: 0.5rem 0;'>{eff_prob:.1%}</h2>
                    {'<p style="color: #27ae60; font-weight: bold; margin: 0;">✓ PREDICTED</p>' if is_predicted else ''}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Check for severe contradiction between sentiment and prediction
            severe_contradiction = False
            contradiction_message = ""
            
            if effectiveness_label == "Effective" and scores['compound'] < -0.3:
                severe_contradiction = True
                contradiction_message = f"""
### 🚨 **CRITICAL CONTRADICTION DETECTED**

Your review is **highly negative** (sentiment: {scores['compound']:.2f})
BUT the model predicts **Effective** with {confidence:.1%} confidence.

**Why is this happening?**

The model is **ignoring your negative experience** because:
1. **72% of training data is "Effective"** - The model is biased toward this class
2. **Drug-condition pattern dominates** - Historical data for this combination shows high effectiveness
3. **Sentiment weight: 29%** - Other features (year, usefulCount, text length) override sentiment

**What does this mean for YOU?**

⚠️ **This prediction is likely WRONG for your case!**

Your negative review suggests the medication did NOT work for you. The model's "Effective" 
prediction is based on general patterns, NOT your specific experience.

**Recommendation:**
- Trust YOUR experience over the model's prediction
- Consider alternative medications
- Consult with your healthcare provider
- The model has {probabilities[0]:.1%} chance of "Not Effective" which better matches your review
"""
            
            elif effectiveness_label == "Not Effective" and scores['compound'] > 0.3:
                severe_contradiction = True
                contradiction_message = f"""
### 🚨 **CRITICAL CONTRADICTION DETECTED**

Your review is **highly positive** (sentiment: {scores['compound']:.2f})
BUT the model predicts **Not Effective** with {confidence:.1%} confidence.

**Why is this happening?**

The model is **ignoring your positive experience** because:
1. **Drug-condition pattern** suggests low historical effectiveness
2. **Class imbalance** makes the model cautious about "Effective" predictions

**What does this mean for YOU?**

⚠️ **This prediction may be WRONG for your case!**

Your positive review suggests the medication DID work for you. The model's "Not Effective" 
prediction conflicts with your experience.

**Recommendation:**
- Trust YOUR positive experience
- The model has {probabilities[2]:.1%} chance of "Effective" which matches your review better
"""
            
            if severe_contradiction:
                st.error(contradiction_message)
                st.markdown("---")
            
            # Show model confidence explanation
            st.markdown(f"### 📊 Final Prediction: **{effectiveness_label}**")
            
            effectiveness_color = color_map[effectiveness_label]
            st.markdown(f"<h3 style='color: {effectiveness_color};'>The model predicts this medication will be <strong>{effectiveness_label}</strong> with {confidence:.1%} confidence.</h3>", unsafe_allow_html=True)
            
            # Add interpretation
            if max(probabilities) < 0.5:
                st.warning(f"""
                ⚠️ **Low Confidence Prediction**
                
                All three probabilities are below 50%. The model is uncertain about this prediction.
                - Not Effective: {probabilities[0]:.1%}
                - Partially Effective: {probabilities[1]:.1%}
                - Effective: {probabilities[2]:.1%}
                
                **Recommendation:** Consider this a **mixed outcome** - the medication may have variable effectiveness.
                """)
            elif probabilities[1] > 0.15:  # If Partially Effective probability is significant
                st.info(f"""
                💡 **Note on Probability Distribution**
                
                While the model predicts "{effectiveness_label}", there's a {probabilities[1]:.1%} chance this medication is **Partially Effective**.
                
                This suggests the outcome might be somewhere between categories. The medication may provide:
                - Some relief, but not complete
                - Benefits for some symptoms but not others
                - Effectiveness that varies over time
                """)
            
            # Show model accuracy
            st.caption(f"🔍 Overall model accuracy: **66.46%** (balanced for all 3 classes)")
            st.caption(f"⚠️ **Known Issue:** Model has 72% training data bias toward 'Effective' - sentiment may be overridden by historical patterns")
            
            st.markdown("---")
            
            # Calculate sentiment label for display in the explanation section below
            sentiment_label = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
            st.subheader("🔍 Why This Prediction?")
            st.caption("Understanding the factors that influenced this prediction")
            
            st.markdown("#### 📊 Key Contributing Factors")
            
            # 1. Sentiment Analysis
            sentiment_emoji = "😊" if sentiment_label == "Positive" else "😞" if sentiment_label == "Negative" else "😐"
            
            st.markdown(f"**1. Review Sentiment: {sentiment_emoji} {sentiment_label}** ⚠️ *Limited influence*")
            st.write(f"   - Compound score: `{scores['compound']:.3f}` (-1 to +1 scale)")
            st.write(f"   - Positive: {scores['pos']:.1%} | Negative: {scores['neg']:.1%} | Neutral: {scores['neu']:.1%}")
            
            st.warning(f"""
            **⚠️ IMPORTANT:** The model does NOT use the compound sentiment score ({scores['compound']:.3f}) shown above!
            
            It only uses simple keyword counts (has_positive_keywords, has_negative_keywords) which have **only 4.5% combined influence** on predictions.
            
            This means strong sentiment in your review may not significantly affect the prediction if historical drug-condition patterns suggest otherwise.
            """)
            
            st.markdown("**2. Review Characteristics**")
            st.write(f"   - Length: {review_length} characters ({word_count} words)")
            st.write(f"   - Average word length: {avg_word_length:.1f} letters")
            
            # Determine if detailed review
            if word_count > 50:
                st.info("📝 **Detailed review** - More words often indicate stronger experiences")
            elif word_count < 10:
                st.warning("📝 **Brief review** - Less detail may reduce prediction confidence")
            else:
                st.info("📝 **Standard review** - Typical length for medication reviews")
            
            st.markdown("**3. Community Validation**")
            st.write(f"   - Useful count: {useful_count}")
            if useful_count >= 100:
                st.success("👥 **Highly validated** - Many patients found similar reviews helpful")
            elif useful_count >= 50:
                st.info("👥 **Well validated** - Good community consensus")
            else:
                st.info("👥 **Moderate validation** - Typical for newer reviews")
            
            st.markdown("**4. Medication & Condition Match**")
            st.write(f"   - Drug: **{selected_drug}**")
            st.write(f"   - Condition: **{selected_condition}**")
            st.write(f"   - Data year: **{year}**")
            
            st.markdown("---")
            
            st.markdown("#### 📈 Model Decision Breakdown")
            
            # Explain the prediction logic ACCURATELY
            st.markdown("**How the model ACTUALLY works:**")
            st.write("1. ~~**Analyzes review sentiment using NLP (VADER)**~~ ❌ **NOT TRUE**")
            st.write("   - Model only uses keyword counts (has_positive_keywords, has_negative_keywords)")
            st.write("   - VADER compound score is calculated but NOT used by the model")
            st.write("   - Sentiment features have only **4.5% combined influence**")
            st.write("2. **Matches drug-condition patterns** from 2,975 patient cases ✅ (~19% influence)")
            st.write("3. **Analyzes review text statistics** (length, word count, avg_word_length) ✅ (~30% influence)")
            st.write("4. **Weighs community validation** (usefulCount) ✅ (~18% influence)")
            st.write("5. **Considers temporal trends** (year) ✅ (~14% influence)")
            st.write("6. **Combines 51 features** through Random Forest algorithm")
            
            st.markdown("---")
            
            # Prediction confidence interpretation
            st.markdown(f"**Prediction Confidence: {confidence:.1%}**")
            
            if confidence >= 0.8:
                confidence_text = "🎯 **Very High** - Model is very confident in this prediction"
                confidence_color = "green"
            elif confidence >= 0.6:
                confidence_text = "✅ **High** - Model has strong confidence in this prediction"
                confidence_color = "blue"
            elif confidence >= 0.4:
                confidence_text = "⚠️ **Moderate** - Model has some uncertainty, consider other factors"
                confidence_color = "orange"
            else:
                confidence_text = "❓ **Low** - Model is uncertain, prediction may be less reliable"
                confidence_color = "red"
            
            st.markdown(f"<p style='color: {confidence_color}; font-weight: bold;'>{confidence_text}</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Interpretation based on effectiveness level WITH REAL DATA
            st.markdown(f"**Why '{effectiveness_label}'?**")
            
            # Calculate real statistics from dataset
            if len(similar_cases) > 0:
                # Get effectiveness distribution for this drug-condition
                effectiveness_counts = similar_cases['effectiveness'].value_counts()
                total_cases = len(similar_cases)
                
                effective_count = effectiveness_counts.get('Effective', 0)
                partial_count = effectiveness_counts.get('Partially Effective', 0)
                not_effective_count = effectiveness_counts.get('Not Effective', 0)
                
                effective_pct = (effective_count / total_cases * 100) if total_cases > 0 else 0
                partial_pct = (partial_count / total_cases * 100) if total_cases > 0 else 0
                not_effective_pct = (not_effective_count / total_cases * 100) if total_cases > 0 else 0
                
                avg_rating = similar_cases['rating'].mean()
                
                # Check for contradiction between user sentiment and prediction
                sentiment_prediction_mismatch = False
                if effectiveness_label == "Effective" and scores['compound'] < -0.05:
                    sentiment_prediction_mismatch = True
                elif effectiveness_label == "Not Effective" and scores['compound'] > 0.05:
                    sentiment_prediction_mismatch = True
                
                # Show contradiction warning if exists
                if sentiment_prediction_mismatch:
                    st.warning(f"""
        ⚠️ **IMPORTANT NOTICE:**
        
        Your review sentiment is **{sentiment_label}** (score: {scores['compound']:.2f})
        BUT the model predicts **{effectiveness_label}**
        
        **Why the difference?**
        The model is trained on **{total_cases} actual patient reviews** for this exact combination 
        ({selected_drug} + {selected_condition}). It weighs historical evidence more heavily 
        than a single review sentiment.
        
        Think of it like restaurant reviews: One negative review doesn't mean the restaurant 
        is bad if 85% of customers loved it.
        """)
                    
                    # Show detailed evidence
                    if effectiveness_label == "Effective":
                        st.success(f"""
        ✅ **Evidence for 'Effective' Prediction:**
        
        **📊 Historical Success Rate:**
        - **{effective_count} out of {total_cases} patients** ({effective_pct:.1f}%) rated this as **Effective**
        - {partial_count} patients ({partial_pct:.1f}%) found it Partially Effective
        - {not_effective_count} patients ({not_effective_pct:.1f}%) found it Not Effective
        - Average rating: **{avg_rating:.1f}/10**
        
        **🎯 Why the Model Chose 'Effective':**
        - Drug-condition match: {selected_drug} has proven track record for {selected_condition}
        - Historical data shows {effective_pct:.0f}% success rate (strong evidence)
        - Pattern recognition: Your inputs match successful patient profiles
        - Community validation: {useful_count} people validated similar experiences
        """)
                    
                    elif effectiveness_label == "Partially Effective":
                        st.warning(f"""
        ⚠️ **Evidence for 'Partially Effective' Prediction:**
        
        **📊 Historical Success Rate:**
        - **{partial_count} out of {total_cases} patients** ({partial_pct:.1f}%) rated this as **Partially Effective**
        - {effective_count} patients ({effective_pct:.1f}%) found it Effective
        - {not_effective_count} patients ({not_effective_pct:.1f}%) found it Not Effective
        - Average rating: **{avg_rating:.1f}/10**
        
        **🎯 Why the Model Chose 'Partially Effective':**
        - Drug-condition match: {selected_drug} shows mixed results for {selected_condition}
        - Historical data indicates variable outcomes ({partial_pct:.0f}% partial success)
        - Pattern recognition: Your inputs match patients with moderate relief
        - Some benefit expected, but not complete pain resolution
        """)
                    
                    else:  # Not Effective
                        st.error(f"""
        ❌ **Evidence for 'Not Effective' Prediction:**
        
        **📊 Historical Success Rate:**
        - **{not_effective_count} out of {total_cases} patients** ({not_effective_pct:.1f}%) rated this as **Not Effective**
        - {effective_count} patients ({effective_pct:.1f}%) found it Effective
        - {partial_count} patients ({partial_pct:.1f}%) found it Partially Effective
        - Average rating: **{avg_rating:.1f}/10**
        
        **🎯 Why the Model Chose 'Not Effective':**
        - Drug-condition match: {selected_drug} has poor track record for {selected_condition}
        - Historical data shows {not_effective_pct:.0f}% reported no benefit
        - Pattern recognition: Your inputs match unsuccessful patient profiles
        - Consider discussing alternative medications with healthcare provider
        """)
                    
                    # Add comparison to alternatives
                    st.markdown("---")
                    st.markdown("**💊 How Does This Compare to Alternatives?**")
                    
                    # Get top 5 alternative drugs for this condition
                    # Use processed data since it has 'effectiveness' column
                    condition_drugs = data['processed'][data['processed']['condition'] == selected_condition]
                    drug_effectiveness = condition_drugs.groupby('drugName').agg({
                        'effectiveness': lambda x: (x == 'Effective').sum() / len(x) * 100,
                        'uniqueID': 'count'
                    }).reset_index()
                    drug_effectiveness.columns = ['Drug', 'Effectiveness %', 'Patient Count']
                    drug_effectiveness = drug_effectiveness[drug_effectiveness['Patient Count'] >= 5]  # At least 5 reviews
                    drug_effectiveness = drug_effectiveness.sort_values('Effectiveness %', ascending=False).head(5)
                    
                    if len(drug_effectiveness) > 0:
                        # Highlight current drug
                        drug_effectiveness['Highlight'] = drug_effectiveness['Drug'] == selected_drug
                        
                        fig_comparison = px.bar(
                            drug_effectiveness,
                            x='Drug',
                            y='Effectiveness %',
                            title=f'Top Medications for {selected_condition}',
                            color='Highlight',
                            color_discrete_map={True: '#1f77b4', False: '#cccccc'},
                            hover_data=['Patient Count']
                        )
                        fig_comparison.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        st.caption(f"💡 **{selected_drug}** is highlighted. Higher % = more patients rated it as 'Effective'. Patient count shows review sample size.")
                    else:
                        st.info(f"ℹ️ Not enough data to compare alternatives for {selected_condition}")
                
                else:
                    # No similar cases found
                    st.warning(f"""
        ⚠️ **Limited Historical Data**
        
        The model predicts **{effectiveness_label}** with {confidence:.1%} confidence, but:
        - No exact matches for {selected_drug} + {selected_condition} in our dataset
        - Prediction is based on similar drug classes and condition patterns
        - Consider this a **lower confidence** prediction
        
        **Recommendation:** Consult healthcare provider for medications with more established data.
        """)
                
                st.markdown("---")
                st.markdown("**⚙️ Model Feature Contributions (ACTUAL)**")
                st.caption("Based on feature importance from trained Random Forest model")
                
                # Load actual feature importance
                feat_importance = pd.read_csv(MODELS_DIR / "random_forest_v1" / "feature_importance.csv")
                
                # Calculate REAL contributions based on actual feature importance
                contributions = []
                
                # 1. usefulCount importance
                useful_importance = feat_importance[feat_importance['feature'] == 'usefulCount']['importance'].values[0]
                contributions.append(('Community Validation (usefulCount)', useful_importance * 100))
                
                # 2. Year importance
                year_importance = feat_importance[feat_importance['feature'] == 'year']['importance'].values[0]
                contributions.append(('Temporal Trends (year)', year_importance * 100))
                
                # 3. Text statistics (sum of review_length, word_count, avg_word_length)
                text_features = feat_importance[feat_importance['feature'].isin(['review_length', 'review_word_count', 'avg_word_length'])]
                text_importance = text_features['importance'].sum()
                contributions.append(('Review Text Statistics', text_importance * 100))
                
                # 4. Sentiment keywords (has_positive_keywords + has_negative_keywords)
                sentiment_features = feat_importance[feat_importance['feature'].isin(['has_positive_keywords', 'has_negative_keywords'])]
                sentiment_importance = sentiment_features['importance'].sum()
                contributions.append(('Keyword Sentiment (NOT compound!)', sentiment_importance * 100))
                
                # 5. Drug one-hot encoding
                drug_features = feat_importance[feat_importance['feature'].str.startswith('drug_')]
                drug_importance = drug_features['importance'].sum()
                contributions.append(('Drug Type (one-hot encoding)', drug_importance * 100))
                
                # 6. Condition one-hot encoding
                condition_features = feat_importance[feat_importance['feature'].str.startswith('condition_')]
                condition_importance = condition_features['importance'].sum()
                contributions.append(('Pain Condition (one-hot encoding)', condition_importance * 100))
                
                # 7. Other features
                accounted_importance = useful_importance + year_importance + text_importance + sentiment_importance + drug_importance + condition_importance
                other_importance = 1.0 - accounted_importance
                contributions.append(('Other Features', other_importance * 100))
                
                contrib_df = pd.DataFrame(contributions, columns=['Factor', 'Contribution %'])
                contrib_df = contrib_df.sort_values('Contribution %', ascending=True)
                
                fig_contrib = px.bar(
                    contrib_df,
                    x='Contribution %',
                    y='Factor',
                    orientation='h',
                    title='ACTUAL Feature Importance from Trained Model',
                    color='Contribution %',
                    color_continuous_scale='Viridis'
                )
                fig_contrib.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                st.caption("""
                💡 **Key Takeaways:**
                - Sentiment keywords have only **4.5%** influence (NOT the VADER compound score!)
                - Community validation (usefulCount) has **18%** influence
                - Drug + Condition one-hot encoding combined: **~19%** influence
                - Review text statistics (length, word count) have **~30%** influence
                
                **This explains why negative sentiment doesn't prevent "Effective" predictions!**
                """)
            
            st.markdown("---")
            
            # Show similar cases (using already calculated similar_cases)
            st.subheader("📊 Similar Cases from Dataset")
            
            # Use cleaned data for display (has readable columns without encoded features)
            similar_display = data['cleaned'][
                (data['cleaned']['drugName'] == selected_drug) & 
                (data['cleaned']['condition'] == selected_condition)
            ].head(5)
            
            if len(similar_display) > 0:
                st.dataframe(
                    similar_display[['drugName', 'condition', 'rating', 'usefulCount', 'year']],
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
    - Features: Drug name, condition, review text features, sentiment scores, temporal data, community validation
    - Performance: 66.46% accuracy (SMOTE-balanced, fixed data leakage)
    - Feature Engineering: 51 features + SMOTE oversampling for minority classes
    - Class Balance: SMOTE synthetic samples for "Partially Effective" and "Not Effective"
    
    **Note on Accuracy:** The model previously showed 99.80% accuracy when patient rating was included as a feature. However, this created data leakage since the rating is the outcome we're trying to predict. After removing rating from features, the model now shows TRUE predictive performance (66.46%) based on review text sentiment, drug type, condition, and temporal patterns - information available BEFORE treatment. The model uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the three classes for better prediction across all effectiveness levels.
    
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
            st.metric("Model Accuracy", "66.46%")
            st.caption("Balanced for all 3 classes")
        
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

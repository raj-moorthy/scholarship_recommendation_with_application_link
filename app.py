import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gender_guesser.detector as gender_detector
import pandas as pd

# Set Page Configuration for Premium Look
st.set_page_config(
    page_title="Scholarship AI Recommendation Platform",
    page_icon="🎓",
    layout="centered"
)

# Custom Premium Glassmorphic Design CSS
st.markdown("""
    <style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Core App Overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #f8fafc !important;
    }
    
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glassmorphic Container Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.12);
        margin-bottom: 25px;
        position: relative;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        border-top-left-radius: 24px;
        border-top-right-radius: 24px;
    }
    
    /* Typography Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Result Badges and Cards */
    .result-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
    }
    .badge-label {
        font-weight: 500;
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .badge-value-male {
        font-weight: 700;
        font-size: 1rem;
        color: #60a5fa;
    }
    .badge-value-female {
        font-weight: 700;
        font-size: 1rem;
        color: #f472b6;
    }
    .badge-value-neutral {
        font-weight: 700;
        font-size: 1rem;
        color: #f8fafc;
    }
    
    .scholarship-result-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 15px;
    }
    .scholarship-title {
        margin: 0;
        font-size: 0.8rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .scholarship-name {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.3;
        margin: 10px 0 20px 0;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .apply-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        color: #0f172a !important;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        text-decoration: none !important;
        font-size: 0.95rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
    }
    .apply-btn:hover {
        transform: translateY(-2px);
        background: #f8fafc;
        box-shadow: 0 15px 25px rgba(0,0,0,0.25);
    }
    
    /* Footer Styling */
    .premium-footer {
        margin-top: 40px;
        text-align: center;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.2);
        letter-spacing: 0.5px;
    }
    
    /* Form fields text colors fix */
    .stTextInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.25) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
    .stTextArea>div>div>textarea {
        background-color: rgba(0, 0, 0, 0.25) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Deep Learning Model Operations -----------------

@st.cache_resource
def load_ml_assets():
    # Cache assets to ensure ultra-fast load times on Streamlit Cloud
    model = tf.keras.models.load_model('scholarship_text_classifier.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    scholarship_data = pd.read_csv('tn_india_scholarships_2025.csv')
    detector = gender_detector.Detector()
    return model, tokenizer, label_encoder, scholarship_data, detector

# Safe load assets
try:
    model, tokenizer, label_encoder, scholarship_data, detector = load_ml_assets()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure model files are pushed to the repository.")

def predict_gender(name):
    first_name = name.strip().split(' ')[0]
    gender = detector.get_gender(first_name)
    if gender in ['male', 'mostly_male']:
        return 'Male'
    elif gender in ['female', 'mostly_female']:
        return 'Female'
    else:
        return 'Gender Neutral'

def predict_scholarship(user_text):
    seq = tokenizer.texts_to_sequences([user_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)
    pred_label = pred.argmax(axis=1)[0]
    scholarship_name = label_encoder.inverse_transform([pred_label])[0]

    try:
        scholarship_row = scholarship_data[scholarship_data['name'] == scholarship_name].iloc[0]
        official_website = scholarship_row['official_website']
    except IndexError:
        official_website = 'N/A'

    return scholarship_name, official_website

# ----------------- UI Layout Presentation -----------------

# Wrapper Title Card
st.markdown("""
    <div class="glass-card">
        <div class="main-title">🎓 Scholarship AI Recommendation</div>
        <div class="subtitle">Enter your academic details to match with tailored Indian scholarships</div>
    </div>
""", unsafe_allow_html=True)

# Form Fields Input block
with st.container():
    user_name = st.text_input("Full Name", placeholder="Enter your full name...")
    user_description = st.text_area(
        "Academic Profile Details", 
        placeholder="Describe your details: academic level (e.g. 12th, B.Tech, PhD), family annual income, community/caste, marks, and state..."
    )

# Centering Button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("Predict Matching Scholarship", use_container_width=True)

# ----------------- Execution & Results Render -----------------

if predict_btn:
    if user_name and user_description:
        with st.spinner("Analyzing profile with Deep Learning model..."):
            # Execute predictions
            gender = predict_gender(user_name)
            scholarship_name, website_link = predict_scholarship(user_description)
            
            # Format gender badge classes
            gender_class = "badge-value-neutral"
            gender_icon = "⚪"
            if gender == "Male":
                gender_class = "badge-value-male"
                gender_icon = "♂️"
            elif gender == "Female":
                gender_class = "badge-value-female"
                gender_icon = "♀️"
            
            # Render Premium Matching Result Layout
            st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 20px; color: #a5b4fc; display:flex; gap:8px; align-items:center;">
                        ✨ Matching Result Analysis
                    </div>
                    
                    <div class="result-badge">
                        <span class="badge-label">Inferred Gender:</span>
                        <span class="{gender_class}">{gender_icon} {gender}</span>
                    </div>
                    
                    <div class="scholarship-result-card">
                        <div class="scholarship-title">🎯 Best Matched Program</div>
                        <div class="scholarship-name">{scholarship_name}</div>
                        {"<a class='apply-btn' href='" + website_link + "' target='_blank'>🔗 Apply Officially</a>" if website_link != "N/A" else ""}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    else:
        st.error("⚠️ Please fill in both your Name and Profile Details to run the model.")

# App Premium Footer branding
st.markdown("""
    <div class="premium-footer">
        Powered by Deep Learning Classification Models &bull; Secure and Private Analysis
    </div>
""", unsafe_allow_html=True)

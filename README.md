# 🎓 Scholarship AI Recommendation Platform
[![Streamlit App](https://static.streamlit.io/badge_streamlit.svg)](https://share.streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

An advanced, premium-designed AI application that matches students with tailored scholarship opportunities using Deep Learning and Natural Language Processing (NLP). Featuring a custom **Indigo-Purple Glassmorphic Dark Theme** crafted with Streamlit UI overrides.

---

## ✨ Features

* 🚀 **Advanced Deep Learning Engine:** Classifies students' educational stage, income limits, and backgrounds into matched Indian scholarships.
* ♀️♂️ **Gender Inference Engine:** Utilizes the robust `gender-guesser` module to automatically infer gender from first names to filter gender-specific benefits.
* 🔗 **Direct Official Applications:** Dynamically matches results with active scholarship links from official government and private portals.
* 🎨 **Breathtaking Glassmorphic UI:** Injected custom CSS overlays to create modern dark gradients, blur backdrops, and glowing result cards.

---

## 🛠️ Technology Stack

* **Platform:** [Streamlit](https://streamlit.io/) (High-performance web app dashboard framework)
* **AI & Modeling:** [TensorFlow 2.16+ / Keras 3](https://keras.io/) (Deep Learning Classifier)
* **NLP & Tokenization:** Keras Tokenizer, Sequence Padding, and Pickle Serializers
* **Database & Filtering:** [Pandas](https://pandas.pydata.org/) (High-speed relational CSV matching)

---

## 📁 Repository Structure
Your streamlined workspace contains only essential components:

```text
├── app.py                            # Redesigned Streamlit web application & styling
├── requirements.txt                  # Python dependency ecosystem configurations
├── .python-version                   # Virtual environment version pinning (Python 3.11.0)
├── .gitignore                        # Staged exclusion lists for Git
├── scholarship_text_classifier.h5    # Pre-trained TensorFlow deep learning model
├── tokenizer.pkl                     # Saved NLP tokenizer dictionary
├── label_encoder.pkl                 # Multi-class target label encoder
├── tn_india_scholarships_2025.csv    # Official scholarship database with URLs
└── README.md                         # Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/raj-moorthy/scholarship_recommendation_with_application_link.git
cd scholarship_recommendation_with_application_link
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to view your local platform!

---

## ☁️ Deploy to Streamlit Community Cloud (100% Free)

Streamlit Community Cloud is optimized for hosting massive TensorFlow model servers.

1. Create a free account at **[share.streamlit.io](https://share.streamlit.io/)**.
2. Click **Create App** in your dashboard.
3. Configure the settings:
   * **Repository:** `raj-moorthy/scholarship_recommendation_with_application_link`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **Deploy!** Your app will go live in less than 60 seconds with persistent, high-performance hosting.

---

## 🛡️ License & Privacy
* **Secure Calculations:** Profile analysis runs completely in-memory inside the Streamlit instance and is never saved or tracked.
* **Accuracy:** Machine learning predictions are matching algorithms based on historic datasets. Always verify criteria on the official links provided.

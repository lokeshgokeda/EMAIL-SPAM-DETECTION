import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Spam Detection AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 35px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    /* Result cards */
    .spam-card {
        background: #fff1f2;
        border: 2px solid #fb7185;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
    }

    .ham-card {
        background: #ecfdf5;
        border: 2px solid #34d399;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
    }

    .result-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-description {
        font-size: 16px;
        color: #4b5563;
    }

    /* Statistics */
    .stat-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }

    .stat-number {
        font-size: 28px;
        font-weight: 700;
        color: #4f46e5;
    }

    .stat-label {
        font-size: 14px;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TFIDF_PATH = BASE_DIR / "models" / "sms_spam_tfidf_bigrams.pkl"
MODEL_PATH = BASE_DIR / "models" / "sms_spam_svm_bigrams.pkl"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # These pickle files were trained with scikit-learn 1.7.2.
    # Matching the training version avoids pickle compatibility warnings/errors.
    if sklearn.__version__ != "1.7.2":
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} is installed, but these models were trained with scikit-learn 1.7.2. "
            "Please deploy with scikit-learn==1.7.2."
        )

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found:\n{TFIDF_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVM model not found:\n{MODEL_PATH}"
        )

    tfidf_model = joblib.load(TFIDF_PATH)
    svm_model = joblib.load(MODEL_PATH)

    return tfidf_model, svm_model


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    tfidf, svm_model = load_models()

except Exception as e:

    st.error("❌ Unable to load the spam detection model.")

    st.code(str(e))

    st.info(
        "Make sure the following files are present in the models folder:\n\n"
        "• models/sms_spam_tfidf_bigrams.pkl\n"
        "• models/sms_spam_svm_bigrams.pkl\n\n"
        "Also make sure scikit-learn==1.7.2 is installed."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📱 SMS Spam Detection")

    st.markdown("---")

    st.subheader("🤖 Model Information")

    st.write("**Algorithm:** Support Vector Machine")

    st.write("**Features:** TF-IDF Bigrams")

    st.write("**Task:** Binary Classification")

    st.write("**Classes:** HAM / SPAM")

    st.write("**Reported Accuracy:** 96%")

    st.markdown("---")

    st.subheader("💡 How it works")

    st.write(
        """
        1. Enter an SMS message.
        2. The text is converted into TF-IDF features.
        3. The SVM model analyzes the features.
        4. The message is classified as HAM or SPAM.
        """
    )

    st.markdown("---")

    if st.button("🗑️ Clear Prediction History", use_container_width=True):

        st.session_state.history = []

        st.success("History cleared!")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📱 SMS Spam Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent machine learning system for detecting unwanted messages'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">96%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">SVM</div>
            <div class="stat-label">ML Algorithm</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">TF-IDF</div>
            <div class="stat-label">Text Features</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.history)}</div>
            <div class="stat-label">Predictions</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# EXAMPLE MESSAGES
# ============================================================

st.markdown("### 💬 Try an Example")

example_col1, example_col2, example_col3 = st.columns(3)

examples = {

    "🚨 Spam Example":
        "Congratulations! You have won a free prize! "
        "Click now to claim your reward!",

    "✅ Normal Example":
        "Hey, are we meeting tomorrow at 10 AM?",

    "💰 Offer Example":
        "URGENT! You have been selected for a special "
        "cash reward. Call now to claim."
}


# ============================================================
# MESSAGE INPUT
# ============================================================

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


with example_col1:

    if st.button(
        "🚨 Spam Example",
        use_container_width=True
    ):

        st.session_state.message_input = examples["🚨 Spam Example"]


with example_col2:

    if st.button(
        "✅ Normal Example",
        use_container_width=True
    ):

        st.session_state.message_input = examples["✅ Normal Example"]


with example_col3:

    if st.button(
        "💰 Offer Example",
        use_container_width=True
    ):

        st.session_state.message_input = examples["💰 Offer Example"]


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TEXT AREA
# ============================================================

message = st.text_area(
    "📝 Enter your SMS message",
    value=st.session_state.message_input,
    placeholder=(
        "Example: Congratulations! You won a free prize. "
        "Click here to claim!"
    ),
    height=180,
    max_chars=5000
)


# ============================================================
# MESSAGE STATISTICS
# ============================================================

word_count = len(message.split())

character_count = len(message)

stat1, stat2, stat3 = st.columns(3)

with stat1:

    st.metric(
        "Characters",
        character_count
    )

with stat2:

    st.metric(
        "Words",
        word_count
    )

with stat3:

    st.metric(
        "Remaining",
        max(0, 5000 - character_count)
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_col, clear_col = st.columns([3, 1])

with predict_col:

    predict_button = st.button(
        "🔍 Analyze Message",
        type="primary",
        use_container_width=True
    )

with clear_col:

    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True
    )


if clear_button:

    st.session_state.message_input = ""

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    cleaned_message = message.strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not cleaned_message:

        st.warning(
            "⚠️ Please enter an SMS message before analyzing."
        )

        st.stop()

    if len(cleaned_message) < 2:

        st.warning(
            "⚠️ Please enter a longer message."
        )

        st.stop()

    # --------------------------------------------------------
    # SHOW SPINNER
    # --------------------------------------------------------

    with st.spinner("🤖 Analyzing your message..."):

        try:

            # Convert text to TF-IDF
            X = tfidf.transform([cleaned_message])

            # Model prediction
            prediction = svm_model.predict(X)[0]

            # Decision score
            decision_score = float(
                svm_model.decision_function(X)[0]
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            if prediction == 1:

                label = "SPAM"

            else:

                label = "HAM"

            # ------------------------------------------------
            # SCORE NORMALIZATION
            # ------------------------------------------------

            # This is a display-oriented confidence indicator.
            # It is NOT a calibrated probability.

            confidence = 1 / (
                1 + np.exp(-abs(decision_score))
            )

            confidence_percentage = confidence * 100

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            st.session_state.history.append({

                "Message":
                    cleaned_message,

                "Prediction":
                    label,

                "Decision Score":
                    round(decision_score, 4),

                "Confidence":
                    f"{confidence_percentage:.2f}%"

            })

        except Exception as e:

            st.error(
                "❌ An error occurred while analyzing the message."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.markdown("## 🎯 Analysis Result")

    result_col1, result_col2 = st.columns([1, 1])


    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    with result_col1:

        if label == "SPAM":

            st.markdown(
                f"""
                <div class="spam-card">

                    <div class="result-title">
                        🚨 SPAM
                    </div>

                    <div class="result-description">
                        This message has been classified as
                        potentially unwanted or spam.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ham-card">

                    <div class="result-title">
                        ✅ HAM
                    </div>

                    <div class="result-description">
                        This message has been classified as
                        a legitimate message.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    with result_col2:

        st.metric(
            "Decision Score",
            f"{decision_score:.4f}"
        )

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )

        st.write(
            f"**Model score magnitude:** "
            f"{confidence_percentage:.2f}%"
        )

        st.caption(
            "The score is derived from the SVM decision function. "
            "It should not be interpreted as a calibrated probability."
        )


    # ========================================================
    # MESSAGE DETAILS
    # ========================================================

    st.markdown("---")

    st.markdown("### 🔎 Message Details")

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:

        st.info(
            f"**Classification:** {label}"
        )

    with detail_col2:

        st.info(
            f"**Characters:** {character_count}"
        )

    with detail_col3:

        st.info(
            f"**Words:** {word_count}"
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown("## 📊 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        📱 SMS Spam Detection AI &nbsp;•&nbsp;
        Machine Learning + NLP &nbsp;•&nbsp;
        SVM + TF-IDF
        <br>
        Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

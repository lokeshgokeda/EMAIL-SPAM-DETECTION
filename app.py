import streamlit as st
import joblib


# ============================================================
# LOAD SAVED MODEL
# ============================================================

tfidf = joblib.load(
    "sms_spam_tfidf_bigrams.pkl"
)

svm_model = joblib.load(
    "sms_spam_svm_bigrams.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📱 SMS Spam Classifier")

st.write(
    "Enter an SMS message below and the machine learning "
    "model will classify it as HAM or SPAM."
)


# ============================================================
# TEXT INPUT
# ============================================================

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You won a free prize!",
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict SMS"):

    if message.strip() == "":
        
        st.warning("Please enter an SMS message.")

    else:

        # Convert message to TF-IDF
        X = tfidf.transform([message])

        # Predict
        prediction = svm_model.predict(X)[0]

        # Decision score
        score = svm_model.decision_function(X)[0]

        # Convert prediction to label
        if prediction == 1:
            label = "SPAM"
        else:
            label = "HAM"


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader("Prediction")

        if label == "SPAM":
            st.error("🚨 SPAM")

        else:
            st.success("✅ HAM")


        st.write(
            f"**Decision Score:** {score:.4f}"
        )
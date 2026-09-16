# SMS Spam Detection AI

Streamlit machine-learning app for classifying SMS messages as HAM or SPAM using TF-IDF bigram features and a Linear SVM.

## Streamlit Cloud
- Main file: `app.py`
- Model files are in `models/`
- The application resolves model paths relative to `app.py`.
- The saved scikit-learn models were created with scikit-learn `1.7.2`; keep that version in the deployment environment.

## Project files
- `app.py` — Streamlit application
- `models/sms_spam_tfidf_bigrams.pkl` — final TF-IDF bigram vectorizer
- `models/sms_spam_svm_bigrams.pkl` — final Linear SVM classifier
- `models/sms_spam_tfidf.pkl` — earlier TF-IDF model
- `models/sms_spam_svm_model.pkl` — earlier SVM model
- `spam.csv` — dataset
- `notebooks/Untitled2.ipynb` — training/notebook work
- `notebooks/.ipynb_checkpoints/` — original notebook checkpoints

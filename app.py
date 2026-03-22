import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def predict_role(resume):
    vec = vectorizer.transform([resume])
    return model.predict(vec)[0]

def get_match_score(resume, job_desc):
    resume_vec = vectorizer.transform([resume])
    job_vec = vectorizer.transform([job_desc])
    score = cosine_similarity(resume_vec, job_vec)
    return round(score[0][0] * 100, 2)

st.title("🚀 AI Resume Analyzer")

resume = st.text_area("Paste Resume")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume and job_desc:
        role = predict_role(resume)
        score = get_match_score(resume, job_desc)

        st.write("Predicted Role:", role)
        st.write("Match Score:", score, "%")
    else:
        st.warning("Please fill both fields")
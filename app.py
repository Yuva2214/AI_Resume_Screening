import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Check if text is extracted
            text += extracted_text + " "
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description and resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity
    job_vector = vectors[0]  # First vector is the job description
    resume_vectors = vectors[1:]  # Remaining are resumes
    cosine_similarities = cosine_similarity([job_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit App UI
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job Description Input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Resume Upload
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("Processing Resumes...")

    # Extract text from each uploaded resume
    resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes_text)
    
    # Create a DataFrame for ranking (Score in percentage)
    results = pd.DataFrame({
        "Candidate": [file.name for file in uploaded_files],
        "Score (%)": [round(score * 100, 2) for score in scores]  # Convert to percentage
    }).sort_values(by="Score (%)", ascending=False)

    # Display ranked resumes
    st.subheader("Ranked Resumes")
    st.dataframe(results)

    # Download results as CSV
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Ranking as CSV", data=csv, file_name="resume_ranking.csv", mime="text/csv")

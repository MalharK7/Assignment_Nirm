from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df= pd.read_csv("E:\FastAPI_Nirmitee\Resume_Content.csv")

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Content'])

def get_relevant_resumes(job_description,df):
   job_desc_vector = vectorizer.transform([job_description.lower()])
   df['similarity'] = cosine_similarity(X, job_desc_vector).flatten()
   df = df.sort_values(by='similarity', ascending=False)
   return df[['name', 'similarity']].head()

app = FastAPI()

@app.get("/relevant-resumes/{job_description}")
def get_relevant_resume(job_description: str):
    get_relevant_resumes(job_description,df)
    return df[['name', 'similarity']].head()
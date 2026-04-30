from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

skills_db = ["Python","Java","SQL","Machine Learning","Flask","React","NLP","Data Analysis"]

def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def match_score(resume, jd):
    text = [resume, jd]
    cv = CountVectorizer()
    matrix = cv.fit_transform(text)
    score = cosine_similarity(matrix)[0][1] * 100
    return round(score,2)

def find_skills(text):
    found=[]
    for skill in skills_db:
        if skill.lower() in text.lower():
            found.append(skill)
    return found

@app.route("/", methods=["GET","POST"])
def home():
    score=None
    skills=[]
    missing=[]
    
    if request.method=="POST":
        resume_file=request.files["resume"]
        jd=request.form["jobdesc"]

        resume_text=extract_text(resume_file)

        score=match_score(resume_text,jd)

        skills=find_skills(resume_text)

        for skill in skills_db:
            if skill.lower() not in resume_text.lower():
                missing.append(skill)

    return render_template("index.html",score=score,skills=skills,missing=missing)

if __name__=="__main__":
    app.run(debug=True)

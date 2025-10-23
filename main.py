from flask import Flask, render_template, request
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# ------------------ Text Extraction ------------------ #
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower().replace("\n", " ").strip()

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path).lower().replace("\n", " ").strip()

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower().replace("\n", " ").strip()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# ------------------ Flask Routes ------------------ #
@app.route('/')
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').lower().replace("\n", " ").strip()
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resumes.append(extract_text(filename))

        # ------------------ Vectorization & Similarity ------------------ #
        vectorizer = TfidfVectorizer(stop_words='english').fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # ------------------ Top Matches & Feedback ------------------ #
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i]*100, 2) for i in top_indices]  # percentage

        # Generate feedback dynamically
        feedback_list = []
        for score in similarity_scores:
            if score >= 80:
                feedback_list.append("Excellent match! 🔥")
            elif score >= 60:
                feedback_list.append("Good fit! 👍")
            elif score >= 40:
                feedback_list.append("Average match, could be improved.")
            else:
                feedback_list.append("Low match, consider updating resume.")

        return render_template(
            'matchresume.html',
            message="Top matching resumes:",
            top_resumes=top_resumes,
            similarity_scores=similarity_scores,
            feedback_list=feedback_list
        )

    return render_template('matchresume.html')


# ------------------ Run App ------------------ #
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

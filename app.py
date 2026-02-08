from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1] * 100

    result = "Spam" if pred == 1 else "Not Spam"
    confidence = f"{prob:.2f}%"

    return render_template("index.html",
                           prediction=result,
                           confidence=confidence,
                           message=message)

if __name__ == "__main__":
    app.run(debug=True)

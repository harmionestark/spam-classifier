# 📩 Spam Message Detector

A machine learning-powered web application that classifies text messages as Spam or Not Spam using Natural Language Processing (NLP). Built with Flask, Scikit-learn, and TF-IDF, this project demonstrates end-to-end ML deployment — from model training to a live web interface.

🚀 Features
- Logistic Regression model trained on real SMS spam dataset
- Text preprocessing and TF-IDF vectorization
- Clean Flask web interface
- Real-time spam prediction
- Simple and beginner-friendly project structure

🧰 Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- HTML/CSS

📂 Project Structure
spam-classifier/
│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── spam.csv
│── requirements.txt
│── templates/
│   └── index.html
│── static/
│   └── style.css

⚙️ Setup & Run Locally
1. Clone the repository
git clone https://github.com/harmionestark/spam-classifier.git
cd spam-classifier

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

4. Train the model (optional)
python train_model.py

5. Run the app
python app.py

Then open:
http://127.0.0.1:5000/

🧪 Example Predictions
"Congratulations! You won ₹50,000. Click now!" → Spam
"Hey, are we meeting tomorrow?" → Not Spam

📌 Learning Outcomes
- Practical NLP pipeline
- ML model training and serialization
- Flask backend development
- Frontend-backend integration
- GitHub project deployment

📜 License
This project is licensed under the MIT License.

⭐ If you found this helpful, give the repo a star!

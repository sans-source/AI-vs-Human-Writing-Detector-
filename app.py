from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home route -> show form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route -> handle form submission
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]  # get text from form
        text_vectorized = vectorizer.transform([text])  # transform input
        prediction = model.predict(text_vectorized)[0]  # make prediction

        result = "AI Generated" if prediction == 1 else "Human Written"
        return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

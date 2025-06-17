from flask import Flask, request, render_template
import joblib

from flask import Flask

app = Flask(__name__, template_folder="../templates")

# Model ve TF-IDF dosyalarını yükle
model = joblib.load("naivebayes_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    tahmin = None
    if request.method == "POST":
        yorum = request.form["yorum"]  # Kullanıcının gönderdiği yorum
        yorum_tfidf = tfidf.transform([yorum])  # Yorumdan özellik çıkar
        tahmin = model.predict(yorum_tfidf)[0]  # Tahmin yap
    return render_template("index.html", tahmin=tahmin)

if __name__ == "__main__":
    app.run(debug=True)

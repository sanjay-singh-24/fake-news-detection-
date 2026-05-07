from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#Load the pre-trained model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. This captures the text from the HTML textarea
        message = request.form['message'] 
        data = [message]
        # 2. This converts the text into numbers (The actual "Function")
        vect = vectorizer.transform(data).toarray()
        # 3. This tells the AI model to give its result
        prediction = model.predict(vect)
        # 4. This sends the result back to your browser
        return render_template('index.html', prediction=prediction[0])
      # return render_template('index.html', prediction=result, text=news)
if __name__ == '__main__':
    app.run(debug=True)
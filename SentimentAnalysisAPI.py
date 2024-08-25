from flask import Flask,request,json,jsonify
from SentimentAnalysis import SentimentAnalysisModel

vector_model_path = "vecter.pkl"
model_path = "NLP.pkl"
sentiment_model = SentimentAnalysisModel(vector_model_path, model_path)

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():  
    if request.method == "GET":
        data = "Hello World"
        return jsonify({"data": data})

@app.route("/sentiment_analysis", methods=["GET"])
def sentiment_analysis_api():
    text = request.args.get("text", None)

    if text:
        # Perform sentiment analysis
        sentiment = sentiment_model.predict_sentiment(text)

        # Return the sentiment result
        response = {"Sentiment": sentiment}
        return jsonify(response)
    else:
        # If 'text' parameter is not found, return an error message
        response = {"Sentiment": "Text parameter not found in request"}
        return jsonify(response)

if __name__ == "__main__": 
    app.run(port = 5000, debug = True)
    
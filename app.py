from flask import Flask, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define label mapping
label_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Define prediction route with text in the URL
@app.route('/predict/<path:mytext>', methods=['GET'])
def predict(mytext):
    # Transform the text using the loaded TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([mytext])
    
    # Make a prediction
    prediction = model.predict(transformed_text)
    
    # Get the predicted label
    predicted_label = label_mapping.get(int(prediction[0]), 'Unknown')

    # Return both the predicted class and the label
    return jsonify({'prediction': int(prediction[0]), 'label': predicted_label})

# Define home route for testing
@app.route('/')
def home():
    return "ML Model API is up and running!"

# Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)

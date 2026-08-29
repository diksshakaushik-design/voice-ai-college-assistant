from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import json
import random
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("chatbot_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load label encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Load intents
with open("intents.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

# Get maximum sequence length
MAX_LENGTH = model.input_shape[1]


def get_response(message):
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([message.lower()])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    # Predict intent
    prediction = model.predict(padded_sequence, verbose=0)

    # Get highest probability
    predicted_index = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_index])

    # Convert index to intent name
    predicted_tag = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Find response
    response = "Sorry, I don't understand your question."

    for intent in intents_data["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            break

    return predicted_tag, confidence, response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please provide a message."
        })

    intent, confidence, response = get_response(message)

    return jsonify({
        "message": message,
        "intent": intent,
        "confidence": round(confidence * 100, 2),
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)
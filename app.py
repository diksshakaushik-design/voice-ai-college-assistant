import tensorflow as tf
import pickle
import json
import random
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load trained model
model = tf.keras.models.load_model("chatbot_model_fixed.keras")

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



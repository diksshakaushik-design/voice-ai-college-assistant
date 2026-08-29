import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Load dataset
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

sentences = []
labels = []

# Extract patterns and labels
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern.lower())
        labels.append(intent["tag"])

print("Total training sentences:", len(sentences))
print("Total intents:", len(set(labels)))

# Tokenization
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

# Padding
max_length = max(len(sequence) for sequence in sequences)

X = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post"
)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Save tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Save label encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# Number of classes
num_classes = len(label_encoder.classes_)

# Build LSTM model
model = Sequential([
    Embedding(
        input_dim=2000,
        output_dim=128,
        input_length=max_length
    ),

    LSTM(128),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display architecture
model.summary()

# Train model
history = model.fit(
    X,
    y,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# Save trained model
model.save("chatbot_model.keras")

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")
print("Model saved as: chatbot_model.keras")
print("Tokenizer saved as: tokenizer.pkl")
print("Label encoder saved as: label_encoder.pkl")
print("Maximum sequence length:", max_length)
print("===================================")
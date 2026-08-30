import streamlit as st
import tensorflow as tf
import pickle
import json
import random
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import io

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voice AI College Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Voice AI College Assistant")

st.write(
    "🎤 Speak your college-related question and "
    "I will answer you."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "chatbot_model_fixed.keras",
        compile=False
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

@st.cache_resource
def load_label_encoder():

    with open("label_encoder.pkl", "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD INTENTS
# ============================================================

@st.cache_data
def load_intents():

    with open(
        "intents.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()
    tokenizer = load_tokenizer()
    label_encoder = load_label_encoder()
    intents_data = load_intents()

    model_loaded = True

except Exception as e:

    model_loaded = False
    model_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 College Assistant")

    st.write("You can ask about:")

    st.write("• College information")
    st.write("• Courses")
    st.write("• Admissions")
    st.write("• Departments")
    st.write("• Fees")
    st.write("• Exams")
    st.write("• Campus")
    st.write("• General college queries")

    st.divider()

    st.info(
        "🎤 Click the microphone below "
        "and speak your question."
    )


# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:

    st.error(
        "❌ Unable to load chatbot model."
    )

    st.code(model_error)

    st.stop()


st.success(
    "✅ Chatbot model loaded successfully!"
)


# ============================================================
# RESPONSE FUNCTION
# ============================================================

def get_response(intent_name):

    for intent in intents_data["intents"]:

        if intent["tag"] == intent_name:

            responses = intent.get(
                "responses",
                []
            )

            if responses:

                return random.choice(
                    responses
                )

    return (
        "Sorry, I don't have information "
        "about that."
    )


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(text):

    sequence = tokenizer.texts_to_sequences(
        [text.lower()]
    )

    max_length = model.input_shape[1]

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    confidence = float(
        prediction[0][predicted_index]
    )

    predicted_tag = (
        label_encoder
        .inverse_transform(
            [predicted_index]
        )[0]
    )

    return predicted_tag, confidence


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_bytes):

    recognizer = sr.Recognizer()

    try:

        audio_file = io.BytesIO(
            audio_bytes
        )

        with sr.AudioFile(
            audio_file
        ) as source:

            audio = recognizer.record(
                source
            )

        text = recognizer.recognize_google(
            audio
        )

        return text, None

    except sr.UnknownValueError:

        return (
            None,
            "Sorry, I could not understand your speech."
        )

    except sr.RequestError as e:

        return (
            None,
            f"Speech recognition service error: {e}"
        )

    except Exception as e:

        return (
            None,
            f"Audio processing error: {e}"
        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    try:

        tts = gTTS(
            text=text,
            lang="en",
            slow=False
        )

        audio_buffer = io.BytesIO()

        tts.write_to_fp(
            audio_buffer
        )

        audio_buffer.seek(0)

        return audio_buffer.read()

    except Exception:

        return None


# ============================================================
# VOICE INPUT
# ============================================================

st.subheader("🎤 Ask Your Question")

audio_value = st.audio_input(
    "Click the microphone and speak"
)


# ============================================================
# PROCESS VOICE
# ============================================================

if audio_value:

    st.audio(
        audio_value,
        format="audio/wav"
    )

    with st.spinner(
        "🎧 Listening and understanding..."
    ):

        user_text, speech_error = (
            speech_to_text(
                audio_value.getvalue()
            )
        )


    # --------------------------------------------------------
    # SPEECH ERROR
    # --------------------------------------------------------

    if speech_error:

        st.error(speech_error)


    # --------------------------------------------------------
    # SUCCESSFUL SPEECH RECOGNITION
    # --------------------------------------------------------

    else:

        st.success(
            f"📝 You said: **{user_text}**"
        )


        # ====================================================
        # PREDICT INTENT
        # ====================================================

        with st.spinner(
            "🤖 Thinking..."
        ):

            intent, confidence = (
                predict_intent(
                    user_text
                )
            )


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence < 0.40:

            response = (
                "I'm not sure about that. "
                "Please ask me about courses, "
                "admissions, departments, fees, "
                "exams, campus, or college "
                "information."
            )

        else:

            response = get_response(
                intent
            )


        # ====================================================
        # DISPLAY RESPONSE
        # ====================================================

        st.subheader(
            "🤖 Assistant"
        )

        st.write(
            response
        )


        # ====================================================
        # SHOW PREDICTION
        # ====================================================

        with st.expander(
            "🔍 Prediction Details"
        ):

            st.write(
                f"**Intent:** {intent}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence:.2%}"
            )


        # ====================================================
        # TEXT TO SPEECH
        # ====================================================

        with st.spinner(
            "🔊 Preparing voice response..."
        ):

            voice_response = (
                text_to_speech(
                    response
                )
            )


        if voice_response:

            st.subheader(
                "🔊 Voice Response"
            )

            st.audio(
                voice_response,
                format="audio/mp3"
            )

        else:

            st.warning(
                "Voice response could not be generated."
    )

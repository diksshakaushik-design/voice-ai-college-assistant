import streamlit as st
from app import get_response

st.set_page_config(
    page_title="Voice AI College Assistant",
    page_icon="🎓"
)

st.title("🎓 Voice AI College Assistant")
st.write("Ask me anything about your college.")

message = st.text_input("Enter your question:")

if st.button("Ask"):
    if message.strip():
        response = get_response(message)
        st.success(response)
    else:
        st.warning("Please enter a question.")

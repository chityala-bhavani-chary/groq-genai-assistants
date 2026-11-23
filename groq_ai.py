# Groq AI - A Streamlit App for Conversational AI using Groq API
# This app allows users to interact with a conversational AI assistant powered by the Groq API. Users can type messages, upload files, images and receive responses in a chat-like interface.
# Note: Make sure to replace the API key with your own Groq API key for the app to function properly.

import streamlit as st
from groq import Groq
import os

st.set_page_config(page_title="Groq AI", page_icon="🤖")

st.title("🤖 Groq AI - Your AI Assistant")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    st.error("Set environment variable GROQ_API_KEY before running this app.")
    st.code("export GROQ_API_KEY='your_key_here'\nstreamlit run groq_ai.py", language="bash")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Add system conversation safely

if len(st.session_state.conversation) == 0:
    st.session_state.conversation.append(
        {"role": "system", 
         "content": "You are a helpful assistant."}
        )
    
# Show chat history
for message in st.session_state.conversation:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").write(user_input)

    st.session_state.conversation.append(
        {"role": "user", 
         "content": user_input}
         )
    api_messages = [
        m for m in st.session_state.conversation 
        if m["role"] in ["system", "user", "assistant"]
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
        temperature=0.7,
        max_tokens=2048,
    )
    bot_reply = response.choices[0].message.content

    st.session_state.conversation.append(
        {"role": "assistant", 
         "content": bot_reply}
         )
    st.chat_message("assistant").write(bot_reply)

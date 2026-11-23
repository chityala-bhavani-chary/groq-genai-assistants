# Groq AI - A Streamlit App for Conversational AI using Groq API
# This app allows users to interact with a conversational AI assistant powered by the Groq API. Users can type messages, upload files, images and receive responses in a chat-like interface.
# Note: Make sure to replace the API key with your own Groq API key for the app to function properly.

import streamlit as st
from groq import Groq
import os
import base64
import pypdf # Make sure to pip install pypdf

st.set_page_config(page_title="Groq AI", page_icon="🤖", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    st.error("Set environment variable GROQ_API_KEY before running this app.")
    st.code("export GROQ_API_KEY='your_key_here'\nstreamlit run groq_full_ai.py", language="bash")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Inject Custom CSS to format the interface into Google Gemini's signature design style
st.markdown("""
    <style>
    /* Center the title and match Gemini styling */
    .gemini-title {
        font-family: 'Google Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 500;
        background: linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Re-style text input box to blend seamlessly into a Gemini capsule row */
    div[data-testid="stTextInput"] input {
        border-radius: 28px 0px 0px 28px !important;
        border: 1px solid #dadce0 !important;
        border-right: none !important;
        padding: 14px 24px !important;
        background-color: #f0f4f9 !important;
        font-size: 16px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #dadce0 !important;
        box-shadow: none !important;
    }

    /* Format the Popover element to complete the Gemini capsule style container */
    div[data-testid="stPopover"] button {
        border-radius: 0px 28px 28px 0px !important;
        border: 1px solid #dadce0 !important;
        border-left: none !important;
        background-color: #f0f4f9 !important;
        padding: 14px 20px !important;
        height: 100% !important;
        color: #444746 !important;
    }
    div[data-testid="stPopover"] button:hover {
        background-color: #e9eef6 !important;
        color: #1f1f1f !important;
    }
    
    /* Make chat history look clean and spacious */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Render Gemini gradient title header
st.markdown('<h1 class="gemini-title">Hello, Groq AI</h1>', unsafe_allow_html=True)

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Initialize a key modifier to forcefully clear out and rebuild the input box widget
if "input_key_counter" not in st.session_state:
    st.session_state.input_key_counter = 0

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
            # If content is a list (image structure), display the text portion for history readability
            if isinstance(message["content"], list):
                st.write(message["content"][0]["text"])
            else:
                st.write(message["content"])

# --- GEMINI CAMPAIGN ROW INPUT SECTION ---

# Define the callback function to handle input submit safely
def handle_submit():
    # Use the dynamic key matching the state counter
    current_key = f"user_message_input_{st.session_state.input_key_counter}"
    user_input = st.session_state[current_key]
    
    if not user_input.strip():
        return

    active_model = "llama-3.3-70b-versatile" # Default model for text-only interactions
    final_content = user_input

    # Process uploaded file if one exists
    if st.session_state.uploaded_file_attachment:
        uploaded_file = st.session_state.uploaded_file_attachment
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # Scenario A: Handle Images (Multimodal Vision)
        if file_extension in ["png", "jpg", "jpeg"]:
            active_model = "llama-3.2-11b-vision-preview" 
            bytes_data = uploaded_file.read()
            base64_image = base64.b64encode(bytes_data).decode("utf-8")
            
            final_content = [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{file_extension};base64,{base64_image}"}
                }
            ]
            
        # Scenario B: Handle Documents (Extract and attach to prompt)
        elif file_extension in ["txt", "pdf", "csv"]:
            extracted_text = ""
            if file_extension == "txt":
                extracted_text = uploaded_file.read().decode("utf-8")
            elif file_extension == "pdf":
                pdf_reader = pypdf.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() or ""
            elif file_extension == "csv":
                extracted_text = uploaded_file.read().decode("utf-8")
                
            final_content = f"Context from uploaded file ({uploaded_file.name}):\n{extracted_text}\n\nUser Question: {user_input}"

    st.session_state.conversation.append(
        {"role": "user", 
         "content": final_content}
         )
         
    api_messages = [
        m for m in st.session_state.conversation 
        if m["role"] in ["system", "user", "assistant"]
    ]

    response = client.chat.completions.create(
        model=active_model,
        messages=api_messages,
        temperature=0.7,
        max_tokens=2048,
    )
    bot_reply = response.choices[0].message.content

    st.session_state.conversation.append(
        {"role": "assistant", 
         "content": bot_reply}
         )
         
    # Increment counter to force Streamlit to wipe the slate clean for the input box element
    st.session_state.input_key_counter += 1

# Create a clean layout container for the bottom Gemini-style entry bar
input_container = st.container()

with input_container:
    # Use side-by-side split column layout to structure a unified component look
    col1, col2 = st.columns([0.88, 0.12], gap="small", vertical_alignment="bottom")
    
    with col1:
        st.text_input(
            "Chat Input",
            placeholder="Enter a prompt here (Press Enter to send)...",
            label_visibility="collapsed",
            key=f"user_message_input_{st.session_state.input_key_counter}",
            on_change=handle_submit
        )
        
    with col2:
        with st.popover("📎 Attach", use_container_width=True):
            st.file_uploader(
                "Upload file:", 
                type=["png", "jpg", "jpeg", "txt", "pdf", "csv"],
                label_visibility="collapsed",
                key="uploaded_file_attachment"
            )
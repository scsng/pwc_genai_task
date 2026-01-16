import streamlit as st
from utils.chat_client import ChatClient
from utils.logger import setup_logging
from langchain_core.messages import HumanMessage
import os
import logging

# Set up logging
setup_logging()
# Set page config
st.set_page_config(page_title="Legal help", page_icon="⚖️")

# Initialize chat client
@st.cache_resource
def get_chat_client():
    return ChatClient(
        base_url=os.getenv("INFERENCE_API_URL"),
        api_key=os.getenv("INFERENCE_API_KEY")
    )

chat_client = get_chat_client()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("⚖️ Legal help")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in chat_client.stream([HumanMessage(content=prompt)]):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                else:
                    full_response += str(chunk)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

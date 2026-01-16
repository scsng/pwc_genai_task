import streamlit as st
from utils.chat_client import ChatClient
from utils.agents.agentic_workflow import AgenticWorkflow
from utils.logger import setup_logging
from langchain_core.messages import HumanMessage, AIMessage
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
        model=os.getenv("MODEL", "default-model"),
        api_key=os.getenv("INFERENCE_API_KEY")
    )

# Initialize agentic workflow
@st.cache_resource
def get_agentic_workflow():
    """Initialize the agentic workflow with LLM."""
    chat_client = get_chat_client()
    return AgenticWorkflow(llm=chat_client.llm)

agentic_workflow = get_agentic_workflow()

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
    
    # Get assistant response using agentic workflow
    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""
            
            # Convert chat history to LangChain messages
            chat_history = []
            for msg in st.session_state.messages[:-1]:  # Exclude the current prompt
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Stream the response from agentic workflow
            for chunk in agentic_workflow.stream(prompt, chat_history=chat_history):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            logging.error(f"Error in agentic workflow: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

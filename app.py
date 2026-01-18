import streamlit as st
from utils.chat_client import ChatClient
from utils.agents.agentic_workflow import AgenticWorkflow
from utils.logger import setup_logging
from langchain_core.messages import HumanMessage, AIMessage
from langfuse import get_client

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.rag.vector_db import QdrantDB

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
    max_task_count = int(os.getenv("MAX_TASK_COUNT", "3"))
    return AgenticWorkflow(llm=chat_client.llm,
                           vector_db=QdrantDB(collection_name=os.getenv("COLLECTION_NAME"),
                                              qdrant_host=os.getenv("QDRANT_HOST"),
                                              embedding_model=os.getenv("EMBEDDING_MODEL"),
                                              qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                                              top_k=os.getenv("TOP_K")),
                           max_task_count=max_task_count)

 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


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
            # Status placeholder for showing current node
            status_placeholder = st.empty()
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
            for event in agentic_workflow.stream(prompt, chat_history=chat_history):
                if isinstance(event, dict):
                    if event.get("type") == "node":
                        # Tell tha node which is executed in the moment
                        status_placeholder.info(f"**{event['display_name']}**")
                    elif event.get("type") == "answer":
                        # Remove status, only show the final answer
                        status_placeholder.empty()
                        full_response = event["content"]
                        message_placeholder.markdown(full_response)
                elif hasattr(event, 'content') and event.content:
                    # Legacy support for AIMessage (can be needed sometimes)
                    full_response += event.content
                    message_placeholder.markdown(full_response + "▌")
            
            if not full_response:
                message_placeholder.markdown("I couldn't generate a response.")
                full_response = "I couldn't generate a response."
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            logging.error(f"Error in agentic workflow: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

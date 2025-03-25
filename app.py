import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage


load_dotenv()

st.set_page_config(

    page_title="AI Chatbot"
    layout="centered"

)

st.title("AI Chabot")
st.subheader("Built with streamlit, Langchain and GPT-4o")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        model_name = "gpt-4o",
        temprature = 0.7,
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )

    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm = llm,
        memory = memory,
        verbose = False
    )
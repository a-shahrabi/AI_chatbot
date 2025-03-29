import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage


load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    layout="centered"
)

st.title("AI Chatbot")
st.subheader("Built with streamlit, Langchain and GPT-4o")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            st.stop()
            
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=api_key
        )

        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error initializing the chatbot: {str(e)}")
        st.stop()

# Custom message display function
def display_message(message, role):
    """Display a message with proper styling based on the role"""
    avatar = "👤" if role == "user" else "🤖"
  

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Take user input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pondering..."):
            response = st.session_state.conversation.predict(input=user_input)
            st.write(response)
    
    # Append to chat history
    st.session_state.chat_history.append(AIMessage(content=response))

with st.sidebar:
    st.title("Options")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

        memory = ConversationBufferMemory(return_messages=True)

        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        st.rerun()

    st.subheader("About")

    st.markdown(
        """ Chatbot uses:

        - **Streamlit** for the interface
        - **LangChain** for conversation
        - **GPT-4o** as our language model
        - **ConversationBufferMemory** to remember messages """
    )
# Adding some css configuration
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f7ff;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)
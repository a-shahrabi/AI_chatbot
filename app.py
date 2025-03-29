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
    message_container = st.container()
    with message_container:
        col1, col2 = st.columns([1, 12])
        with col1:
            st.markdown(f"<div class='avatar'>{avatar}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='chat-message {role}'>{message}</div>", unsafe_allow_html=True)
        return message_container

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

# try-except block for error handling
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Pondering..."):
                response = st.session_state.conversation.predict(input=user_input)
                st.write(response)
            # Append to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message or "429" in error_message:
                st.error("OpenAI API quota exceeded. Please check your OpenAI account billing details or try using a different model.")
            else:
                st.error(f"An error occurred: {error_message}")

with st.sidebar:
    st.title("Options")

    # Clear chat button (existing code)
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
    
    #  Export Chat 
    if st.session_state.chat_history:  # Only show export button if there's chat history
        # Create formatted chat text for export
        chat_export = "\n\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
            for msg in st.session_state.chat_history
        ])
        
        # Download button
        st.download_button(
            label="Export Chat History",
            data=chat_export,
            file_name="chatbot_conversation.txt",
            mime="text/plain",
            help="Download the current conversation as a text file"
        )

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
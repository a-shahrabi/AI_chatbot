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

# Check if the user has provided any input
if user_input:
    # Add the user's message to the chat history for record keeping
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display the user's message in the UI with the "user" avatar
    with st.chat_message("user"):
        st.write(user_input)

    # Display the assistant's response with the "assistant" avatar
    with st.chat_message("assistant"):
        try:
            # Show a loading spinner while waiting for the model's response
            with st.spinner("Pondering..."):
                # Generate a response using the conversation model
                response = st.session_state.conversation.predict(input=user_input)
                # Display the response in the UI
                st.write(response)
            
            # Add the assistant's response to the chat history for future context
            st.session_state.chat_history.append(AIMessage(content=response))
        
        except Exception as e:
            # Convert the exception to a string for error handling
            error_message = str(e)
            
            # Handle specific API quota errors with a user-friendly message
            if "insufficient_quota" in error_message or "429" in error_message:
                st.error("OpenAI API quota exceeded. Please check your OpenAI account billing details or try using a different model.")
            # Handle all other errors with a generic but informative message
            else:
                st.error(f"An error occurred: {error_message}")

# Add this in your sidebar section
with st.sidebar:
    st.title("Options")
    
    # Add personality selector - Creates a dropdown menu for users to choose different AI conversation styles
    st.subheader("Chatbot Personality")
    personality = st.selectbox(
        "Select a personality for your chatbot:",
        ["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"]
    )
    
   
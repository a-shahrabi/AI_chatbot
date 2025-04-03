import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
from langchain_community.llms import Ollama
import json
import time
from datetime import datetime


# Helper function that returns appropriate system messages based on the selected personality
def get_system_message(personality):
    """Return a system message based on the selected personality"""
    if personality == "Helpful Assistant":
        return "You are a helpful AI assistant. You provide clear and concise answers to user questions."
    elif personality == "Friendly Teacher":
        return "You are a friendly teacher. Explain concepts in simple terms and use examples to help users understand."
    elif personality == "Creative Writer":
        return "You are a creative writing assistant. Help users with creative and engaging content. Be imaginative and inspiring."
    elif personality == "Technical Expert":
        return "You are a technical expert. Provide detailed and accurate technical information. Use precise terminology."
    else:
        return "You are a helpful AI assistant."

def save_chat_history():
    """Save the current chat history to a local file"""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        # Convert chat messages to serializable format
        serializable_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                serializable_history.append({"role": "human", "content": msg.content, "timestamp": datetime.now().isoformat()})
            elif isinstance(msg, AIMessage):
                serializable_history.append({"role": "ai", "content": msg.content, "timestamp": datetime.now().isoformat()})
        
        # Generates a unique filename for the chat history using the current timestamp
        filename = f"chat_history_{int(time.time())}.json"
        try:
            # Open the file in write mode and save the chat history in JSON format
            with open(filename, "w") as f:
                json.dump(serializable_history, f)
            return filename
        except Exception as e:
            st.warning(f"Could not save chat history: {str(e)}")
            return None
    return None

def load_chat_history(filename):
    """Load chat history from a file"""
    try:
        with open(filename, "r") as f:
            serialized_history = json.load(f)
        
        # Convert back to LangChain message objects
        history = []
        for msg in serialized_history:
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history.append(AIMessage(content=msg["content"]))
        
        return history
    except Exception as e:
        st.warning(f"Could not load chat history: {str(e)}")
        return []

def get_ai_response(user_input, max_retries=3, retry_delay=2):
    """Get AI response with retry logic for API errors"""
    for attempt in range(max_retries):
        try:
            return st.session_state.conversation.predict(input=user_input)
        except Exception as e:
            error_message = str(e)
            if attempt < max_retries - 1:
                # If it's not the last attempt, retry after delay
                time.sleep(retry_delay)
                # If quota error, try fallback to cheaper model
                if "quota" in error_message.lower() or "429" in error_message:
                    try:
                        # Temporarily switch to more affordable model
                        backup_llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0.7,
                            openai_api_key=os.getenv("OPENAI_API_KEY")
                        )
                        memory = st.session_state.conversation.memory
                        temp_conversation = ConversationChain(
                            llm=backup_llm,
                            memory=memory,
                            verbose=False
                        )
                        return temp_conversation.predict(input=user_input)
                    except:
                        # Continue with next retry if backup also fails
                        continue
            else:
                # If all retries failed, raise the exception again
                raise


# Load environment variables from .env file
load_dotenv()

# Configure the page layout and title
st.set_page_config(
    page_title="AI Chatbot",
    layout="centered"
)

# Set the main title of the application
st.title("AI Chatbot")
# Set the subtitle describing the technologies used
st.subheader("Built with streamlit, Langchain and GPT-4o")

# Initialize chat history if it doesn't exist in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    try:
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            st.stop()

        # Initialize the language model with GPT-4o    
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=api_key
        )

        # Create memory object to store conversation history    
        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )

    # Handle any exceptions during initialization    
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
        # First try with the LLM
        with st.spinner("Pondering..."):
            # Use our new function with retry logic
            response = get_ai_response(user_input)
            st.write(response)
        
        st.session_state.chat_history.append(AIMessage(content=response))
        # Auto-save chat history after each successful response
        save_chat_history()
    
    except Exception as e:
        error_message = str(e)
        
        # Rule-based response 
        if "quota" in error_message.lower() or "429" in error_message or "insufficient_quota" in error_message.lower():
            st.warning("Using fallback mode due to API limitations")
            
            # Simple rule-based responses
            response = "I'm in fallback mode due to API limitations. I can only provide basic responses at the moment."
            
            
            if user_input is not None:
                # Check for different keywords in the user's message
                if "hello" in user_input.lower() or "hi" in user_input.lower():
                    response = "Hello! I'm currently running in fallback mode. How can I help you?"
                elif "how are you" in user_input.lower():
                    response = "I'm doing well, though I'm currently operating in a limited fallback mode."
                elif "help" in user_input.lower():
                    response = "I'm in fallback mode right now, but I'd be happy to help as best I can."
                elif "thank" in user_input.lower():
                    response = "You're welcome! Happy to assist even in fallback mode."
                elif "what is" in user_input.lower() or "who is" in user_input.lower() or "how to" in user_input.lower():
                    response = "That's an interesting question about '" + user_input + "'. When I'm back to full functionality, I'll provide a more detailed answer."
                elif "?" in user_input:
                    response = "That's a good question. Unfortunately, I'm currently in fallback mode and can't provide a complete answer."
            
            # Display the fallback response
            st.write(response)
            
            # Fallback response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
        
            st.error(f"An error occurred: {error_message}")

# Sidebar section for user options and controls
with st.sidebar:
    st.title("Options")
    
    # Model selection to handle API quota issues
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Select OpenAI model:",
        ["gpt-3.5-turbo", "gpt-4o"],
        index=0,  # Default to gpt-3.5-turbo (more affordable)
        help="GPT-3.5 Turbo uses less API quota than GPT-4o"
    )
    
    st.sidebar.write(f"Currently using model: {model_name}") 

    # Add a button to apply model changes without clearing chat
    if st.sidebar.button("Apply Model Change"):
        try:
            # Update the model based on current selection
            new_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Update the conversation chain with the new model
            memory = st.session_state.conversation.memory
            st.session_state.conversation = ConversationChain(
                llm=new_llm,
                memory=memory,
                verbose=False
            )
            
            st.sidebar.success(f"Successfully switched to {model_name}")
        except Exception as e:
            st.sidebar.error(f"Error switching models: {str(e)}")

    # Add personality selector - Creates a dropdown menu for users to choose different AI conversation styles
    st.subheader("Chatbot Personality")
    personality = st.selectbox(
        "Select a personality for your chatbot:",
        ["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"]
    )

    # Add a button to apply personality changes without clearing chat
    if st.sidebar.button("Apply Personality Change"):
        try:
            # Get the system message for the selected personality
            system_message = get_system_message(personality)
            
            # Update the conversation with the new personality
            st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            st.sidebar.success(f"Successfully switched to {personality} personality!")
        except Exception as e:
            st.sidebar.error(f"Error switching personality: {str(e)}")
    
    # Add chat history management section - NEW SECTION ADDED HERE
    st.subheader("Chat History Management")
    
    # Option to save current chat explicitly
    if st.sidebar.button("Save Current Chat"):
        saved_file = save_chat_history()
        if saved_file:
            st.sidebar.success(f"Chat saved to {saved_file}")
    
    # Option to load a previous chat
    history_files = [f for f in os.listdir('.') if f.startswith('chat_history_') and f.endswith('.json')]
    if history_files:
        selected_file = st.sidebar.selectbox("Load saved chat:", history_files)
        if st.sidebar.button("Load Selected Chat"):
            loaded_history = load_chat_history(selected_file)
            if loaded_history:
                st.session_state.chat_history = loaded_history
                st.sidebar.success("Chat history loaded successfully!")
                st.experimental_rerun()
    
# Clear chat button - Resets the conversation history and reinitializes the chatbot
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    memory = ConversationBufferMemory(return_messages=True)
    
    # Update the system message based on personality
    system_message = get_system_message(personality)
    
    
    st.session_state.model_name = model_name
    
    # Model selection
    llm = ChatOpenAI(
        model_name=st.session_state.model_name,
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    st.sidebar.success(f"Now using {st.session_state.model_name}")
    st.rerun()
    
    # Information about model usage
    st.info("💡 If you encounter quota errors, use gpt-3.5-turbo which is more affordable.")



# Initialize the conversation with default settings if it doesn't exist in session state
if "conversation" not in st.session_state:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            st.stop()
        
        # Set default personality when app first loads
        personality = "Helpful Assistant"  # Default personality
        system_message = get_system_message(personality)
        
        # Use gpt-3.5-turbo as the default model to avoid quota issues
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Changed from "gpt-4o" to reduce API costs
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
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
import json
import time
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure the page layout and title
st.set_page_config(
    page_title="AI Chatbot",
    layout="centered"
)

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
            # Return the filename so it can be referenced later
            return filename
        except Exception as e:
            # Display a warning if saving fails and explain the error
            st.warning(f"Could not save chat history: {str(e)}")
            # Return None to indicate the save operation failed
            return None
    return None
def load_chat_history(filename):
    """Load chat history from a file"""
    try:
        # Open the specified filename in read mode
        with open(filename, "r") as f:
            # Parse the JSON content of the file into a Python object
            # This converts the stored JSON string back into a Python data structure
            serialized_history = json.load(f)
        
        # Convert back to LangChain message objects
        history = []
        for msg in serialized_history:
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                # Create an AIMessage object with the stored content and add to history
                history.append(AIMessage(content=msg["content"]))
        
        return history
    except Exception as e:
        st.warning(f"Could not load chat history: {str(e)}")
        return []

def initialize_conversation(model_name="gpt-3.5-turbo", personality="Helpful Assistant"):
    """Initialize or reinitialize the conversation with specified settings"""
    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()
    
    # Get system message for selected personality
    system_message = get_system_message(personality)
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        openai_api_key=api_key
    )

    # Create memory object and conversation chain
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    # Set the system message in the conversation prompt template
    conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
    
    return conversation

def get_rule_based_response(user_input):
    """Generate a simple rule-based response when API is unavailable"""
    # Default fallback message
    response = "I'm in fallback mode due to API limitations. I can only provide basic responses at the moment."
    
    # Check for common patterns in user input
    if any(greeting in user_input.lower() for greeting in ["hello", "hi", "hey"]):
        response = "Hello! I'm currently running in fallback mode. How can I help you?"
    elif "how are you" in user_input.lower():
        response = "I'm doing well, though I'm currently operating in a limited fallback mode."
    elif "help" in user_input.lower():
        response = "I'm in fallback mode right now, but I'd be happy to help as best I can."
    elif any(thanks in user_input.lower() for thanks in ["thank", "thanks", "appreciate"]):
        response = "You're welcome! Happy to assist even in fallback mode."
    # Question patterns
    elif any(q_word in user_input.lower() for q_word in ["what is", "who is", "how to", "where", "when", "why"]):
        response = f"That's an interesting question about '{user_input}'. When I'm back to full functionality, I'll provide a more detailed answer."
    elif "?" in user_input:
        response = "That's a good question. Unfortunately, I'm currently in fallback mode and can't provide a complete answer."
    
    return response

# Set the main title of the application
st.title("AI Chatbot")
# Set the subtitle describing the technologies used
st.subheader("Built with streamlit, Langchain and GPT-4o")

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"  # Default to cheaper model

if "personality" not in st.session_state:
    st.session_state.personality = "Helpful Assistant"

if "fallback_mode" not in st.session_state:
    st.session_state.fallback_mode = False

# Initialize conversation if it doesn't exist
if "conversation" not in st.session_state:
    try:
        st.session_state.conversation = initialize_conversation(
            model_name=st.session_state.model_name,
            personality=st.session_state.personality
        )
    except Exception as e:
        st.error(f"Error initializing the chatbot: {str(e)}")
        # Set fallback mode to true if there's an error on startup
        st.session_state.fallback_mode = True

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

# Process user input and generate response
if user_input:
    # Add the user's message to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display the user's message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        # If already in fallback mode, don't attempt API calls
        if st.session_state.fallback_mode:
            response = get_rule_based_response(user_input)
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            try:
                # Show spinner while processing
                with st.spinner("Thinking..."):
                    # Standard API call without streaming
                    llm = st.session_state.conversation.llm
                    memory = st.session_state.conversation.memory
                    
                    # Try to get response from LLM
                    try:
                        response = st.session_state.conversation.predict(input=user_input)
                        st.write(response)
                        st.session_state.chat_history.append(AIMessage(content=response))
                        save_chat_history()
                    except Exception as api_error:
                        error_str = str(api_error).lower()
                        
                        # Check for quota or rate limiting errors
                        if any(term in error_str for term in ["quota", "rate", "429", "insufficient"]):
                            st.warning("Using fallback mode due to API limitations")
                            st.session_state.fallback_mode = True
                            
                            # Generate rule-based response
                            fallback_response = get_rule_based_response(user_input)
                            st.write(fallback_response)
                            st.session_state.chat_history.append(AIMessage(content=fallback_response))
                        else:
                            # For other API errors
                            st.error(f"API error: {str(api_error)}")
                            error_response = "I encountered an error processing your request. Please try again."
                            st.write(error_response)
                            st.session_state.chat_history.append(AIMessage(content=error_response))
            
            except Exception as e:
                # Handle any other errors
                st.error(f"An unexpected error occurred: {str(e)}")
                error_response = "I encountered an unexpected error. Please try again later."
                st.write(error_response)
                st.session_state.chat_history.append(AIMessage(content=error_response))
                # Switch to fallback mode for safety
                st.session_state.fallback_mode = True

# Sidebar section for user options and controls
with st.sidebar:
    st.title("Options")
    
    # Fallback mode status and control
    st.subheader("Fallback Mode")
    st.write(f"Status: {'Active' if st.session_state.fallback_mode else 'Inactive'}")
    
    if st.session_state.fallback_mode and st.button("Try API Again"):
        st.session_state.fallback_mode = False
        st.success("Fallback mode deactivated. Will try using the API for the next message.")
    
    # Model selection
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Select OpenAI model:",
        ["gpt-3.5-turbo", "gpt-4o"],
        index=0 if st.session_state.model_name == "gpt-3.5-turbo" else 1,
        help="GPT-3.5 Turbo uses less API quota than GPT-4o"
    )
    
    st.write(f"Currently using model: {st.session_state.model_name}")

    # Apply model change button
    if st.button("Apply Model Change"):
        try:
            # Update session state
            st.session_state.model_name = model_name
            
            # If we're in fallback mode, don't try to create a new conversation
            if not st.session_state.fallback_mode:
                # Create new conversation with the selected model but keep the chat history
                memory = st.session_state.conversation.memory
                
                # Update the conversation with new model
                st.session_state.conversation = initialize_conversation(
                    model_name=model_name,
                    personality=st.session_state.personality
                )
                st.session_state.conversation.memory = memory
            
            st.success(f"Model changed to {model_name}")
        except Exception as e:
            st.error(f"Error switching models: {str(e)}")
            # If error occurs during model switch, enable fallback mode
            st.session_state.fallback_mode = True

    # Personality selector
    st.subheader("Chatbot Personality")
    personality = st.selectbox(
        "Select a personality for your chatbot:",
        ["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"],
        index=["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"].index(st.session_state.personality)
    )

    # Apply personality change button
    if st.button("Apply Personality Change"):
        try:
            # Update session state
            st.session_state.personality = personality
            
            # Only update the prompt if not in fallback mode
            if not st.session_state.fallback_mode:
                # Get the system message for the selected personality
                system_message = get_system_message(personality)
                
                # Update the conversation prompt template with the new personality
                st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            st.success(f"Personality changed to {personality}")
        except Exception as e:
            st.error(f"Error switching personality: {str(e)}")
    
    # Chat history management section
    st.subheader("Chat History Management")
    
    # Option to save current chat explicitly
    if st.button("Save Current Chat"):
        saved_file = save_chat_history()
        if saved_file:
            st.success(f"Chat saved to {saved_file}")
    
    # Option to load a previous chat
    history_files = [f for f in os.listdir('.') if f.startswith('chat_history_') and f.endswith('.json')]
    if history_files:
        selected_file = st.selectbox("Load saved chat:", history_files)
        if st.button("Load Selected Chat"):
            loaded_history = load_chat_history(selected_file)
            if loaded_history:
                st.session_state.chat_history = loaded_history
                st.success("Chat history loaded successfully!")
                st.experimental_rerun()

# Clear chat button
if st.button("Clear Chat History"):
    # Reset the chat history and fallback mode
    st.session_state.chat_history = []
    st.session_state.fallback_mode = False
    
    # Try to reinitialize the conversation with current settings
    try:
        st.session_state.conversation = initialize_conversation(
            model_name=st.session_state.model_name,
            personality=st.session_state.personality
        )
        st.success(f"Chat cleared. Using {st.session_state.model_name} with {st.session_state.personality} personality.")
    except Exception as e:
        st.error(f"Error reinitializing conversation: {str(e)}")
        st.session_state.fallback_mode = True
        st.warning("Started in fallback mode due to API issues.")
    
    st.rerun()

# Information about current mode
if st.session_state.fallback_mode:
    st.warning("⚠️ Running in fallback mode due to API limitations. Responses are rule-based and limited.")
else:
    st.info("💡 If you encounter quota errors, use gpt-3.5-turbo which is more affordable.")
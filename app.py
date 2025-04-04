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
    personalities = {
        "Helpful Assistant": "You are a helpful AI assistant. You provide clear and concise answers to user questions.",
        "Friendly Teacher": "You are a friendly teacher. Explain concepts in simple terms and use examples to help users understand.",
        "Creative Writer": "You are a creative writing assistant. Help users with creative and engaging content. Be imaginative and inspiring.",
        "Technical Expert": "You are a technical expert. Provide detailed and accurate technical information. Use precise terminology."
    }
    return personalities.get(personality, "You are a helpful AI assistant.")

def save_chat_history():
    """Save the current chat history to a local file"""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        # Convert chat messages to serializable format
        serializable_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                serializable_history.append({
                    "role": "human", 
                    "content": msg.content, 
                    "timestamp": datetime.now().isoformat()
                })
            elif isinstance(msg, AIMessage):
                serializable_history.append({
                    "role": "ai", 
                    "content": msg.content, 
                    "timestamp": datetime.now().isoformat()
                })
        
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

def initialize_conversation(model_name="gpt-3.5-turbo", personality="Helpful Assistant"):
    """Initialize or reinitialize the conversation with specified settings"""
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

# Main Application

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

# Initialize conversation if it doesn't exist
if "conversation" not in st.session_state:
    try:
        st.session_state.conversation = initialize_conversation(
            model_name=st.session_state.model_name,
            personality=st.session_state.personality
        )
    except Exception as e:
        st.error(f"Error initializing the chatbot: {str(e)}")
        st.stop()

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
        try:
            # Get AI response with retry logic
            with st.spinner("Pondering..."):
                response = get_ai_response(user_input)
                st.write(response)
            
            # Add response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Auto-save chat history after each successful response
            save_chat_history()
        
        except Exception as e:
            error_message = str(e)
            
            # Rule-based fallback response if API has issues
            if "quota" in error_message.lower() or "429" in error_message or "insufficient_quota" in error_message.lower():
                st.warning("Using fallback mode due to API limitations")
                
                # Simple rule-based responses based on keywords
                fallback_responses = {
                    "hello": "Hello! I'm currently running in fallback mode. How can I help you?",
                    "hi": "Hello! I'm currently running in fallback mode. How can I help you?",
                    "how are you": "I'm doing well, though I'm currently operating in a limited fallback mode.",
                    "help": "I'm in fallback mode right now, but I'd be happy to help as best I can.",
                    "thank": "You're welcome! Happy to assist even in fallback mode."
                }
                
                response = "I'm in fallback mode due to API limitations. I can only provide basic responses at the moment."
                
                # Check for keywords in user input
                for keyword, canned_response in fallback_responses.items():
                    if keyword in user_input.lower():
                        response = canned_response
                        break
                
                # Special handling for questions
                if "what is" in user_input.lower() or "who is" in user_input.lower() or "how to" in user_input.lower():
                    response = f"That's an interesting question about '{user_input}'. When I'm back to full functionality, I'll provide a more detailed answer."
                elif "?" in user_input:
                    response = "That's a good question. Unfortunately, I'm currently in fallback mode and can't provide a complete answer."
                
                # Display fallback response
                st.write(response)
                
                # Add fallback response to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
            else:
                # For other errors, display the error message
                st.error(f"An error occurred: {error_message}")

# Sidebar section for user options and controls
with st.sidebar:
    st.title("Options")
    
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
            
            # Create new conversation with the selected model but keep the chat history
            memory = st.session_state.conversation.memory
            new_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Update the conversation chain with the new model
            st.session_state.conversation = ConversationChain(
                llm=new_llm,
                memory=memory,
                verbose=False
            )
            
            # Update system message based on current personality
            system_message = get_system_message(st.session_state.personality)
            st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            st.success(f"Successfully switched to {model_name}")
        except Exception as e:
            st.error(f"Error switching models: {str(e)}")

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
            
            # Get the system message for the selected personality
            system_message = get_system_message(personality)
            
            # Update the conversation prompt template with the new personality
            st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            st.success(f"Successfully switched to {personality} personality!")
        except Exception as e:
            st.error(f"Error switching personality: {str(e)}")
    
    # Chat history management
    st.subheader("Chat History Management")
    
    # Save current chat button
    if st.button("Save Current Chat"):
        saved_file = save_chat_history()
        if saved_file:
            st.success(f"Chat saved to {saved_file}")
    
    # Load previous chat option
    history_files = [f for f in os.listdir('.') if f.startswith('chat_history_') and f.endswith('.json')]
    if history_files:
        selected_file = st.selectbox("Load saved chat:", history_files)
        if st.button("Load Selected Chat"):
            loaded_history = load_chat_history(selected_file)
            if loaded_history:
                st.session_state.chat_history = loaded_history
                st.success("Chat history loaded successfully!")
                st.experimental_rerun()

# Clear chat button at the bottom of the main area
if st.button("Clear Chat History"):
    # Reset the chat history
    st.session_state.chat_history = []
    
    # Reinitialize the conversation with current settings
    st.session_state.conversation = initialize_conversation(
        model_name=st.session_state.model_name,
        personality=st.session_state.personality
    )
    
    st.success(f"Chat cleared. Using {st.session_state.model_name} with {st.session_state.personality} personality.")
    st.rerun()

# Information about model usage
st.info("💡 If you encounter quota errors, use gpt-3.5-turbo which is more affordable.")
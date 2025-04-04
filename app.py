import streamlit as st  # Main framework for creating web apps
import os  # For accessing environment variables and file operations
from dotenv import load_dotenv  # For loading environment variables from .env file
from langchain_community.chat_models import ChatOpenAI  # LLM model wrapper
from langchain.memory import ConversationBufferMemory  # For storing conversation history
from langchain.chains import ConversationChain  # For creating conversation flow
from langchain.schema import HumanMessage, AIMessage  # Message schema types
import json  # For serializing/deserializing chat history
import time  # For timestamps and delays
from datetime import datetime  # For detailed timestamps

# Load environment variables from .env file (including OPENAI_API_KEY)
load_dotenv()

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="AI Chatbot",  # Browser tab title
    layout="centered"  # Centered layout for better readability
)

# Helper function that returns appropriate system messages based on the selected personality
def get_system_message(personality):
    """
    Return a system message based on the selected personality
    Each personality has a unique prompt that guides the AI's behavior
    """
    personalities = {
        "Helpful Assistant": "You are a helpful AI assistant. You provide clear and concise answers to user questions.",
        "Friendly Teacher": "You are a friendly teacher. Explain concepts in simple terms and use examples to help users understand.",
        "Creative Writer": "You are a creative writing assistant. Help users with creative and engaging content. Be imaginative and inspiring.",
        "Technical Expert": "You are a technical expert. Provide detailed and accurate technical information. Use precise terminology."
    }
    # Return the corresponding system message or default if personality not found
    return personalities.get(personality, "You are a helpful AI assistant.")

def save_chat_history():
    """
    Save the current chat history to a local file
    Creates a JSON file with timestamped messages
    """
    if "chat_history" in st.session_state and st.session_state.chat_history:
        # Convert chat messages to serializable format
        serializable_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                # Format human messages
                serializable_history.append({
                    "role": "human", 
                    "content": msg.content, 
                    "timestamp": datetime.now().isoformat()
                })
            elif isinstance(msg, AIMessage):
                # Format AI messages
                serializable_history.append({
                    "role": "ai", 
                    "content": msg.content, 
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate a unique filename using current timestamp
        filename = f"chat_history_{int(time.time())}.json"
        try:
            # Write the serialized history to a JSON file
            with open(filename, "w") as f:
                json.dump(serializable_history, f)
            # Return the filename for reference
            return filename
        except Exception as e:
            # Handle any errors during saving
            st.warning(f"Could not save chat history: {str(e)}")
            return None
    return None

def load_chat_history(filename):
    """
    Load chat history from a JSON file
    Converts stored messages back to LangChain message objects
    """
    try:
        # Read the JSON file
        with open(filename, "r") as f:
            serialized_history = json.load(f)
        
        # Convert serialized messages back to LangChain message objects
        history = []
        for msg in serialized_history:
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history.append(AIMessage(content=msg["content"]))
        
        return history
    except Exception as e:
        # Handle any errors during loading
        st.warning(f"Could not load chat history: {str(e)}")
        return []

def get_ai_response(user_input, max_retries=3, retry_delay=2):
    """
    Get AI response with retry logic for API errors
    Implements backoff strategy and fallback to cheaper model
    
    Args:
        user_input: The text input from the user
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries
        
    Returns:
        The AI's response text
    """
    for attempt in range(max_retries):
        try:
            # Try to get a response from the current model
            return st.session_state.conversation.predict(input=user_input)
        except Exception as e:
            error_message = str(e)
            # If not the last attempt, try again
            if attempt < max_retries - 1:
                # Wait before retrying to avoid overwhelming the API
                time.sleep(retry_delay)
                # If it's a quota error, try fallback to a cheaper model
                if "quota" in error_message.lower() or "429" in error_message:
                    try:
                        # Temporarily switch to more affordable model
                        backup_llm = ChatOpenAI(
                            model_name="gpt-3.5-turbo",  # Cheaper, more available model
                            temperature=0.7,  # Control randomness (0.0 = deterministic, 1.0 = creative)
                            openai_api_key=os.getenv("OPENAI_API_KEY")
                        )
                        # Preserve the conversation memory
                        memory = st.session_state.conversation.memory
                        # Create temporary conversation with backup model
                        temp_conversation = ConversationChain(
                            llm=backup_llm,
                            memory=memory,
                            verbose=False
                        )
                        return temp_conversation.predict(input=user_input)
                    except:
                        # Continue to next retry attempt if backup also fails
                        continue
            else:
                # If all retries fail, reraise the exception
                raise

def initialize_conversation(model_name="gpt-3.5-turbo", personality="Helpful Assistant"):
    """
    Initialize or reinitialize the conversation with specified settings
    
    Args:
        model_name: The OpenAI model to use
        personality: The conversation personality to apply
        
    Returns:
        A configured ConversationChain instance
    """
    # Get API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()
    
    # Get system message for selected personality
    system_message = get_system_message(personality)
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,  # Balanced between deterministic and creative responses
        openai_api_key=api_key
    )

    # Create memory object to store conversation history
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create the conversation chain that connects the LLM with memory
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # Set to True for debugging
    )
    
    # Set the system message in the conversation prompt template
    # This influences how the AI responds based on personality
    conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
    
    return conversation

# -------------------- MAIN APPLICATION --------------------

# Set the main title and subtitle
st.title("AI Chatbot")
st.subheader("Built with streamlit, Langchain and GPT-4o")

# Initialize session state variables if they don't exist
# These persist across reruns of the app
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store message history

if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"  # Default to cheaper model

if "personality" not in st.session_state:
    st.session_state.personality = "Helpful Assistant"  # Default personality

# Initialize conversation if it doesn't exist in session state
if "conversation" not in st.session_state:
    try:
        st.session_state.conversation = initialize_conversation(
            model_name=st.session_state.model_name,
            personality=st.session_state.personality
        )
    except Exception as e:
        st.error(f"Error initializing the chatbot: {str(e)}")
        st.stop()  # Stop execution if initialization fails

# Display chat history
# This renders all previous messages in the chat
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        # Display user messages with user avatar
        with st.chat_message("user"):
            st.write(message.content)
    else:
        # Display AI messages with assistant avatar
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input box at the bottom of the chat area
user_input = st.chat_input("Type your message...")

# Process user input and generate response
if user_input:
    # Add the user's message to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display the user's message in the UI
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        try:
            # Get AI response with retry logic
            with st.spinner("Pondering..."):  # Show a spinner while processing
                response = get_ai_response(user_input)
                st.write(response)  # Display the response
            
            # Add response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Auto-save chat history after each successful response
            save_chat_history()
        
        except Exception as e:
            error_message = str(e)
            
            # Rule-based fallback response if API has issues
            if "quota" in error_message.lower() or "429" in error_message or "insufficient_quota" in error_message.lower():
                st.warning("Using fallback mode due to API limitations")
                
                # Simple rule-based responses based on keywords in user input
                # This allows basic functionality even when API access fails
                fallback_responses = {
                    "hello": "Hello! I'm currently running in fallback mode. How can I help you?",
                    "hi": "Hello! I'm currently running in fallback mode. How can I help you?",
                    "how are you": "I'm doing well, though I'm currently operating in a limited fallback mode.",
                    "help": "I'm in fallback mode right now, but I'd be happy to help as best I can.",
                    "thank": "You're welcome! Happy to assist even in fallback mode."
                }
                
                # Default fallback message
                response = "I'm in fallback mode due to API limitations. I can only provide basic responses at the moment."
                
                # Check for keywords in user input to provide better responses
                for keyword, canned_response in fallback_responses.items():
                    if keyword in user_input.lower():
                        response = canned_response
                        break
                
                # Special handling for question patterns
                if "what is" in user_input.lower() or "who is" in user_input.lower() or "how to" in user_input.lower():
                    response = f"That's an interesting question about '{user_input}'. When I'm back to full functionality, I'll provide a more detailed answer."
                elif "?" in user_input:
                    response = "That's a good question. Unfortunately, I'm currently in fallback mode and can't provide a complete answer."
                
                # Display the fallback response
                st.write(response)
                
                # Add fallback response to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
            else:
                # For other errors, display the error message
                st.error(f"An error occurred: {error_message}")

# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.title("Options")
    
    # Model selection dropdown
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Select OpenAI model:",
        ["gpt-3.5-turbo", "gpt-4o"],  # Available models
        index=0 if st.session_state.model_name == "gpt-3.5-turbo" else 1,  # Set current selection
        help="GPT-3.5 Turbo uses less API quota than GPT-4o"  # Helpful tooltip
    )
    
    # Display currently active model
    st.write(f"Currently using model: {st.session_state.model_name}")

    # Button to apply model change
    if st.button("Apply Model Change"):
        try:
            # Update session state with new model
            st.session_state.model_name = model_name
            
            # Create new conversation with the selected model but keep the chat history
            memory = st.session_state.conversation.memory  # Preserve existing memory
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
            
            # Reapply system message based on current personality
            system_message = get_system_message(st.session_state.personality)
            st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            # Show success message
            st.success(f"Successfully switched to {model_name}")
        except Exception as e:
            st.error(f"Error switching models: {str(e)}")

    # Personality selector dropdown
    st.subheader("Chatbot Personality")
    personality = st.selectbox(
        "Select a personality for your chatbot:",
        ["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"],
        index=["Helpful Assistant", "Friendly Teacher", "Creative Writer", "Technical Expert"].index(st.session_state.personality)
    )

    # Button to apply personality change
    if st.button("Apply Personality Change"):
        try:
            # Update session state with new personality
            st.session_state.personality = personality
            
            # Get the system message for the selected personality
            system_message = get_system_message(personality)
            
            # Update the conversation prompt template with the new personality
            st.session_state.conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
            
            # Show success message
            st.success(f"Successfully switched to {personality} personality!")
        except Exception as e:
            st.error(f"Error switching personality: {str(e)}")
    
    # Chat history management section
    st.subheader("Chat History Management")
    
    # Button to save current chat
    if st.button("Save Current Chat"):
        saved_file = save_chat_history()
        if saved_file:
            st.success(f"Chat saved to {saved_file}")
    
    # Dropdown to load previous chats
    history_files = [f for f in os.listdir('.') if f.startswith('chat_history_') and f.endswith('.json')]
    if history_files:
        selected_file = st.selectbox("Load saved chat:", history_files)
        if st.button("Load Selected Chat"):
            loaded_history = load_chat_history(selected_file)
            if loaded_history:
                st.session_state.chat_history = loaded_history
                st.success("Chat history loaded successfully!")
                st.experimental_rerun()  # Refresh UI to show loaded chat

# Clear chat button at the bottom of the main area
if st.button("Clear Chat History"):
    # Reset the chat history
    st.session_state.chat_history = []
    
    # Reinitialize the conversation with current settings
    st.session_state.conversation = initialize_conversation(
        model_name=st.session_state.model_name,
        personality=st.session_state.personality
    )
    
    # Show success message
    st.success(f"Chat cleared. Using {st.session_state.model_name} with {st.session_state.personality} personality.")
    st.rerun()  # Refresh the UI

# Information tip about model usage
st.info("💡 If you encounter quota errors, use gpt-3.5-turbo which is more affordable.")
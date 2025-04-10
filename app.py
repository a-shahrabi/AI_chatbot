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

# New imports for knowledge base and RAG
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Configure the page layout and title
st.set_page_config(
    page_title="AI Chatbot with Knowledge Base",
    layout="centered"
)

# Helper function that returns appropriate system messages based on the selected personality
def get_system_message(personality):
    """Return a system message based on the selected personality"""
    system_messages = {
        "Helpful Assistant": "You are a helpful AI assistant. You provide clear and concise answers to user questions.",
        "Friendly Teacher": "You are a friendly teacher. Explain concepts in simple terms and use examples to help users understand.",
        "Creative Writer": "You are a creative writing assistant. Help users with creative and engaging content. Be imaginative and inspiring.",
        "Technical Expert": "You are a technical expert. Provide detailed and accurate technical information. Use precise terminology."
    }
    return system_messages.get(personality, "You are a helpful AI assistant.")

def save_chat_history():
    """Save the current chat history to a local file"""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        # Convert chat messages to serializable format
        serializable_history = []
        current_time = datetime.now().isoformat()
        
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                serializable_history.append({"role": "human", "content": msg.content, "timestamp": current_time})
            elif isinstance(msg, AIMessage):
                serializable_history.append({"role": "ai", "content": msg.content, "timestamp": current_time})
        
        # Generates a unique filename for the chat history using the current timestamp
        filename = f"chat_history_{int(time.time())}.json"
        try:
            # Open the file in write mode and save the chat history in JSON format
            with open(filename, "w") as f:
                json.dump(serializable_history, f, indent=2)
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

# New function to handle document uploading and processing
def process_documents(uploaded_files):
    """Process uploaded documents and create a vector database"""
    if not uploaded_files:
        return None
    
    # Create a directory for storing uploaded documents if it doesn't exist
    if not os.path.exists("uploaded_docs"):
        os.makedirs("uploaded_docs")
    
    # Process each uploaded file
    documents = []
    for file in uploaded_files:
        # Save the file temporarily
        file_path = os.path.join("uploaded_docs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Load the document based on file type
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file.name.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file.name}. Skipping.")
                continue
            
            # Load the document
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    
    if not documents:
        return None
    
    # Split the documents into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store them in a vector database
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        return None

# Initialize the RAG-enhanced conversation
def initialize_rag_conversation(model_name="gpt-3.5-turbo", personality="Helpful Assistant", vectorstore=None):
    """Initialize or reinitialize the conversation with RAG capabilities if vectorstore is provided"""
    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            openai_api_key=api_key
        )

        # If we have a vectorstore, create a RAG chain
        if vectorstore is not None:
            # Create a retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
            )
            
            # Create a template for using context from documents
            template = """
            You are an AI assistant with the personality of a {personality}.
            
            Use the following pieces of context to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Current conversation:
            {history}
            Human: {input}
            AI:
            """
            
            prompt = PromptTemplate(
                input_variables=["personality", "context", "history", "input"],
                template=template
            )
            
            # Create a QA chain with memory
            memory = ConversationBufferMemory(return_messages=True)
            
            # Create a customized RAG conversation chain
            from langchain.chains import LLMChain
            
            class RAGConversationChain(LLMChain):
                def __init__(self, llm, prompt, retriever, memory=None, verbose=False):
                    super().__init__(llm=llm, prompt=prompt, verbose=verbose, memory=memory)
                    self.retriever = retriever
                
                def predict(self, input):
                    # Retrieve relevant documents
                    docs = self.retriever.get_relevant_documents(input)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Get conversation history
                    history = ""
                    if self.memory:
                        history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" 
                                           for msg in self.memory.chat_memory.messages])
                    
                    # Run the chain
                    return self({"personality": st.session_state.personality, 
                                "context": context, 
                                "history": history, 
                                "input": input})["text"]
            
            conversation = RAGConversationChain(
                llm=llm,
                prompt=prompt,
                retriever=retriever,
                memory=memory,
                verbose=False
            )
            
        else:
            # If no vectorstore, create a regular conversation chain
            # Create memory object and conversation chain
            memory = ConversationBufferMemory(return_messages=True)
            conversation = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=False
            )
            
            # Get system message for selected personality
            system_message = get_system_message(personality)
            # Set the system message in the conversation prompt template
            conversation.prompt.template = f"{system_message}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI:"
        
        return conversation
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

def get_rule_based_response(user_input):
    """Generate a simple rule-based response when API is unavailable"""
    # Check for common patterns in user input
    user_input_lower = user_input.lower()
    
    if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey"]):
        return "Hello! I'm currently running in fallback mode. How can I help you?"
    elif "how are you" in user_input_lower:
        return "I'm doing well, though I'm currently operating in a limited fallback mode."
    elif "help" in user_input_lower:
        return "I'm in fallback mode right now, but I'd be happy to help as best I can."
    elif any(thanks in user_input_lower for thanks in ["thank", "thanks", "appreciate"]):
        return "You're welcome! Happy to assist even in fallback mode."
    # Question patterns
    elif any(q_word in user_input_lower for q_word in ["what is", "who is", "how to", "where", "when", "why"]):
        return f"That's an interesting question about '{user_input}'. When I'm back to full functionality, I'll provide a more detailed answer."
    elif "?" in user_input:
        return "That's a good question. Unfortunately, I'm currently in fallback mode and can't provide a complete answer."
    
    # Default fallback message
    return "I'm in fallback mode due to API limitations. I can only provide basic responses at the moment."

# Set up the main UI
def setup_ui():
    # Set the main title of the application
    st.title("AI Chatbot with Knowledge Base")
    # Set the subtitle describing the technologies used
    st.subheader("Built with Streamlit, Langchain and RAG")

def handle_sidebar():
    """Handle all sidebar functionality"""
    with st.sidebar:
        st.title("Options")
        
        # Knowledge Base Upload Section
        st.subheader("Knowledge Base")
        uploaded_files = st.file_uploader("Upload documents for the bot to reference", 
                                        accept_multiple_files=True,
                                        type=["pdf", "txt", "csv", "docx"])
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                vectorstore = process_documents(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                    
                    # Reinitialize conversation with the new vectorstore
                    st.session_state.conversation = initialize_rag_conversation(
                        model_name=st.session_state.model_name,
                        personality=st.session_state.personality,
                        vectorstore=vectorstore
                    )
        
        # Show knowledge base status
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            st.info("Knowledge base is active. The bot will use your documents to answer questions.")
            if st.button("Clear Knowledge Base"):
                st.session_state.vectorstore = None
                # Reinitialize conversation without vectorstore
                st.session_state.conversation = initialize_rag_conversation(
                    model_name=st.session_state.model_name,
                    personality=st.session_state.personality
                )
                st.success("Knowledge base cleared.")
        
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
                    # Get the existing vectorstore if available
                    vectorstore = st.session_state.vectorstore if "vectorstore" in st.session_state else None
                    
                    # Update the conversation with new model
                    st.session_state.conversation = initialize_rag_conversation(
                        model_name=model_name,
                        personality=st.session_state.personality,
                        vectorstore=vectorstore
                    )
                    
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
                
                # Only update if not in fallback mode
                if not st.session_state.fallback_mode:
                    # Get the existing vectorstore if available
                    vectorstore = st.session_state.vectorstore if "vectorstore" in st.session_state else None
                    
                    # Reinitialize the conversation with the new personality
                    st.session_state.conversation = initialize_rag_conversation(
                        model_name=st.session_state.model_name,
                        personality=personality,
                        vectorstore=vectorstore
                    )
                
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

def process_user_message(user_input):
    """Process the user message and generate a response"""
    # Add the user's message to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display the user's message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        # If in fallback mode, use rule-based responses
        if st.session_state.fallback_mode:
            response = get_rule_based_response(user_input)
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            try:
                # Show spinner while processing
                with st.spinner("Thinking..."):
                    # Try to get response from LLM
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

# Main function to run the app
def main():
    # Setup the UI
    setup_ui()
    
    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo"  # Default to cheaper model

    if "personality" not in st.session_state:
        st.session_state.personality = "Helpful Assistant"

    if "fallback_mode" not in st.session_state:
        st.session_state.fallback_mode = False
        
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Initialize conversation if it doesn't exist
    if "conversation" not in st.session_state:
        try:
            st.session_state.conversation = initialize_rag_conversation(
                model_name=st.session_state.model_name,
                personality=st.session_state.personality
            )
            if not st.session_state.conversation:
                st.session_state.fallback_mode = True
        except Exception as e:
            st.error(f"Error initializing the chatbot: {str(e)}")
            # Set fallback mode to true if there's an error on startup
            st.session_state.fallback_mode = True

    # Handle sidebar functionality
    handle_sidebar()

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
        process_user_message(user_input)

    # Clear chat button
    if st.button("Clear Chat History"):
        # Reset the chat history and fallback mode
        st.session_state.chat_history = []
        st.session_state.fallback_mode = False
        
        # Try to reinitialize the conversation with current settings
        try:
            # Get the existing vectorstore if available
            vectorstore = st.session_state.vectorstore if "vectorstore" in st.session_state else None
            
            st.session_state.conversation = initialize_rag_conversation(
                model_name=st.session_state.model_name,
                personality=st.session_state.personality,
                vectorstore=vectorstore
            )
            
            kb_status = "with Knowledge Base" if vectorstore else "without Knowledge Base"
            st.success(f"Chat cleared. Using {st.session_state.model_name} with {st.session_state.personality} personality {kb_status}.")
        except Exception as e:
            st.error(f"Error reinitializing conversation: {str(e)}")
            st.session_state.fallback_mode = True
            st.warning("Started in fallback mode due to API issues.")
        
        st.rerun()

    # Information about current mode
    if st.session_state.fallback_mode:
        st.warning("⚠️ Running in fallback mode due to API limitations. Responses are rule-based and limited.")
    elif "vectorstore" in st.session_state and st.session_state.vectorstore:
        st.success("🧠 Knowledge base is active. The bot can reference your uploaded documents.")
    else:
        st.info("💡 Upload documents to the knowledge base to make the bot more helpful with specific information.")

# Run the app
if __name__ == "__main__":
    main()
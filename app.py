import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage


# Load environment variables from .env file (contains OpenAI API key)
load_dotenv()

# Configure the Streamlit page settings
st.set_page_config(
    page_title="AI Chatbot",  # Sets the browser tab title
    layout="centered"         # Centers the content for better readability
)

# Create a custom header with logo and title
col1, col2 = st.columns([1, 5])  # Create two columns with 1:5 ratio
with col1:
    st.markdown("# 🤖")  # Robot emoji as logo
with col2:
    st.title("AI Chatbot")  # Main application title
    st.markdown("<p style='color: #666666; margin-top: -10px;'>Built with Streamlit, LangChain and GPT-4o-mini</p>", unsafe_allow_html=True)  # Subtitle with custom styling

# Add a divider for visual separation between header and chat
st.divider()

# Initialize chat history in session state if it doesn't exist
# Session state persists data between reruns of the script
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add welcome message when chat first initializes
    welcome_message = AIMessage(content="👋 Hello! I'm your AI assistant. How can I help you today?")
    st.session_state.chat_history.append(welcome_message)

# Initialize the conversation chain if it doesn't exist in session state
if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

# Display chat history with custom styling
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid #4b8bbe;'>{message.content}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 8px; border-left: 5px solid #0066cc;'>{message.content}</div>", unsafe_allow_html=True)

# Take user input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response from conversation
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)
        
        
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)  # Adjust typing speed
            message_placeholder.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 8px; border-left: 5px solid #0066cc;'>{full_response}▌</div>", unsafe_allow_html=True)
        
        # Display final response
        message_placeholder.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 8px; border-left: 5px solid #0066cc;'>{response}</div>", unsafe_allow_html=True)
    
    
    st.session_state.chat_history.append(AIMessage(content=response))
    
    
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("👍", key=f"thumbs_up_{len(st.session_state.chat_history)}"):
            st.success("Thanks for the positive feedback!")
            
        if st.button("👎", key=f"thumbs_down_{len(st.session_state.chat_history)}"):
            st.error("Thanks for your feedback. We'll try to improve!")

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px">
        <h2 style="color:#0066cc;">AI Chat Assistant</h2>
        <p style="font-size:14px; color:#666666;">Your intelligent conversation partner</p>
    </div>
    """, unsafe_allow_html=True)
    st.title("Options")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        st.rerun()
    
    st.divider()
    
    # Export chat feature
    st.subheader("💾 Export Chat")
    
    if st.button("Export Conversation"):
        chat_export = ""
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                chat_export += f"User: {msg.content}\n\n"
            else:
                chat_export += f"AI: {msg.content}\n\n"
        
        # Create a download button
        st.download_button(
            label="Download Chat History",
            data=chat_export,
            file_name="ai_chat_history.txt",
            mime="text/plain"
        )
    
    st.divider()
    st.subheader("⚙️ Settings")
    
    # Temperature slider
    temp_value = st.slider(
        "Model Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Model selection
    model_option = st.selectbox(
        "Choose Model",
        options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        index=0,
        help="Select the model you want to use"
    )
    
    # Apply settings button
    if st.button("Apply Settings"):
        # Update the model with new settings
        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatOpenAI(
            model_name=model_option,
            temperature=temp_value,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        st.success(f"Settings updated! Using {model_option} with temperature {temp_value}")
        st.rerun()

    st.subheader("About")

    st.markdown("""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:15px">
            <h3 style="color:#0066cc; margin-bottom:10px;">Powered By:</h3>
            <ul style="list-style-type:none; padding-left:10px;">
                <li style="margin-bottom:8px;">
                    <span style="color:#0066cc; font-weight:bold;">⚡ Streamlit</span> 
                    <span style="color:#444444;">for the sleek interface</span>
                </li>
                <li style="margin-bottom:8px;">
                    <span style="color:#0066cc; font-weight:bold;">🔗 LangChain</span> 
                    <span style="color:#444444;">for conversation management</span>
                </li>
                <li style="margin-bottom:8px;">
                    <span style="color:#0066cc; font-weight:bold;">🧠 GPT-4o-mini</span> 
                    <span style="color:#444444;">for intelligent responses</span>
                </li>
                <li style="margin-bottom:8px;">
                    <span style="color:#0066cc; font-weight:bold;">💾 ConversationBufferMemory</span> 
                    <span style="color:#444444;">for conversation history</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
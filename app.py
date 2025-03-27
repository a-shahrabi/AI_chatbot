import streamlit as st
import os
import time
import base64
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

# Take user input - both text and voice
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.chat_input("Type your message...")
    
with col2:
    # Add voice input functionality
    # HTML and JavaScript for the voice input button
    voice_input_js = """
    <script>
    const startButton = document.getElementById('startButton');
    let mediaRecorder;
    let audioChunks = [];

    startButton.addEventListener('click', () => {
        // Update button state
        if (startButton.textContent === '🎤') {
            // Start recording
            startButton.textContent = '⏹️';
            startButton.classList.add('recording');
            document.getElementById('status').textContent = 'Recording...';
            
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64data = reader.result.split(',')[1];
                            // Send to Streamlit
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: base64data
                            }, '*');
                        };
                    };
                    
                    mediaRecorder.start();
                })
                .catch(error => {
                    console.error('Error accessing microphone:', error);
                    document.getElementById('status').textContent = 'Microphone access denied';
                    startButton.textContent = '🎤';
                    startButton.classList.remove('recording');
                });
        } else {
            // Stop recording
            startButton.textContent = '🎤';
            startButton.classList.remove('recording');
            document.getElementById('status').textContent = 'Processing...';
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                audioChunks = [];
                
                // Stop all tracks
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
    });
    </script>
    """
    
    # CSS for styling the voice input button
    voice_input_css = """
    <style>
    #voiceInputContainer {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 10px;
    }
    #startButton {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #f0f2f6;
        border: 2px solid #0066cc;
        color: #0066cc;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s;
    }
    #startButton:hover {
        background-color: #e6f3ff;
    }
    #startButton.recording {
        background-color: #ff6b6b;
        color: white;
        border-color: #ff6b6b;
        animation: pulse 1.5s infinite;
    }
    #status {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
        height: 15px;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>
    """
    
    # HTML for the voice input button
    voice_input_html = f"""
    {voice_input_css}
    <div id="voiceInputContainer">
        <button id="startButton">🎤</button>
        <div id="status"></div>
    </div>
    {voice_input_js}
    """
    
    # Display the voice input button
    st.markdown(voice_input_html, unsafe_allow_html=True)
    
    # Create a component to receive the voice data
    voice_data = st.empty()
    
    # Custom component to handle the voice data
    voice_component = st.text_input("Voice Input", key="voice_input", label_visibility="collapsed")
    
    if voice_component and len(voice_component) > 20:  # Check if it looks like base64 data
        try:
            # Here you would process the voice data
            # This is where you'd integrate with a speech-to-text API
            # For now, we'll simulate it with a placeholder message
            user_input = "I used voice input! (Speech-to-text would convert actual speech here)"
            st.session_state.speech_processed = True
        except Exception as e:
            st.error(f"Error processing voice: {e}")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; border-left: 5px solid #4b8bbe;'>{user_input}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response from conversation
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)
        
        # Simulate typing effect
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)  # Adjust typing speed
            message_placeholder.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 8px; border-left: 5px solid #0066cc;'>{full_response}▌</div>", unsafe_allow_html=True)
        
        # Display final response
        message_placeholder.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 8px; border-left: 5px solid #0066cc;'>{response}</div>", unsafe_allow_html=True)
    
    # Append to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
    
    # Add a feedback system
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
    
    # Add information about voice feature
    with st.expander("🔊 Voice Input Instructions"):
        st.markdown("""
        **How to use voice input:**
        1. Click the microphone button in the chat input area
        2. Allow microphone access when prompted
        3. Speak clearly into your microphone
        4. Click the stop button (⏹️) when finished
        5. Wait for your speech to be processed
        
        **Note:** For this demo, actual speech-to-text conversion is simulated. 
        In a production app, you would integrate with services like:
        - Google Speech-to-Text
        - Amazon Transcribe
        - Microsoft Azure Speech Services
        """)
    
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
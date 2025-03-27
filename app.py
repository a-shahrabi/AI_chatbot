import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage


load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    layout="centered"
)

st.title("AI Chatbot")
st.subheader("Built with streamlit, Langchain and GPT-4o-mini")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    with st.chat_message("assistant"):
        with st.spinner("Pondering..."):
            response = st.session_state.conversation.predict(input=user_input)
            st.write(response)
    
    # Append to chat history
    st.session_state.chat_history.append(AIMessage(content=response))

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
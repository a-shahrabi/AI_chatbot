# AI Chatbot

A sophisticated AI chatbot built with Streamlit, LangChain, and OpenAI's GPT models.

![Chatbot Preview](https://raw.githubusercontent.com/username/ai-chatbot/main/preview.png)

## Features

- **Intelligent Conversations**: Powered by OpenAI's GPT models
- **Voice Input**: Speak to your AI assistant instead of typing
- **Typing Animation**: Realistic typing effect for more natural interaction
- **Customizable Settings**: Adjust temperature and choose between different models
- **Conversation Memory**: Chatbot remembers the context of your conversation
- **Export Functionality**: Download your chat history as a text file
- **User Feedback System**: Rate responses with thumbs up/down
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ai-chatbot.git
   cd ai-chatbot
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the app:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Start chatting with your AI assistant!

## Voice Input Usage

1. Click the microphone button in the chat input area
2. Allow microphone access when prompted by your browser
3. Speak clearly into your microphone
4. Click the stop button (⏹️) when finished
5. Wait for your speech to be processed

## Customization

### Model Selection

You can choose between different OpenAI models:
- GPT-4o-mini (default)
- GPT-3.5-turbo
- GPT-4o (requires higher API usage limits)

### Temperature Setting

Adjust the temperature slider to control the randomness of responses:
- Lower values (0.0-0.3): More focused, deterministic responses
- Medium values (0.4-0.7): Balanced creativity and focus
- Higher values (0.8-1.0): More creative, varied responses

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API key
- dotenv

See `requirements.txt` for complete dependencies.

## Files

- `app.py`: Main application code
- `.env`: Environment variables (API keys)
- `requirements.txt`: Required Python packages

## Developer Notes

- The voice input feature uses the browser's MediaRecorder API
- For a production environment, connect to a speech-to-text API like Google's Speech-to-Text, Whisper API, or Amazon Transcribe

## License

MIT License

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the wonderful web framework
- [LangChain](https://langchain.com/) for the conversation management
- [OpenAI](https://openai.com/) for the GPT models

---

Created with by Ardavan
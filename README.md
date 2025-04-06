# AI Chatbot

A sophisticated AI chatbot built with Streamlit, LangChain, and OpenAI's GPT models, featuring knowledge base integration.

![Chatbot Preview](https://raw.githubusercontent.com/username/ai-chatbot/main/preview.png)

## Features

- **Intelligent Conversations**: Powered by OpenAI's GPT models
- **Knowledge Base Integration**: Upload documents for the bot to reference when answering questions
- **RAG (Retrieval-Augmented Generation)**: Provides more accurate, grounded responses based on your documents
- **Multiple Document Formats**: Support for PDF, TXT, CSV, and DOCX files
- **Adjustable Personalities**: Choose between Helpful Assistant, Friendly Teacher, Creative Writer, or Technical Expert
- **Multiple Models**: Select between GPT-3.5-turbo and GPT-4o
- **Conversation Memory**: Chatbot remembers the context of your conversation
- **Chat History Management**: Save and load previous conversations
- **Fallback Mode**: Continues to function with basic responses when API is unavailable
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

## Knowledge Base Usage

1. In the sidebar, under "Knowledge Base," click "Browse files" to select documents
2. Upload PDFs, TXT files, CSVs, or DOCX documents that contain information you want the bot to reference
3. Click "Process Documents" to analyze and embed the documents
4. Once processed, the bot will automatically use these documents to provide more accurate answers
5. You can clear the knowledge base at any time using the "Clear Knowledge Base" button

## Customization

### Model Selection

You can choose between different OpenAI models:
- GPT-3.5-turbo (default, uses less API quota)
- GPT-4o (more powerful but uses more API quota)

### Personality Selection

Adjust the chatbot's personality:
- **Helpful Assistant**: Provides clear and concise answers
- **Friendly Teacher**: Explains concepts in simple terms with examples
- **Creative Writer**: Helps with creative and engaging content
- **Technical Expert**: Provides detailed technical information with precise terminology

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API key
- dotenv
- FAISS for vector storage
- Document processing libraries (pypdf, docx2txt)

See `requirements.txt` for complete dependencies.

## Additional Dependencies

For knowledge base functionality, install:
```
pip install langchain-openai faiss-cpu pypdf docx2txt
```

## Files

- `app.py`: Main application code
- `.env`: Environment variables (API keys)
- `requirements.txt`: Required Python packages
- `uploaded_docs/`: Directory where uploaded documents are temporarily stored
- `chat_history_*.json`: Saved chat history files

## Developer Notes

- The RAG implementation uses OpenAI Embeddings and FAISS for vector search
- Documents are split into chunks of 1000 characters with 200 character overlap for better retrieval
- The top 3 most relevant document chunks are retrieved for each user query

## License

MIT License

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the wonderful web framework
- [LangChain](https://langchain.com/) for the conversation management and RAG capabilities
- [OpenAI](https://openai.com/) for the GPT models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector similarity search

---

Created with ❤️ by Ardavan
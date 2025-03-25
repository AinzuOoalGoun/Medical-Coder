# Medical Coding Assistant

A ChatGPT-like interface for medical coding assistance using Streamlit, Gemini, and LangGraph.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the required environment variables:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   ```

## Running the Application

Run the Streamlit app:
```
streamlit run streamlit_app.py
```

This will start the web interface on http://localhost:8501

## Features

- ChatGPT-like user interface
- Medical coding assistance for ICD codes
- Chat history persistence
- Thread-based conversations

## Usage

1. Type your medical coding query in the text input field
2. Click "Send" or press Enter to submit
3. The AI will respond with relevant ICD-10 codes
4. You can click on example queries in the sidebar for quick access
5. Your conversation history is maintained during the session
6. View real-time metrics of your coding queries
7. You can reset the conversation using the "Reset Conversation" button in the sidebar

## Examples of Queries

- "What is the ICD-10 code for hypertension?"
- "What are the ICD-10 codes for type 2 diabetes with complications?"

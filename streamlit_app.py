import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore  
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool

# Load environment variables
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGSMITH_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

# If no API keys in environment variables, allow user input
with st.sidebar:
    st.title("⚠️ Debug Mode (For Developers)")
    
    # Add debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False, help="Show all agent messages including tool calls")
    
    # Add advanced debugging options
    if debug_mode:
        st.session_state["verbose_debug"] = st.checkbox("Verbose Debug", value=False, help="Show even more detailed debugging information")
    else:
        st.session_state["verbose_debug"] = False
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This Medical Coding Assistant helps identify:
    - ICD-10 codes for diagnoses
    - CPT codes for procedures
    
    Simply describe the medical condition or procedure in the chat.
    """)
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your Medical Coding Assistant. How can I help you today?"}]
        st.session_state["thread_id"] = f"thread-{hash(str(st.session_state))}"
        st.rerun()

# Initialize the model
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True,
        google_api_key=GOOGLE_API_KEY
    )

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    if not GOOGLE_API_KEY:
        st.error("Google API Key is required for embeddings.")
        st.stop()
        
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

# Initialize Pinecone and retrievers
@st.cache_resource
def get_retrievers():
    if not PINECONE_API_KEY:
        st.error("Pinecone API Key is required for vector search.")
        st.stop()
        
    embeddings = get_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("medical-coding-index")
    
    text_field = "text"
    vector_store_ICD = PineconeVectorStore(  
        index, embeddings, text_field, namespace="ICD10"
    )
    
    vector_store_CPT = PineconeVectorStore(  
        index, embeddings, text_field, namespace="CPT"
    )
    
    retriever_ICD = vector_store_ICD.as_retriever(search_kwargs={"k": 4})
    retriever_CPT = vector_store_CPT.as_retriever(search_kwargs={"k": 4})
    
    retriever_ICD_tool = create_retriever_tool(
        retriever_ICD,
        "retrieve_ICD_codes",
        "Search and return information about ICD codes for a given condition/disease.",
    )
    
    retriever_CPT_tool = create_retriever_tool(
        retriever_CPT,
        "retrieve_CPT_codes",
        "Search and return information about CPT codes for a given procedure.",
    )
    
    return [retriever_ICD_tool, retriever_CPT_tool]

# Initialize the agent
@st.cache_resource
def get_agent():
    tools = get_retrievers()
    model = get_model()
    
    template = """
                You are a medical coding assistant designed to accurately identify ICD-10 codes with descriptions from clinical descriptions.
                
                Your primary task is to search and provide accurate medical codes based on the user's query.
                
                IMPORTANT INSTRUCTIONS:
                1. Always use the appropriate retrieval tool to look up codes.
                2. If you find relevant codes, clearly present them to the user.
                3. You must ALWAYS provide a Final Answer after using tools.
                4. Keep your responses focused on providing accurate medical coding information.
                5. Format your Final Answer clearly and directly, including the codes and their descriptions.
                6. Even for complex clinical scenarios or when you can't find exact matches, provide your best recommendation.
                7. DO NOT start your response with phrases like "You asked" or "Your query".
                8. suggest some relevant codes for the user's query.
                9. markdown the response.
                
    TOOLS:
    ------

    Assistant has access to the following tools:

    {tools}
    """
    
    memory = MemorySaver()
    # Create the agent with explicit settings for better response handling
    return create_react_agent(
        model, 
        tools, 
        prompt=template, 
        checkpointer=memory,
    )

# Helper function to clean response format
def clean_response(response_text):
    # Remove any tool call artifacts or formatting issues
    # Look for common patterns in the full response that indicate tool calls
    if "Action:" in response_text or "Action Input:" in response_text:
        # Try to extract just the final answer
        parts = response_text.split("Final Answer: ")
        if len(parts) > 1:
            return parts[1].strip()
    
    # Remove repeated user query if present
    lines = response_text.split('\n')
    if len(lines) > 1 and lines[0].strip().endswith('?') or lines[0].startswith(lines[0].lower()):
        response_text = '\n'.join(lines[1:])
    
    # Remove query echo at the beginning of the response (common pattern)
    for query_starter in ["your query", "you asked", "your question", "what is my name"]:
        if response_text.lower().startswith(query_starter):
            # Find first sentence break after the echo
            for i, char in enumerate(response_text):
                if i > len(query_starter) and char in ['.', '!', '?', '\n']:
                    response_text = response_text[i+1:].strip()
                    break
    
    # Clean up any formatting issues
    response_text = response_text.replace("  ", " ").strip()
    return response_text

# Helper function to extract content from message
def extract_message_content(message):
    """Extract content from different message formats."""
    if isinstance(message, tuple) and len(message) > 1:
        return message[1]
    elif hasattr(message, 'content'):
        return message.content
    elif isinstance(message, dict) and 'content' in message:
        return message['content']
    elif isinstance(message, str):
        return message
    return None

# Streamlit UI
st.title("💬 Medical Coding Assistant")
st.caption("🚀 Powered by Gemini and LangChain")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your Medical Coding Assistant. How can I help you today?"}]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"thread-{hash(str(st.session_state))}"
if "verbose_debug" not in st.session_state:
    st.session_state["verbose_debug"] = False

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input
if prompt := st.chat_input():
    # Check if keys are provided in sidebar
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        st.error("Please provide the required API keys in the sidebar to continue.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get agent response
    agent = get_agent()
    config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "1"}}
    inputs = {"messages": [("user", prompt)]}
    
    # In debug mode, show what we're sending to the agent
    if debug_mode and st.session_state["verbose_debug"]:
        st.write("Input to agent:", inputs)
        st.write("Config to agent:", config)
    
    # Create a placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        last_content = ""
        
        try:
            # Stream the response but do not display intermediate results
            all_messages = []
            observed_tools = []
            final_response = ""
            temp_response = ""
            
            # Process the stream with a spinner
            with st.spinner("Thinking..."):
                for s in agent.stream(inputs, config, stream_mode="values"):
                    # Debug: Print all responses from agent
                    if debug_mode:
                        st.write("Agent Response:", s)
                    
                    # Get the last message from the stream
                    if "messages" in s and s["messages"]:
                        message = s["messages"][-1]
                        all_messages.append(message)  # Keep track of all messages
                        
                        # Extract content regardless of message format
                        content = extract_message_content(message)
                        
                        if content:
                            # Capture content but don't display during processing
                            if debug_mode:
                                full_response += content + "\n\n"
                            else:
                                # In non-debug mode, only capture final content
                                if not any(marker in content for marker in ["Action:", "Tool:", "Thought:"]):
                                    # Only show final answer without the "Final Answer:" prefix
                                    if "Final Answer:" in content:
                                        parts = content.split("Final Answer:")
                                        if len(parts) > 1:
                                            content = parts[1].strip()
                                            final_response = content  # Replace with final answer
                                    else:
                                        # For non-tool, non-thought messages
                                        if not "Final Answer:" in final_response:
                                            final_response = content  # Use the latest AI message
            
            # Add additional debug output at the end of each response if in debug mode
            if debug_mode:
                st.write(f"Observed {len(all_messages)} total messages")
                if not full_response:
                        st.warning("No response content was captured")
            else:
                # Clean up the response for display
                if "Final Answer:" in final_response:
                    parts = final_response.split("Final Answer:")
                    if len(parts) > 1:
                        final_response = parts[1].strip()
                
                # Remove any tool usage or thought process that might remain
                lines = final_response.split("\n")
                filtered_lines = []
                for line in lines:
                    if not any(marker in line for marker in ["Action:", "Tool:", "Thought:", "🔍", "💭"]):
                        filtered_lines.append(line)
                final_response = "\n".join(filtered_lines)
                
                # Strip off any repeated user query
                user_query = prompt.lower().strip()
                if final_response.lower().startswith(user_query):
                    final_response = final_response[len(user_query):].strip()
                
                # Clean up any other formatting issues without replacing content
                final_response = final_response.replace("  ", " ").strip()
                
                # Show the final response once processing is complete
                message_placeholder.markdown(final_response or "No response was generated. Try enabling debug mode.")
                
            # Save the final response for chat history
            full_response = full_response if debug_mode else final_response
        except Exception as e:
            # Handle any errors during response generation
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.markdown(error_msg)
            if debug_mode:
                st.error(error_msg)
                import traceback
                st.code(traceback.format_exc(), language="python")
            full_response = "Sorry, I encountered an error processing your request."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

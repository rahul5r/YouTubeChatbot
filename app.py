import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api.proxies import WebshareProxyConfig
import re
import os

# Load environment variables
load_dotenv()


# Initialize components
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return embeddings, llm


embeddings, llm = load_rag_components()


# Function to extract video ID from YouTube URL
def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume it's already a video ID
    if len(url) == 11 and not '/' in url:
        return url
    
    return None


# Function to get transcript and build vector store
@st.cache_resource
def build_vector_store(video_id):
    try:
        ytt_api = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username="cynupwmo",
                        proxy_password="v3v3z8xkzmy1",
                    )
                )
        transcript_list = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([transcript])
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to fetch transcript: {e}")
        return None


# Streamlit app
st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")
st.title("🎥 YouTube Video Chatbot")

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL or Video ID:", placeholder="https://www.youtube.com/watch?v=...")
st.session_state.setdefault("messages", [])


if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if not video_id:
        st.error("Invalid YouTube URL or Video ID. Please check and try again.")
    else:
        # Build vector store
        with st.spinner("Loading transcript..."):
            vectorstore = build_vector_store(video_id)
        
        if vectorstore:
            st.success("✅ Transcript loaded successfully!")
            
            # Create two columns: left for video thumbnail, right for chat
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Video Preview")                
                # Embed video player
                st.video(f"https://www.youtube.com/watch?v={video_id}")
            
            with col2:
                st.subheader("Chat with Video")

                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Create a container for chat history so input is placed after it
                chat_container = st.container()

                with chat_container:
                    # Display chat messages
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                # Chat input stays OUTSIDE the container (so it's always below the messages)
                question = st.chat_input("Ask a question about the video...")

                if question:
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": question})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(question)

                    # Build RAG chain
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    prompt = PromptTemplate(
                        template=("You are a helpful YouTube RAG assistant. Answer ONLY from the given "
                                "transcript content. If transcript is insufficient, just say you don't know.\n\n"
                                "Context: {context}\n\nQuestion: {question}"),
                        input_variables=["context", "question"]
                    )
                    format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
                    rag_chain = (
                        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Get assistant response
                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                answer = rag_chain.invoke(question)
                                st.markdown(answer)

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        else:
            st.error("❌ Could not load transcript for this video. Make sure the video has captions/subtitles enabled.")

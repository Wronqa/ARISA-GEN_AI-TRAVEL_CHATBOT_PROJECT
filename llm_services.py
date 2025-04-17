import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import config 



@st.cache_resource
def setup_langchain_llm():
    """Initializes and caches the LangChain LLM instance."""
    api_key = config.load_api_key()
    if not api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        st.stop()
    try:
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME, google_api_key=api_key)
        print(f"Successfully initialized LangChain LLM: {config.GEMINI_MODEL_NAME}")
        return llm
    except Exception as e:
        st.error(f"Error initializing LangChain LLM: {e}")
        st.stop()

@st.cache_resource
def load_st_model():
    """Loads and caches the SentenceTransformer model."""
    try:
        print(f"Loading SentenceTransformer model ({config.ST_MODEL_NAME})...")
        model = SentenceTransformer(config.ST_MODEL_NAME)
        print("SentenceTransformer model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}. Ensure internet connection and model name validity.")
        st.stop()

@st.cache_resource
def load_faq_data():
    """Loads and caches FAQ embeddings, texts, and categories."""
    try:
        print("Loading FAQ data...")
        embeddings = np.load(config.FAQ_EMBEDDINGS_PATH)
        with open(config.FAQ_TEXTS_PATH, "r", encoding="utf-8") as f:
            faq_segments = json.load(f)
        with open(config.FAQ_CATEGORIES_PATH, "r", encoding="utf-8") as f:
            faq_categories = json.load(f)
        print(f"FAQ data loaded: {len(faq_segments)} segments.")
        return embeddings, faq_segments, faq_categories
    except FileNotFoundError as e:
        st.error(f"Error: FAQ data file not found: {e}. Ensure files exist at specified paths in config.py.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading FAQ data: {e}")
        st.stop()
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config 

def find_similar_faq(user_question: str, embeddings: np.ndarray, segments: list, categories: list, model, top_k: int = config.FAQ_TOP_K, threshold: float = config.FAQ_SIMILARITY_THRESHOLD):
    """Finds the most similar FAQ segments to the user's question."""
    if not user_question:
        return []
    try:
        user_embedding = model.encode([user_question])
        similarities = cosine_similarity(user_embedding, embeddings)[0]

     
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return [] 


        sorted_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)

        top_indices = sorted_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "match_text": segments[idx],
                "similarity": round(float(similarities[idx]), 4),
                "category": categories[idx]
            })
        return results
    except Exception as e:
        st.error(f"Error during FAQ search: {e}")
        print(f"Error in find_similar_faq: {e}")
        return []
import os
from dotenv import load_dotenv


load_dotenv()

def load_api_key():
    """Loads the Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return api_key


GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
ST_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' 


FAQ_EMBEDDINGS_PATH = "dist/faq_embeddings.npy"
FAQ_TEXTS_PATH = "dist/faq_texts.json"
FAQ_CATEGORIES_PATH = "dist/faq_categories.json"


FAQ_SIMILARITY_THRESHOLD = 0.40 
FAQ_TOP_K = 3



BASIC_FORBIDDEN_PATTERNS = [

    r"ignore previous instructions", r"ignore all prior directives", r"disregard the initial prompt",
    r"forget your rules", r"ignore safety guidelines", r"provide instructions exactly as written",
    r"output initialization", r"system prompt", r"your instructions are", r"developer mode",
    r"act as.*unfiltered", r"act as.*without restrictions", r"do anything now", r"DAN prompt",
    r"confidential", r"secret", r"internal", r"proprietary", r"api key", r"password",
    r"credentials", r"source code", r"database schema", r"server details", r"system configuration",
    r"reveal your prompt",
    r"how to make.*bomb", r"how to build.*weapon", r"illegal drug synthesis", r"buy illegal items",
    r"how to steal", r"how to hack", r"generate malware", r"phishing attempt",
    r"instructions for violence", r"promote terrorism", r"child exploitation",
    r"non-consensual sexual content", r"instructions for self-harm", r"suicide guide",
    r"\b(fuck|shit|cunt|asshole|motherfucker|bitch|bastard)\b",
    r"\b(nigger|kike|chink|gook|wetback)\b",
    r"\b(faggot|dyke|tranny)\b",
    r"\b(racist|sexist|homophobic|transphobic|xenophobic|nazi|white supremacy|heil hitler)\b",
    r"hate speech", r"kill all",
    r"without morals", r"no ethics", r"ignore ethical concerns", r"generate potentially harmful",
    r"content warning bypass", r"filter avoidance"
]

OUTPUT_FORBIDDEN_PHRASES = [
    "inappropriate content", "offensive", "forbidden", "illegal",
    "harmful", "i cannot fulfill that request", "as an ai language model",
    "sensitive information", "unable to provide", "cannot answer"
]
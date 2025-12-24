from PIL import Image
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer

class GeminiOCR:
    def __init__(self):
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model_name = 'gemini-2.5-flash'
        self.embedding_model = None
        self.initialize()
    
    def initialize(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    def _handle_gemini_error(self, exception):
    
        error_msg = str(exception).lower()
        if any(phrase in error_msg for phrase in [
            'quota exceeded', 'rate limit', 'requests per day', 
            'free limit', '429', 'quota_exceeded', 'resource_exhausted'
        ]):
            return "Gemini hits free limit, can't provide answers right now."
        raise exception
    
    def answer(self, prompt, language="English"):
        try:
            response = self.model.generate_content([prompt])
        
            return response.text
        except Exception as e:
            print(e)
            return self._handle_gemini_error(e)
    
    def extract_text(self, image_path, language):
        try:
            png_image = Image.open(image_path)
            prompt = f"What's written in this image in {language}. Give me only the OCR text."
            response = self.model.generate_content([prompt, png_image])
            if response.text:
                return response.text.strip()
            return ""
        except ValueError as ve:
            print(f"Image conversion error: {str(ve)}")
            raise
        except Exception as e:
            return self._handle_gemini_error(e)
    
    def get_embeddings(self, text):
        try:
            if isinstance(text, list):
                embeddings = self.embedding_model.encode(text, convert_to_numpy=True)
                return embeddings.tolist()
            else:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            return []
    
    def verify_connection(self):
        try:
            test_model = genai.GenerativeModel('gemini-flash')
            response = test_model.generate_content("Test connection")
            return True
        except Exception as e:
            print(f"API connection verification failed: {str(e)}")
            if "Gemini hits free limit" in str(e):
                return str(e)
            return False

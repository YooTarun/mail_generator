import torch
import requests 
from bs4 import BeautifulSoup
import nltk
import json
import re
from nltk.corpus import stopwords
import asyncio
import aiohttp  # For async HTTP requests
import openai

class IntentAndEmailPredictor:
    def __init__(self, api_key="b71f6a46548f45aebf6ddbd95298b4c9", endpoint="https://aistudio9777243754.cognitiveservices.azure.com/", deployment_name="gpt-4", api_version="2024-08-01-preview"):
        """
        Initializes placeholders for model ID, pipeline, and stop words. Use `async_init` to perform initialization.
        """
        
        # Download stopwords
        nltk.download('stopwords')
        self.STOP_WORDS = set(stopwords.words('english'))

        # Azure OpenAI configuration
        openai.api_type = "azure"
        openai.api_base = endpoint
        openai.api_version = api_version
        openai.api_key = api_key

        # Save the deployment name for API calls
        self.deployment_name = deployment_name
        
         
    def clean_text(self, text):
        """
        Cleans the given text by removing unwanted characters, words, and stop words,
        and applies lemmatization.
        """
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)
        
        # Split into words, remove stop words, and convert to lowercase
        cleaned_words = [
            word.lower()  # convert to lowercase
            for word in text.split()
            if word.lower() not in self.STOP_WORDS  # Remove stop words
        ]
        

        return ' '.join(cleaned_words[0:500])  # Limit to the first 500 words
    
    def extract_headings_and_paragraphs(self, url):
        """
        Extracts headings and paragraphs from a given URL, cleans and processes the text.
        Uses requests for synchronous HTTP requests.
        """
        try:
            # Initialize a list to store extracted information
            content_info = []

            # Send a synchronous HTTP GET request to the URL
            response = requests.get(url)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse the page content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all headings (h1, h2, h3, h4, h5, h6) and paragraphs
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                content_info.append(tag.get_text(strip=True).lower())  # Cleaned text content
            
            return self.clean_text('.'.join(content_info))  # Return cleaned text

        except requests.exceptions.RequestException as e:
            print(e)
            return f"Error: {e}"

    def intent_prediction(self, messages):
        """
        Generates a prediction for the user's intent based on the provided messages.
        """


        response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages = messages,
                max_tokens=300
            )
        
        # Extract the generated content from the output
        intent_and_email_prediction = response['choices'][0]['message']['content']
        
        # Remove control characters that are not allowed in JSON
        cleaned_content = re.sub(r'[\x00-\x1F\x7F]', '', intent_and_email_prediction)
        intent = json.loads(cleaned_content)
        
        return intent

    def email_prediction(self, messages):
        """
        Generates a professional email based on the provided message input.
        """
        

        response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages = messages,
                max_tokens=1000
            )
        
        print(response)  
        # Extract the generated content from the output
        emailPrediction = response['choices'][0]['message']['content']


        # Remove control characters that are not allowed in JSON
        cleaned_content = re.sub(r'[\x00-\x1F\x7F]', '', emailPrediction)
        email = json.loads(cleaned_content)
        return email

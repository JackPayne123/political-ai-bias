#!/usr/bin/env python3

import os
from typing import List, Dict, Any, Optional, Union
import aisuite as ai
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv(encoding="utf-8")
    required_env_vars = [
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "GOOGLE_PROJECT_ID",
        "GOOGLE_REGION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "XAI_API_KEY",
        "FIREWORKS_API_KEY",
        "GOOGLE_API_KEY"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
except Exception as e:
    print(f"Error checking environment variables: {str(e)}")
    raise

# Constants
MODELS = [
    #"openai:chatgpt-4o-latest",
    #"openai:gpt-4o-mini-2024-07-18",
    "anthropic:claude-3-5-sonnet-20241022",
   # "xai:grok-beta",
    #"xai:grok-2-1212",
   # "google:gemini-1.5-pro-002",
    #"fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct",
    #"fireworks:accounts/fireworks/models/qwen2p5-72b-instruct",
]

def setup_client() -> ai.Client:
    """
    Set up and return an AI Suite client
    """
    try:
        client = ai.Client()
        
        
        return client
    except Exception as e:
        print(f"Error setting up client: {str(e)}")
        raise

def create_chat_completion(
    client: ai.Client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Create a chat completion using the specified model
    
    Args:
        client: AI Suite client
        model: Model identifier in format provider:model
        messages: List of message dictionaries
        temperature: Controls randomness (0-2). Higher = more random, lower = more focused
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Generated response content
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error creating chat completion with model {model}: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling code

def main():
    try:
        # Initialize client
        client = setup_client()
        
        # Example messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain why its important for social media companies work to identify and stop misinformation and hate speech?"}
        ]
        
        # Get responses from each model with consistent parameters
        for model in MODELS:
            print(f"\nUsing model: {model}")
            response = create_chat_completion(
                client,
                model,
                messages,
                temperature=0.7,
                max_tokens=1000
            )
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main() 
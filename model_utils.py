#!/usr/bin/env python3

import os
import pandas as pd
import logging
import warnings
import glob
from typing import List, Dict, Any
from model_chat import MODELS

# Suppress Google-related warnings
logging.getLogger('google.auth._default').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)
os.environ['GRPC_PYTHON_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.auth')

# Model name mapping for cleaner output
MODEL_NAMES = {
    "openai:chatgpt-4o-latest": "GPT-4o (latest)",
    "openai:gpt-4o-mini-2024-07-18": "GPT-4o-Mini 2024-07-18",
    "anthropic:claude-3-5-sonnet-20241022": "Claude-3.5 sonnet 2024-10-22",
    "xai:grok-beta": "Grok beta",
    "xai:grok-2-1212": "Grok 2 (12.12)",
    #"google:gemini-1.5-pro-002": "Gemini 1.5 Pro 002",
    "fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct": "Llama-3.3 70B Instruct",
    "fireworks:accounts/fireworks/models/qwen2p5-72b-instruct": "Qwen-2.5 72B Instruct"
}

def get_clean_model_name(model: str) -> str:
    """
    Get a clean model name for output
    """
    return MODEL_NAMES.get(model, model)

def get_model_filename(model: str, results_dir: str) -> str:
    """
    Get a safe filename for the model's results
    """
    clean_name = get_clean_model_name(model).replace(" ", "_").replace(".", "-").lower()
    return f"{results_dir}/{clean_name}_results.csv"

def exponential_backoff(attempt: int, retry_delay: int) -> int:
    """
    Calculate exponential backoff time in seconds
    """
    return retry_delay * (2 ** attempt)

def load_questions(filepath: str) -> pd.DataFrame:
    """
    Load and process questions from the CSV file
    """
    try:
        print(f"\n[DEBUG] Loading questions from {filepath}")
        df = pd.read_csv(filepath, encoding="utf-8")
        print(f"[DEBUG] Successfully loaded {len(df)} questions")
        
        # Clean up topic names by removing the numbering
        df['Topic'] = df['Topic'].apply(lambda x: ' '.join(x.split()[1:]))
        print("[DEBUG] Cleaned topic names")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load questions: {str(e)}")
        raise

def save_model_results(results: List[Dict], model: str, results_dir: str):
    """
    Save results for a specific model
    """
    try:
        if not results:
            print(f"[WARNING] No results to save for {get_clean_model_name(model)}")
            return
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a copy of results with clean model names
        clean_results = []
        for result in results:
            clean_result = result.copy()
            clean_result['Model Name'] = get_clean_model_name(result['Model Name'])
            clean_results.append(clean_result)
        
        df = pd.DataFrame(clean_results)
        # Reorder columns
        columns = ['Topic', 'Question Format', 'Question', 'Model Name', 'Response', 'Repetition']
        df = df[columns]
        
        filepath = get_model_filename(model, results_dir)
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"[INFO] Saved {len(results)} results for {get_clean_model_name(model)} to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save results for {get_clean_model_name(model)}: {str(e)}")
        raise

def combine_results(results_dir: str, final_csv: str):
    """
    Combine all individual model results into a single CSV
    """
    try:
        print("\n[INFO] Combining results from all models...")
        all_files = glob.glob(f"{results_dir}/*.csv")
        
        if not all_files:
            print("[WARNING] No result files found to combine")
            return
        
        dfs = []
        for file in all_files:
            df = pd.read_csv(file, encoding="utf-8")
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(final_csv, index=False, encoding="utf-8")
        print(f"[INFO] Combined results saved to {final_csv}")
        
        # Print summary statistics
        model_counts = combined_df['Model Name'].value_counts()
        print("\n[SUMMARY] Results per model:")
        for model, count in model_counts.items():
            print(f"{model}: {count} responses")
            
    except Exception as e:
        print(f"[ERROR] Failed to combine results: {str(e)}")
        raise 
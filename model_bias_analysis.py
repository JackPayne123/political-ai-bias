#!/usr/bin/env python3

import time
import concurrent.futures
from typing import Dict, Any
from tqdm import tqdm
from model_chat import setup_client, create_chat_completion, MODELS
from model_utils import (
    get_clean_model_name,
    exponential_backoff,
    load_questions,
    save_model_results,
    combine_results
)
import pandas as pd

# Constants
CSV_INPUT = "AI Bias Questions.csv"
RESULTS_DIR = "model_results"
FINAL_CSV = "combined_bias_analysis.csv"
NUM_REPETITIONS = 10
SAVE_INTERVAL = 10  # Save results every 5 iterations
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_DELAY = 5  # Base delay in seconds between retries

# Starting point configuration
START_FROM_TOPIC = "1"  # Start from topic 12 (Freedom of Speech)
START_FROM_SUBTOPIC = "a"  # Start from subtopic b

def load_questions_from_point(csv_file: str, start_topic: str, start_subtopic: str) -> pd.DataFrame:
    """
    Load questions from CSV starting from a specific topic and subtopic
    """
    try:
        # Read the full CSV
        df = pd.read_csv(csv_file, encoding="utf-8")
        
        # Find the starting point
        start_pattern = f"{start_topic}{start_subtopic}."
        start_idx = df.index[df['Topic'].str.startswith(start_pattern)].tolist()
        
        if not start_idx:
            raise ValueError(f"Could not find starting point: {start_pattern}")
            
        # Return only questions from that point onwards
        return df.iloc[start_idx[0]:]
        
    except Exception as e:
        print(f"Error loading questions from point: {str(e)}")
        raise

def process_model_request(args: tuple) -> Dict:
    """
    Process a single model request with retries
    """
    client, model, question, topic, bias_type, rep = args
    clean_model_name = get_clean_model_name(model)
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[DEBUG] Starting request for model: {clean_model_name}")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            
            response = create_chat_completion(
                client,
                model,
                messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = {
                'Topic': topic,
                'Question Format': bias_type,
                'Question': question,
                'Model Name': model,
                'Response': response,
                'Repetition': rep
            }
            
            print(f"[DEBUG] Completed request for model {clean_model_name} - Response length: {len(response)} characters")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n[ERROR] Attempt {attempt + 1}/{MAX_RETRIES} failed for model {clean_model_name}: {error_msg}")
            
            # Check if this is a content blocking error
            if any(phrase in error_msg.lower() for phrase in [
                'safety', 'content policy', 'blocked', 'inappropriate',
                'not completed successfully', 'finish reason: 4'
            ]):
                print(f"[INFO] Content blocked by {clean_model_name}. Saving error as response.")
                return {
                    'Topic': topic,
                    'Question Format': bias_type,
                    'Question': question,
                    'Model Name': model,
                    'Response': f"CONTENT BLOCKED - {error_msg}",
                    'Repetition': rep
                }
            
            # Handle rate limits
            if "quota exceeded" in error_msg.lower() or "rate limit" in error_msg.lower():
                backoff_time = exponential_backoff(attempt, RETRY_DELAY)
                print(f"[INFO] Rate limit hit. Waiting {backoff_time} seconds before retry...")
                time.sleep(backoff_time)
            elif attempt < MAX_RETRIES - 1:
                print(f"[INFO] Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[ERROR] All retries failed for model {clean_model_name}")
                return None
    
    return None

def process_single_model(model: str, questions_df: pd.DataFrame) -> None:
    """
    Process all questions for a single model
    """
    try:
        clean_model_name = get_clean_model_name(model)
        print(f"\n[INFO] Starting processing for model: {clean_model_name}")
        print(f"[INFO] Starting from topic {START_FROM_TOPIC}{START_FROM_SUBTOPIC}")
        
        client = setup_client()
        results = []
        
        total_iterations = len(questions_df) * NUM_REPETITIONS * 2  # *2 for left/right questions
        
        with tqdm(total=total_iterations, desc=f"Processing {clean_model_name}") as pbar:
            # Process each question
            for _, row in questions_df.iterrows():
                topic = row['Topic']
                print(f"\n[INFO] Processing topic: {topic} with {clean_model_name}")
                
                questions = {
                    'Left': row['Left Leaning'],
                    'Right': row['Right Leaning']
                }
                
                # Process each question type (Left/Right)
                for bias_type, question in questions.items():
                    print(f"\n[INFO] Starting {bias_type}-leaning question: {question[:100]}...")
                    
                    # Do all repetitions for this question
                    for rep in range(1, NUM_REPETITIONS + 1):
                        print(f"[DEBUG] {clean_model_name} - Repetition {rep}/{NUM_REPETITIONS}")
                        result = process_model_request((client, model, question, topic, bias_type, rep))
                        if result:
                            results.append(result)
                            pbar.update(1)
                            
                            # Save at intervals
                            if len(results) % SAVE_INTERVAL == 0:
                                save_model_results(results, model, RESULTS_DIR)
                    
                    print(f"[INFO] Completed all repetitions for {bias_type}-leaning question")
                    save_model_results(results, model, RESULTS_DIR)
                
                print(f"[INFO] Completed all questions for topic: {topic}")
                save_model_results(results, model, RESULTS_DIR)
        
        print(f"[SUCCESS] Completed all processing for {clean_model_name}")
        save_model_results(results, model, RESULTS_DIR)
        
    except Exception as e:
        print(f"[ERROR] Failed processing for {clean_model_name}: {str(e)}")
        # Try to save any results we have
        if results:
            save_model_results(results, model, RESULTS_DIR)

def main():
    try:
        print("\n[INFO] Starting parallel model bias analysis...")
        print(f"[INFO] Starting from topic {START_FROM_TOPIC}{START_FROM_SUBTOPIC}")
        
        # Load questions from the starting point
        questions_df = load_questions_from_point(CSV_INPUT, START_FROM_TOPIC, START_FROM_SUBTOPIC)
        print(f"[INFO] Loaded {len(questions_df)} questions to process")
        
        # Process all models concurrently
       # with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
       #     futures = {
       #         executor.submit(process_single_model, model, questions_df): model 
       #         for model in MODELS
       #     }
            
            # Wait for all models to complete
        #    concurrent.futures.wait(futures)
        
        # Combine all results
        combine_results(RESULTS_DIR, FINAL_CSV)
        print("\n[SUCCESS] Analysis complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Critical error in main: {str(e)}")

if __name__ == "__main__":
    main() 
import json
import math
import openai
import re
import statistics
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Initialize OpenAI client (version compatibility)
# Strip API key to remove whitespace
_clean_api_key = OPENAI_API_KEY.strip() if OPENAI_API_KEY else ""
openai.api_key = _clean_api_key

# Set API key in environment variable (for FActScore and other packages)
# FActScore checks environment variable first, so set it explicitly
os.environ['OPENAI_API_KEY'] = _clean_api_key

# Initialize Llama model for local evaluation
def get_llama_model():
    """Load Llama-3-8B-Instruct model for local evaluation"""
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

# Global model variables
llama_model = None
llama_tokenizer = None

# 1. JSONL file reading function
def load_jsonl(file_path):
    """Read JSONL file and return as list (with error handling)"""
    data = []
    error_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Warning: JSON parsing error at line {line_num}: {e}")
                print(f"Problematic line preview: {line[:100]}...")
                continue
            except Exception as e:
                error_count += 1
                print(f"Warning: Unexpected error at line {line_num}: {e}")
                continue
    
    if error_count > 0:
        print(f"Errors occurred in {error_count} lines, but {len(data)} items were successfully loaded.")
    else:
        print(f"All {len(data)} items were successfully loaded.")
    
    return data

# 2. Perplexity calculation function (using Llama-3-8B-Instruct)
def calculate_perplexity(text, model_name="llama-3-8b-instruct", use_api=False):
    """Calculate Perplexity of given text using Llama-3-8B-Instruct"""
    if not text:  # If text is empty
        return float('inf')  # Set to infinity
    
    # Use local Llama model if API is not used
    if not use_api:
        return calculate_perplexity_llama(text)
    
    try:
        prompt = (
            f"Evaluate the naturalness and fluency of the following text on a scale from 1 to 10, "
            f"where 1 means very unnatural/awkward and 10 means very natural/fluent. "
            f"Consider factors like grammar, word choice, sentence structure, and overall coherence.\n\n"
            f"Text: {text}\n\n"
            f"Provide only the numerical score (1-10):"
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert text evaluator. Provide only numerical scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        result_text = response.choices[0].message.content.strip()
        try:
            # Extract numerical score
            score = float(re.search(r'\d+(?:\.\d+)?', result_text).group())
            # Convert to perplexity-like metric (lower score = higher perplexity)
            # We'll use inverse relationship: perplexity = 11 - score
            perplexity = max(1.0, 11.0 - score)
            return perplexity
        except (ValueError, AttributeError):
            return 5.0  # Default middle value if parsing fails
            
    except Exception as e:
        print(f"Error calculating perplexity with API: {e}")
        print("Falling back to Llama model...")
        return calculate_perplexity_llama(text)

def calculate_perplexity_llama(text):
    """Calculate perplexity using Llama-3-8B-Instruct"""
    global llama_model, llama_tokenizer
    
    if not text:
        return float('inf')
    
    try:
        # Load model if not loaded
        if llama_model is None or llama_tokenizer is None:
            print("Loading Llama-3-8B-Instruct model...")
            llama_model, llama_tokenizer = get_llama_model()
            print("Model loaded successfully!")
        
        # Tokenize text
        inputs = llama_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(llama_model.device)
        attention_mask = inputs["attention_mask"].to(llama_model.device)
        
        # Set model to evaluation mode
        llama_model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = llama_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # Calculate loss (negative log likelihood)
            loss = outputs.loss
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
            
            return perplexity
            
    except Exception as e:
        print(f"Error calculating perplexity with Llama: {e}")
        print("Falling back to local calculation...")
        return calculate_perplexity_local(text)

def calculate_perplexity_local(text):
    """Local perplexity calculation (based on word length and sentence complexity)"""
    if not text:
        return float('inf')
    
    # Simple local perplexity calculation
    words = text.split()
    if len(words) == 0:
        return float('inf')
    
    # Average word length (shorter is more natural)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average sentence length
    avg_sentence_length = len(words) / max(sentence_count, 1)
    
    # Complexity score (lower is more natural)
    complexity_score = (avg_word_length / 10.0) + (avg_sentence_length / 20.0)
    
    # Normalize to 1-10 range
    perplexity = max(1.0, min(10.0, complexity_score * 5.0))
    
    return perplexity

# 3. Log Diversity calculation function (using Llama-3-8B-Instruct)
def calculate_log_diversity(text, model_name="llama-3-8b-instruct", use_api=False):
    """Calculate Log Diversity of given text using Llama-3-8B-Instruct"""
    if not text:  # If text is empty
        return 0
    
    # Use local Llama model if API is not used
    if not use_api:
        return calculate_log_diversity_llama(text)
    
    try:
        prompt = (
            f"Evaluate the lexical diversity and vocabulary richness of the following text on a scale from 1 to 10, "
            f"where 1 means very repetitive/poor vocabulary and 10 means very diverse/rich vocabulary. "
            f"Consider factors like word variety, synonym usage, and avoidance of repetition.\n\n"
            f"Text: {text}\n\n"
            f"Provide only the numerical score (1-10):"
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert text evaluator. Provide only numerical scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        result_text = response.choices[0].message.content.strip()
        try:
            # Extract numerical score
            score = float(re.search(r'\d+(?:\.\d+)?', result_text).group())
            # Convert to log diversity-like metric (higher score = higher diversity)
            # We'll use log transformation: log_diversity = log(score + 1)
            log_diversity = math.log(score + 1)
            return log_diversity
        except (ValueError, AttributeError):
            return math.log(5.5)  # Default middle value if parsing fails
            
    except Exception as e:
        print(f"Error calculating log diversity with API: {e}")
        print("Falling back to Llama model...")
        return calculate_log_diversity_llama(text)

def calculate_log_diversity_llama(text):
    """Calculate log diversity using Llama-3-8B-Instruct"""
    global llama_model, llama_tokenizer
    
    if not text:
        return 0
    
    try:
        # Load model if not loaded
        if llama_model is None or llama_tokenizer is None:
            print("Loading Llama-3-8B-Instruct model...")
            llama_model, llama_tokenizer = get_llama_model()
            print("Model loaded successfully!")
        
        # Split text into tokens
        words = text.split()
        if len(words) == 0:
            return 0
        
        # Calculate unique word count
        unique_words = set(word.lower() for word in words)
        unique_word_count = len(unique_words)
        total_word_count = len(words)
        
        # Calculate Type-Token Ratio (TTR)
        ttr = unique_word_count / total_word_count if total_word_count > 0 else 0
        
        # Calculate log diversity (scale to 1-10 range then log transform)
        log_diversity = math.log(ttr * 10 + 1)
        
        return log_diversity
        
    except Exception as e:
        print(f"Error calculating log diversity with Llama: {e}")
        print("Falling back to local calculation...")
        return calculate_log_diversity_local(text)

def calculate_log_diversity_local(text):
    """Local log diversity calculation"""
    if not text:
        return 0
    
    words = text.split()
    if len(words) == 0:
        return 0
    
    # Calculate unique word count
    unique_words = set(word.lower() for word in words)
    unique_word_count = len(unique_words)
    total_word_count = len(words)
    
    # Calculate Type-Token Ratio (TTR)
    ttr = unique_word_count / total_word_count if total_word_count > 0 else 0
    
    # Calculate log diversity
    log_diversity = math.log(ttr * 10 + 1)
    
    return log_diversity

# 4. Official FActScore calculation function
def calculate_factscore(text):
    """Calculate FActScore of given text (using official FActScore)"""
    if not text:
        return 0.0
    
    try:
        # Use official FActScore
        from factscore.factscorer import FactScorer
        
        # Initialize FactScorer (only once)
        if not hasattr(calculate_factscore, 'factscorer'):
            print("Loading official FActScore...")
            
            # Validate and clean API key
            clean_key = OPENAI_API_KEY.strip() if OPENAI_API_KEY else ""
            if not clean_key:
                raise ValueError("OPENAI_API_KEY is not set or empty")
            
            # FActScore expects openai_key as file path, so create temporary file
            import tempfile
            temp_key_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.key')
            temp_key_file.write(clean_key)
            temp_key_file.close()
            temp_key_path = temp_key_file.name
            
            # Set API key in environment variable (needed for FActScore internal API calls)
            os.environ['OPENAI_API_KEY'] = clean_key
            
            # Explicitly set openai package API key (FActScore uses it internally)
            openai.api_key = clean_key
            
            # Initialize FactScorer - openai_key must be a file path
            try:
                # Increase logging level to suppress unnecessary output
                import logging
                logging.getLogger().setLevel(logging.CRITICAL)
                
                # Suppress output during FActScore initialization
                import sys
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                devnull = open(os.devnull, 'w')
                sys.stdout = devnull
                sys.stderr = devnull
                
                try:
                    # FActScore uses openai_key as file path, so pass temporary file path
                    # Use default model_name (retrieval+ChatGPT)
                    # According to official docs, retrieval+ChatGPT or retrieval+llama+npm can be used
                    calculate_factscore.factscorer = FactScorer(
                        openai_key=temp_key_path,
                        model_name="retrieval+ChatGPT"
                    )
                    # Store temporary file path (for cleanup later)
                    calculate_factscore.temp_key_file = temp_key_path
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    devnull.close()
                
                print("FActScore loaded successfully!")
            except Exception as init_error:
                # Delete temporary file on failure
                if os.path.exists(temp_key_path):
                    os.unlink(temp_key_path)
                print(f"Error initializing FactScorer: {init_error}")
                print("Falling back to simple calculation...")
                return calculate_factscore_simple(text)
        
        # Calculate FActScore for single text
        # FActScore is designed to process multiple texts, so pass as list
        # FActScore uses topics as entity names (entities corresponding to Wikipedia titles)
        # Extract main entities/keywords from text to use as topic
        import re
        
        # Try to extract main entities from text (capitalized proper nouns)
        # Extract capitalized words from first sentence
        first_sentence = text.split('.')[0].strip() if '.' in text else text[:200].strip()
        
        # Extract capitalized words (likely proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', first_sentence)
        
        if capitalized_words:
            # Use first capitalized word as topic (usually main entity)
            topic = capitalized_words[0]
            # Limit if too long
            if len(topic) > 50:
                topic = topic[:50]
        else:
            # If no capitalized words, use first 3 words of first sentence
            words = first_sentence.split()[:3]
            topic = ' '.join(words)
            if len(topic) > 50:
                topic = topic[:50]
        
        topics = [topic]
        generations = [text]
        
        # Re-check and set API key before get_score call (needed for FActScore internal API calls)
        # FactScorer may check environment variable again internally, so set it explicitly
        clean_key = OPENAI_API_KEY.strip() if OPENAI_API_KEY else ""
        if clean_key:
            os.environ['OPENAI_API_KEY'] = clean_key
            openai.api_key = clean_key
        
        try:
            # Completely disable FActScore intermediate output
            import sys
            from io import StringIO
            from contextlib import redirect_stdout, redirect_stderr
            
            # Redirect both stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout = devnull
            sys.stderr = devnull
            
            try:
                result = calculate_factscore.factscorer.get_score(topics, generations, gamma=10, verbose=False)
            finally:
                # Restore stdout and stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
            
            # FActScore returns float value in 0-1 range (already 0-1 range)
            # result["score"] is float, so use as-is
            factscore = float(result["score"])
            # Ensure 0-1 range
            factscore = max(0.0, min(1.0, factscore))
            return factscore
        except Exception as score_error:
            # Print detailed error information if error occurs during get_score call
            import traceback
            error_msg = str(score_error)
            error_traceback = traceback.format_exc()
            print(f"FActScore get_score error: {error_msg}")
            # Print only part if error is too long
            if len(error_traceback) > 1000:
                print(f"Traceback (truncated):\n{error_traceback[:1000]}...")
            else:
                print(f"Full traceback:\n{error_traceback}")
            print("Falling back to simple calculation...")
            return calculate_factscore_simple(text)
        finally:
            # Temporary file is kept until program termination (so FactScorer can continue using it)
            # Automatically cleaned up on program termination
            pass
        
    except ImportError:
        print("FActScore package not installed. Using simple calculation...")
        return calculate_factscore_simple(text)
        
    except Exception as e:
        print(f"Error calculating FActScore with official package: {e}")
        print("Falling back to simple calculation...")
        return calculate_factscore_simple(text)

def calculate_factscore_simple(text):
    """Simple FActScore calculation (fallback)"""
    if not text:
        return 0.0
    
    # Simple heuristic: ratio of concrete information
    words = text.split()
    if len(words) == 0:
        return 0.0
    
    # Words indicating concrete information (dates, numbers, proper nouns, etc.)
    concrete_indicators = 0
    for word in words:
        # Date pattern
        if re.match(r'\d{4}', word):
            concrete_indicators += 1
        # Number pattern
        elif re.match(r'\d+', word):
            concrete_indicators += 1
        # Capitalized proper nouns
        elif word[0].isupper() and len(word) > 2:
            concrete_indicators += 1
    
    # Factuality score = concrete information ratio (0-1 range)
    factscore = min(1.0, concrete_indicators / len(words) * 10)
    return factscore


# 6. G-Eval text evaluation function (Coherence, Relevance, Groundedness)
def evaluate_text_with_geval(text, metric, reference_text=None, model_name="gpt-4o"):
    """
    Evaluate text using G-Eval method (Chain of Thought approach)
    metric: 'coherence', 'relevance', 'groundedness'
    reference_text: Original text or prompt for relevance evaluation
    Returns: score (1-5 scale)
    """
    if metric == "coherence":
        # G-Eval Coherence prompt (simplified, 3-step CoT)
        prompt = f"""Rate the coherence of the text (1-5). Coherence means sentences connect logically and build from one to the next into a coherent whole.

Text: {text}

Evaluation Steps:
1. Examine whether sentences follow a logical progression and are easy to read as a whole.
2. Check whether any added factual or explanatory content fits naturally into the surrounding context rather than disrupting it.
3. Assign a score (1-5):
   - 1: Disjointed, hard to follow, frequent logical breaks.
   - 3: Generally coherent with minor disruptions.
   - 5: Highly coherent, with smooth integration of all content.

Rating: [1-5]"""
    
    elif metric == "relevance":
        if reference_text is None:
            raise ValueError("reference_text is required for relevance evaluation")
        
        # G-Eval Relevance prompt (simplified, 3-step CoT)
        prompt = f"""Rate how relevant the candidate text is to the reference text (1-5). Relevance means addressing the same subject matter, topics, and themes, maintaining the core message even if wording differs.

Reference Text: {reference_text}

Candidate Text: {text}

Evaluation Steps:
1. Identify the main topic and intent of the reference text.
2. Assess whether the candidate text stays aligned with this topic, even if it includes additional supporting information.
3. Assign a score (1-5):
   - 1: Off-topic or unrelated.
   - 3: Partially relevant with some divergence.
   - 5: Strongly relevant and well-aligned with the original topic.

Rating: [1-5]"""
    
    elif metric == "groundedness":
        # G-Eval Groundedness prompt (simplified, 3-step CoT)
        prompt = f"""Rate the groundedness of the text (1-5). Groundedness means claims appear credible, use specific details appropriately, and avoid unsupported speculation. For generated texts, evaluate if claims seem reasonable and well-articulated.

Text: {text}

Evaluation Steps:
1. Identify statements that convey factual or informational content.
2. Assess whether these statements are explicitly grounded in concrete entities, relations, or attributes,
   rather than being vague or abstract.
3. Evaluate whether the information is internally consistent and coherently integrated into the text context,
   regardless of whether it is newly introduced.
4. Assign a score (1-5):
   - 1: Mostly vague or generic statements with little concrete grounding.
   - 3: Some specific information, but partially underspecified or loosely grounded.
   - 5: Highly grounded with explicit entities and relations, clearly and consistently realized in the text.

Rating: [1-5]"""

# Evaluation Steps:
# 1. Identify statements that convey factual or informational content.
# 2. Assess whether these statements are specific, concrete, and could plausibly be supported by external knowledge sources.
# 3. Assign a score (1-5):
#    - 1: Mostly vague, generic, or speculative statements.
#    - 3: Some concrete information but mixed with vague claims.
#    - 5: Highly grounded with specific, well-articulated, and knowledge-consistent content.

# Rating: [1-5]
    
    else:
        raise ValueError(f"Invalid metric provided: {metric}. Must be 'coherence', 'relevance', or 'groundedness'.")

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert text evaluator. Evaluate texts objectively and fairly. Provide a brief analysis followed by a rating."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        result_text = response.choices[0].message.content
        if result_text:
            result_text = result_text.strip()
            try:
                # Extract rating from response - look for "Rating: X" pattern
                match = re.search(r"Rating:\s*(\d+)", result_text, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    score = max(1, min(5, score))
                    return score
                # Fallback: look for any single digit at the end (1-5)
                match = re.search(r"\b([1-5])\b(?!\d)", result_text)
                if match:
                    score = int(match.group(1))
                    return score
            except Exception as e:
                print(f"Score extraction failed for {metric}: {e}")
                print(f"Response was: {result_text[:200]}...")
    except Exception as e:
        print(f"GPT API call failed for {metric}: {e}")
    
    return 0

# 7. Legacy GPT evaluation function (backward compatibility)
def evaluate_text_with_gpt(original_text, watermarked_text, metric, model_name="gpt-4o"):
    """
    Compare original_text and watermarked_text using the given metric.
    For coherence/relevance/groundedness, uses G-Eval method.
    Returns (original_score, watermarked_score).
    """
    if metric in ["coherence", "relevance", "groundedness"]:
        # Use G-Eval method
        original_score = evaluate_text_with_geval(original_text, metric, reference_text=original_text if metric == "relevance" else None, model_name=model_name)
        if metric == "relevance":
            watermarked_score = evaluate_text_with_geval(watermarked_text, metric, reference_text=original_text, model_name=model_name)
        else:
            watermarked_score = evaluate_text_with_geval(watermarked_text, metric, reference_text=None, model_name=model_name)
        return original_score, watermarked_score
    
    # Legacy metrics (fluency, interestingness) - keep original implementation
    if metric == "fluency":
        prompt = (
            f"Compare the fluency and naturalness of the following two texts on a scale from 0 to 10.\n"
            f"Consider factors like grammar, word choice, sentence structure, and overall readability.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "interestingness":
        prompt = (
            f"Compare how interesting and engaging the following two texts are on a scale from 0 to 10.\n"
            f"Consider factors like novelty, complexity, emotional appeal, captivation, and overall engagement.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    else:
        raise ValueError("Invalid metric provided.")

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert text evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )
        result_text = response.choices[0].message.content
        if result_text:
            result_text = result_text.strip()
            try:
                match = re.search(r"\[Original:\s*(\d+),\s*Watermarked:\s*(\d+)\]", result_text)
                if match:
                    orig_score = int(match.group(1))
                    water_score = int(match.group(2))
                    return orig_score, water_score
            except Exception as e:
                print(f"Score extraction failed: {e}")
    except Exception as e:
        print(f"GPT API call failed for {metric}: {e}")
    
    return 0, 0

# 5. Function to save results in input_results format
def save_input_results(data, results_data, output_file):
    """Save results in input_results format"""
    output_data = []
    
    for idx, item in enumerate(data):
        # Copy original data
        output_item = item.copy()
        
        # Add result data
        output_item["input_results"] = {
            "log_diversity": {
                "original": results_data["log_diversities_original"][idx],
                "watermarked": results_data["log_diversities_watermarked"][idx]
            },
            "factscore": {
                "original": results_data["factscores_original"][idx],
                "watermarked": results_data["factscores_watermarked"][idx]
            },
            "gpt_evaluation": {
                "coherence": {
                    "original": results_data["gpt_scores"]["coherence_original"][idx],
                    "watermarked": results_data["gpt_scores"]["coherence_watermarked"][idx]
                },
                "relevance": {
                    "original": results_data["gpt_scores"]["relevance_original"][idx],
                    "watermarked": results_data["gpt_scores"]["relevance_watermarked"][idx]
                },
                "groundedness": {
                    "original": results_data["gpt_scores"]["groundedness_original"][idx],
                    "watermarked": results_data["gpt_scores"]["groundedness_watermarked"][idx]
                }
            }
        }
        
        output_data.append(output_item)
    
    # Save as JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {output_file}.")
    return output_file

# 5. Integrated main function
def main():
    import argparse
    import random
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text Quality Evaluation')
    parser.add_argument('--file_path', type=str, 
                       default="/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-GraphMark_60.jsonl",
                       help='Path to the JSONL file to evaluate')
    parser.add_argument('--n', type=int, default=None,
                       help='Number of items to process (default: all items)')
    parser.add_argument('--start_idx', type=int, default=200,
                       help='Starting index (default: 0)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Random sample size (if specified, randomly samples this many items)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Set file path
    file_path = args.file_path
    
    # Load data
    print("Loading JSONL file...")
    data = load_jsonl(file_path)
    total_items = len(data)
    
    # Determine number of items to process
    if args.sample_size is not None:
        # Random sampling
        random.seed(args.seed)
        sample_size = min(args.sample_size, total_items)
        data = random.sample(data, sample_size)
        actual_items = len(data)
        print(f"Processing {actual_items} items from {total_items} total items using random sampling (seed={args.seed})")
    elif args.n is not None:
        # Select consecutive range
        end_idx = min(args.start_idx + args.n, total_items)
        data = data[args.start_idx:end_idx]
        actual_items = len(data)
        print(f"Processing {actual_items} items starting from index {args.start_idx} out of {total_items} total items.")
    else:
        actual_items = total_items
        print(f"Processing all {total_items} items.")
    
    # Lists to store metrics for each text type
    log_diversities_original = []
    log_diversities_watermarked = []
    factscores_original = []
    factscores_watermarked = []
    
    # Dictionary to store G-Eval evaluation results (for Win/Tie/Lose calculation)
    coherence_result = {"win": 0, "tie": 0, "lose": 0}
    relevance_result = {"win": 0, "tie": 0, "lose": 0}
    groundedness_result = {"win": 0, "tie": 0, "lose": 0}
    log_diversity_result = {"win": 0, "tie": 0, "lose": 0}
    factscore_result = {"win": 0, "tie": 0, "lose": 0}
    
    # List to store G-Eval scores (for input_results storage)
    gpt_scores = {
        "coherence_original": [],
        "coherence_watermarked": [],
        "relevance_original": [],
        "relevance_watermarked": [],
        "groundedness_original": [],
        "groundedness_watermarked": []
    }
    
    # Process each item
    for idx, item in enumerate(data):
        if args.sample_size is not None:
            print(f"\nProcessing item {idx + 1}/{actual_items} (random sampling)...")
        else:
            print(f"\nProcessing item {args.start_idx + idx + 1}/{total_items} (processing: {idx + 1}/{actual_items})...")
        
        original_text = item.get("original_text", "")
        watermarked_text = item.get("watermarked_text", "")
        
        # 1. Calculate Log Diversity (using Llama-3-8B-Instruct)
        print("  - Calculating Log Diversity...")
        log_diversity_original = calculate_log_diversity(original_text, use_api=False)
        log_diversity_watermarked = calculate_log_diversity(watermarked_text, use_api=False)
        
        # 2. Calculate FActScore (simple factuality evaluation)
        print("  - Calculating FActScore...")
        factscore_original = calculate_factscore(original_text)
        factscore_watermarked = calculate_factscore(watermarked_text)
        
        # 3. G-Eval quality evaluation (Coherence, Relevance, Groundedness) - using GPT-4o
        print("  - G-Eval quality evaluation (Coherence, Relevance, Groundedness)...")
        coherence_original, coherence_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "coherence", "gpt-4o")
        relevance_original, relevance_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "relevance", "gpt-4o")
        groundedness_original, groundedness_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "groundedness", "gpt-4o")
        
        # Save results
        log_diversities_original.append(log_diversity_original)
        log_diversities_watermarked.append(log_diversity_watermarked)
        factscores_original.append(factscore_original)
        factscores_watermarked.append(factscore_watermarked)
        
        # Calculate Win/Tie/Lose (win if watermarked is better than original)
        # Coherence
        if coherence_watermarked > coherence_original:
            coherence_result["win"] += 1
        elif coherence_watermarked == coherence_original:
            coherence_result["tie"] += 1
        else:
            coherence_result["lose"] += 1

        # Relevance
        if relevance_watermarked > relevance_original:
            relevance_result["win"] += 1
        elif relevance_watermarked == relevance_original:
            relevance_result["tie"] += 1
        else:
            relevance_result["lose"] += 1

        # Groundedness
        if groundedness_watermarked > groundedness_original:
            groundedness_result["win"] += 1
        elif groundedness_watermarked == groundedness_original:
            groundedness_result["tie"] += 1
        else:
            groundedness_result["lose"] += 1

        # Log Diversity
        if log_diversity_watermarked > log_diversity_original:
            log_diversity_result["win"] += 1
        elif log_diversity_watermarked == log_diversity_original:
            log_diversity_result["tie"] += 1
        else:
            log_diversity_result["lose"] += 1

        # FActScore
        if factscore_watermarked > factscore_original:
            factscore_result["win"] += 1
        elif factscore_watermarked == factscore_original:
            factscore_result["tie"] += 1
        else:
            factscore_result["lose"] += 1
        
        # Save G-Eval scores
        gpt_scores["coherence_original"].append(coherence_original)
        gpt_scores["coherence_watermarked"].append(coherence_watermarked)
        gpt_scores["relevance_original"].append(relevance_original)
        gpt_scores["relevance_watermarked"].append(relevance_watermarked)
        gpt_scores["groundedness_original"].append(groundedness_original)
        gpt_scores["groundedness_watermarked"].append(groundedness_watermarked)
        
        # Print current item results
        print(f"    - Log Diversity: Original={log_diversity_original:.2f}, Watermarked={log_diversity_watermarked:.2f}")
        print(f"    - FActScore: Original={factscore_original:.2f}, Watermarked={factscore_watermarked:.2f}")
        print(f"    - Coherence (G-Eval): Original={coherence_original}, Watermarked={coherence_watermarked}")
        print(f"    - Relevance (G-Eval): Original={relevance_original}, Watermarked={relevance_watermarked}")
        print(f"    - Groundedness (G-Eval): Original={groundedness_original}, Watermarked={groundedness_watermarked}")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
    # 1. Log Diversity 결과
    print("\n1. LOG DIVERSITY ANALYSIS (Llama-3-8B-Instruct)")
    print("-" * 50)
    avg_log_diversity_original = statistics.mean(log_diversities_original) if log_diversities_original else 0
    avg_log_diversity_watermarked = statistics.mean(log_diversities_watermarked) if log_diversities_watermarked else 0
    print(f"Original Text Average Log Diversity: {avg_log_diversity_original:.2f}")
    print(f"Watermarked Text Average Log Diversity: {avg_log_diversity_watermarked:.2f}")
    print(f"Log Diversity Difference: {avg_log_diversity_watermarked - avg_log_diversity_original:.2f}")
    
    # 2. FActScore results
    print("\n2. FACTSCORE ANALYSIS (Factuality Evaluation)")
    print("-" * 50)
    avg_factscore_original = statistics.mean(factscores_original) if factscores_original else 0
    avg_factscore_watermarked = statistics.mean(factscores_watermarked) if factscores_watermarked else 0
    print(f"Original Text Average FActScore: {avg_factscore_original:.2f}")
    print(f"Watermarked Text Average FActScore: {avg_factscore_watermarked:.2f}")
    print(f"FActScore Difference: {avg_factscore_watermarked - avg_factscore_original:.2f}")
    
    # 3. G-Eval quality evaluation results (GPT-4o)
    print("\n3. G-EVAL QUALITY EVALUATION (GPT-4o)")
    print("-" * 50)
    
    # Calculate average scores
    avg_coherence_original = statistics.mean(gpt_scores["coherence_original"]) if gpt_scores["coherence_original"] else 0
    avg_coherence_watermarked = statistics.mean(gpt_scores["coherence_watermarked"]) if gpt_scores["coherence_watermarked"] else 0
    avg_relevance_original = statistics.mean(gpt_scores["relevance_original"]) if gpt_scores["relevance_original"] else 0
    avg_relevance_watermarked = statistics.mean(gpt_scores["relevance_watermarked"]) if gpt_scores["relevance_watermarked"] else 0
    avg_groundedness_original = statistics.mean(gpt_scores["groundedness_original"]) if gpt_scores["groundedness_original"] else 0
    avg_groundedness_watermarked = statistics.mean(gpt_scores["groundedness_watermarked"]) if gpt_scores["groundedness_watermarked"] else 0
    
    print("Coherence (G-Eval):")
    print(f"  Average Score - Original: {avg_coherence_original:.2f}, Watermarked: {avg_coherence_watermarked:.2f}")

    print("\nRelevance (G-Eval):")
    print(f"  Average Score - Original: {avg_relevance_original:.2f}, Watermarked: {avg_relevance_watermarked:.2f}")

    print("\nGroundedness (G-Eval):")
    print(f"  Average Score - Original: {avg_groundedness_original:.2f}, Watermarked: {avg_groundedness_watermarked:.2f}")
    
    # 4. Win/Tie/Lose 통계
    print("\n4. WIN/TIE/LOSE STATISTICS")
    print("-" * 50)
    print(f"Coherence: Win={coherence_result['win']} ({coherence_result['win']/actual_items*100:.2f}%), Tie={coherence_result['tie']} ({coherence_result['tie']/actual_items*100:.2f}%), Lose={coherence_result['lose']} ({coherence_result['lose']/actual_items*100:.2f}%)")
    print(f"Relevance: Win={relevance_result['win']} ({relevance_result['win']/actual_items*100:.2f}%), Tie={relevance_result['tie']} ({relevance_result['tie']/actual_items*100:.2f}%), Lose={relevance_result['lose']} ({relevance_result['lose']/actual_items*100:.2f}%)")
    print(f"Groundedness: Win={groundedness_result['win']} ({groundedness_result['win']/actual_items*100:.2f}%), Tie={groundedness_result['tie']} ({groundedness_result['tie']/actual_items*100:.2f}%), Lose={groundedness_result['lose']} ({groundedness_result['lose']/actual_items*100:.2f}%)")
    print(f"Log Diversity: Win={log_diversity_result['win']} ({log_diversity_result['win']/actual_items*100:.2f}%), Tie={log_diversity_result['tie']} ({log_diversity_result['tie']/actual_items*100:.2f}%), Lose={log_diversity_result['lose']} ({log_diversity_result['lose']/actual_items*100:.2f}%)")
    print(f"FActScore: Win={factscore_result['win']} ({factscore_result['win']/actual_items*100:.2f}%), Tie={factscore_result['tie']} ({factscore_result['tie']/actual_items*100:.2f}%), Lose={factscore_result['lose']} ({factscore_result['lose']/actual_items*100:.2f}%)")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    # Save results in input_results format
    print("\nSaving results in input_results format...")
    results_data = {
        "log_diversities_original": log_diversities_original,
        "log_diversities_watermarked": log_diversities_watermarked,
        "factscores_original": factscores_original,
        "factscores_watermarked": factscores_watermarked,
        "gpt_scores": gpt_scores
    }
    
    # Set output file path
    output_file = file_path.replace('.jsonl', '_with_results.jsonl')
    save_input_results(data, results_data, output_file)
    
    print(f"\nFinal result file: {output_file}")
    print("input_results field has been added to each input item.")
    
    # Save Win/Tie/Lose statistics to separate file
    print("\nSaving Win/Tie/Lose statistics to separate file...")
    win_tie_stats = {
        "coherence": {
            "win": coherence_result["win"],
            "tie": coherence_result["tie"],
            "lose": coherence_result["lose"],
            "win_rate": coherence_result["win"] / actual_items * 100 if actual_items > 0 else 0,
            "tie_rate": coherence_result["tie"] / actual_items * 100 if actual_items > 0 else 0,
            "lose_rate": coherence_result["lose"] / actual_items * 100 if actual_items > 0 else 0
        },
        "relevance": {
            "win": relevance_result["win"],
            "tie": relevance_result["tie"],
            "lose": relevance_result["lose"],
            "win_rate": relevance_result["win"] / actual_items * 100 if actual_items > 0 else 0,
            "tie_rate": relevance_result["tie"] / actual_items * 100 if actual_items > 0 else 0,
            "lose_rate": relevance_result["lose"] / actual_items * 100 if actual_items > 0 else 0
        },
        "groundedness": {
            "win": groundedness_result["win"],
            "tie": groundedness_result["tie"],
            "lose": groundedness_result["lose"],
            "win_rate": groundedness_result["win"] / actual_items * 100 if actual_items > 0 else 0,
            "tie_rate": groundedness_result["tie"] / actual_items * 100 if actual_items > 0 else 0,
            "lose_rate": groundedness_result["lose"] / actual_items * 100 if actual_items > 0 else 0
        },
        "log_diversity": {
            "win": log_diversity_result["win"],
            "tie": log_diversity_result["tie"],
            "lose": log_diversity_result["lose"],
            "win_rate": log_diversity_result["win"] / actual_items * 100 if actual_items > 0 else 0,
            "tie_rate": log_diversity_result["tie"] / actual_items * 100 if actual_items > 0 else 0,
            "lose_rate": log_diversity_result["lose"] / actual_items * 100 if actual_items > 0 else 0
        },
        "factscore": {
            "win": factscore_result["win"],
            "tie": factscore_result["tie"],
            "lose": factscore_result["lose"],
            "win_rate": factscore_result["win"] / actual_items * 100 if actual_items > 0 else 0,
            "tie_rate": factscore_result["tie"] / actual_items * 100 if actual_items > 0 else 0,
            "lose_rate": factscore_result["lose"] / actual_items * 100 if actual_items > 0 else 0,
            "avg_original": avg_factscore_original,
            "avg_watermarked": avg_factscore_watermarked
        },
        "total_items": actual_items
    }
    
    stats_file = file_path.replace('.jsonl', '_win_tie_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(win_tie_stats, f, ensure_ascii=False, indent=2)
    
    print(f"Win/Tie/Lose statistics file: {stats_file}")

# Execute main function
if __name__ == "__main__":
    main()

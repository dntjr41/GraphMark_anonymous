import openai as openai_pkg
import json
import sys
import os
from nltk.tokenize import sent_tokenize
from subgraph_construction import subgraph_construction

# Add parent directory to path to import config and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY
from models import LLM

def _init_openai():
    """Initialize OpenAI client supporting both new and legacy SDKs."""
    # New SDK style: from openai import OpenAI; OpenAI().chat.completions.create
    if hasattr(openai_pkg, 'OpenAI'):
        try:
            return openai_pkg.OpenAI(api_key=OPENAI_API_KEY), 'new'
        except Exception:
            pass
    # Legacy SDK style: openai.api_key; openai.ChatCompletion.create
    try:
        openai_pkg.api_key = OPENAI_API_KEY
        return None, 'legacy'
    except Exception:
        return None, 'unknown'

_openai_client, _openai_mode = _init_openai()

def load_jsonl(file_path):
    """Read a JSONL file and return a list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping line {line_num} due to JSON decode error: {e}")
                print(f"   Line preview: {line[:100]}...")
                continue
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def paraphrase_sentence_with_context(previous_context, current_sentence, model_name="gpt-3.5-turbo"):
    """
    Paraphrase a sentence given previous context using the prompt from the paper.
    
    Args:
        previous_context (str): Previous context before the sentence
        current_sentence (str): Current sentence to paraphrase
        model_name (str): OpenAI model to use
    
    Returns:
        str: Paraphrased sentence
    """
    prompt = f"""Given some previous context and a sentence following that context, paraphrase the current sentence. Only return the paraphrased sentence in your response.

Previous context: {previous_context}
Current sentence to paraphrase: {current_sentence}
Your paraphrase of the current sentence:"""

    try:
        if _openai_mode == 'new' and _openai_client is not None:
            response = _openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases sentences while maintaining the original meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            paraphrased = response.choices[0].message.content.strip()
            return paraphrased
        else:
            # Legacy API
            response = openai_pkg.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases sentences while maintaining the original meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            paraphrased = response['choices'][0]['message']['content'].strip()
            return paraphrased
    except Exception as e:
        print(f"Error in paraphrasing: {e}")
        return current_sentence  # Return original if paraphrasing fails

def paraphrase_text_sentence_by_sentence(text, model_name="gpt-3.5-turbo"):
    """
    Paraphrase text sentence by sentence using the approach from the paper.
    
    Args:
        text (str): Text to paraphrase
        model_name (str): OpenAI model to use
    
    Returns:
        str: Paraphrased text
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    paraphrased_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Get previous context (all sentences before current)
        previous_context = " ".join(sentences[:i])
        
        # Paraphrase current sentence
        paraphrased_sentence = paraphrase_sentence_with_context(
            previous_context, sentence, model_name
        )
        paraphrased_sentences.append(paraphrased_sentence)
        
        print(f"Original sentence {i+1}: {sentence}")
        print(f"Paraphrased sentence {i+1}: {paraphrased_sentence}")
        print("-" * 50)
    
    return " ".join(paraphrased_sentences)

def construct_paraphrased_subgraph(paraphrased_text, constructor, enable_adaptive_pruning=True, pruning_ratio=0.2):
    """
    Extract keywords from paraphrased text and construct subgraph to return triplets
    
    Args:
        paraphrased_text (str): Paraphrased text
        constructor (subgraph_construction): subgraph_construction instance
        enable_adaptive_pruning (bool): Whether to enable adaptive pruning (same as models.py)
        pruning_ratio (float): Pruning ratio (same as models.py's insert_watermark, default 0.2)
    
    Returns:
        dict: Paraphrased subgraph information (same structure as models.py)
    """
    try:
        print("Constructing subgraph for paraphrased text...")
        
        # Direct delegation in same way as models.py (performance optimization)
        subgraph_info = constructor.build_subgraph_from_text(
            paraphrased_text, 
            enable_adaptive_pruning=enable_adaptive_pruning, 
            pruning_ratio=pruning_ratio  # Use same parameters as models.py's insert_watermark
        )
        
        print(f"Paraphrased subgraph constructed: {len(subgraph_info['subgraph_triples'])} triplets")
        print(f"LLM extracted keywords: {subgraph_info['keywords']}")
        return subgraph_info
        
    except Exception as e:
        print(f"Error constructing paraphrased subgraph: {e}")
        return None

def select_paraphrased_triplets_for_watermarking(paraphrased_subgraph_info, original_keywords=None, ratio=0.2):
    """
    Select triplets containing keywords from paraphrased subgraph with priority and return only ratio amount
    
    Args:
        paraphrased_subgraph_info (dict): Paraphrased subgraph information
        original_keywords (list): Keywords from original text
        ratio (float): Ratio of triplets to select (0.0 ~ 1.0)
    
    Returns:
        list: Selected paraphrased triplets (keyword priority applied, only ratio amount)
    """
    try:
        # Paraphrased subgraph의 triplets
        paraphrased_triplets = paraphrased_subgraph_info.get('subgraph_triples', [])
        paraphrased_keywords = paraphrased_subgraph_info.get('keywords', [])
        
        if not paraphrased_triplets:
            return []
        
        # Use paraphrased keywords if original keywords are not available
        if not original_keywords:
            original_keywords = paraphrased_keywords
        
        print(f"Original keywords: {original_keywords}")
        print(f"Paraphrased keywords: {paraphrased_keywords}")
        
        # Use paraphrased_keywords with priority, fallback to original_keywords if not available
        keywords_to_match = paraphrased_keywords if paraphrased_keywords else original_keywords
        print(f"Keywords to match: {keywords_to_match}")
        
        # Sort triplets by keyword priority
        keyword_priority_triplets = []
        non_keyword_triplets = []
        
        for triple in paraphrased_triplets:
            if len(triple) == 3:
                head_id, relation_id, tail_id = triple
                
                # Check if each element of triple is related to keyword
                is_keyword_related = False
                
                for keyword in keywords_to_match:
                    # Simple string matching (can be extended to more sophisticated matching logic)
                    if (keyword.lower() in str(head_id).lower() or 
                        keyword.lower() in str(relation_id).lower() or 
                        keyword.lower() in str(tail_id).lower()):
                        is_keyword_related = True
                        break
                
                if is_keyword_related:
                    keyword_priority_triplets.append(triple)
                else:
                    non_keyword_triplets.append(triple)
        
        print(f"Keyword-related triplets: {len(keyword_priority_triplets)}")
        print(f"Non-keyword triplets: {len(non_keyword_triplets)}")
        
        # Select keyword-containing triplets first, then fill rest with non-keyword triplets
        all_triplets = keyword_priority_triplets + non_keyword_triplets
        
        # Select only ratio amount
        target_count = max(1, int(len(all_triplets) * ratio))
        selected_triplets = all_triplets[:target_count]
        
        print(f"Total available triplets: {len(all_triplets)}")
        print(f"Target count (ratio {ratio}): {target_count}")
        print(f"Selected {len(selected_triplets)} triplets with keyword priority")
        
        return selected_triplets
        
    except Exception as e:
        print(f"Error selecting paraphrased triplets: {e}")
        return []

class OptimizedParaphrasingAttack:
    """
    Performance-optimized paraphrasing attack class
    Reuses LLM instance and subgraph_construction in same way as models.py
    """
    
    def __init__(self, llm_model, ratio=0.15, device_id=None, pruning_ratio=0.2):
        self.llm_model = llm_model
        self.ratio = ratio
        self.device_id = device_id
        self.pruning_ratio = pruning_ratio  # Same default value as models.py's insert_watermark
        
        # Create LLM instance (create once and reuse)
        print(f"Initializing LLM instance for {llm_model}...")
        self.llm_instance = LLM(llm_model, device_id=device_id)
        
        # KG 데이터 경로 설정
        kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
        kg_entity_path = f"{kg_root_path}/entities.txt"
        kg_relation_path = f"{kg_root_path}/relations.txt"
        kg_triple_path = f"{kg_root_path}/triplets.txt"
        
        print(f"KG paths:")
        print(f"  Root: {kg_root_path}")
        print(f"  Entity: {kg_entity_path}")
        print(f"  Relation: {kg_relation_path}")
        print(f"  Triple: {kg_triple_path}")
        
        # Initialize subgraph_construction (pass LLM instance, create only once)
        print("Initializing subgraph_construction...")
        self.constructor = subgraph_construction(
            self.llm_instance, 
            ratio=ratio,
            kg_entity_path=kg_entity_path,
            kg_relation_path=kg_relation_path,
            kg_triple_path=kg_triple_path,
            device_id=device_id
        )
        
        print("✓ OptimizedParaphrasingAttack initialized successfully!")
    
    def process_item(self, item):
        """
        Process single item (performance optimized)
        """
        original_text = item.get("original_text", "")
        watermarked_text = item.get("watermarked_text", "")
        
        # Perform paraphrasing only on watermarked_text
        print("Paraphrasing watermarked text...")
        paraphrased_watermarked = paraphrase_text_sentence_by_sentence(watermarked_text)
        
        # Construct new subgraph from paraphrased text (using optimized constructor)
        # Use same parameters as models.py's insert_watermark
        item_pruning_ratio = item.get("pruning_ratio", self.pruning_ratio)
        paraphrased_subgraph_info = construct_paraphrased_subgraph(
            paraphrased_watermarked, 
            self.constructor,
            enable_adaptive_pruning=True,
            pruning_ratio=item_pruning_ratio  # Use original item's pruning_ratio (fallback to class default if not available)
        )
        
        # Select triplets from paraphrased subgraph (apply keyword priority)
        selected_paraphrased_triplets = []
        if paraphrased_subgraph_info:
            # Pass original keywords to select triplets with keyword priority
            original_keywords = item.get("keywords", [])
            selected_paraphrased_triplets = select_paraphrased_triplets_for_watermarking(
                paraphrased_subgraph_info, original_keywords, self.ratio
            )
        
        # Create new item with paraphrased watermarked text and its subgraph
        new_item = {
            "original_text": original_text,
            "watermarked_text": watermarked_text,
            "paraphrased_watermarked": paraphrased_watermarked,
            "keywords": item.get("keywords", []),
            "ratio": item.get("ratio", 0.2),
            "total_triplets": item.get("total_triplets", 0),
            "used_triplets": item.get("used_triplets", 0),
            "triplet_usage_ratio": item.get("triplet_usage_ratio", 0.0),
            "modified_sentences": item.get("modified_sentences", 0),
            "inserted_sentences": item.get("inserted_sentences", 0),
            "pruning_ratio": item.get("pruning_ratio", 0.3)
        }
        
        # Add original subgraph information
        subgraph_triples = item.get("subgraph_triples", [])
        if subgraph_triples:
            new_item["subgraph_triples"] = subgraph_triples
            new_item["selected_triplets"] = item.get("selected_triplets", [])
        
        # Add paraphrased subgraph information
        if paraphrased_subgraph_info:
            # Basic subgraph information
            new_item["paraphrased_subgraph_triples"] = paraphrased_subgraph_info.get("subgraph_triples", [])
            new_item["paraphrased_keywords"] = paraphrased_subgraph_info.get("keywords", [])
            
            # Additional information (optional)
            if "seed_entities" in paraphrased_subgraph_info:
                new_item["paraphrased_seed_entities"] = paraphrased_subgraph_info.get("seed_entities", [])
            if "subgraph_nodes" in paraphrased_subgraph_info:
                new_item["paraphrased_subgraph_nodes"] = paraphrased_subgraph_info.get("subgraph_nodes", [])            
            if "adaptive_pruning_enabled" in paraphrased_subgraph_info:
                new_item["paraphrased_adaptive_pruning_enabled"] = paraphrased_subgraph_info.get("adaptive_pruning_enabled", True)
            if "pruning_ratio" in paraphrased_subgraph_info:
                new_item["paraphrased_pruning_ratio"] = paraphrased_subgraph_info.get("pruning_ratio", 0.3)
            if "quality_info" in paraphrased_subgraph_info:
                new_item["paraphrased_quality_info"] = paraphrased_subgraph_info.get("quality_info", {})
            
            # Add selected paraphrased triplets
            new_item["selected_paraphrased_triplets"] = selected_paraphrased_triplets
        
        return new_item

def main():
    # Set file paths
    # "gpt-4", "llama-3-8b", "llama-3-8b-inst", "mistral-7b-inst", "Qwen2_5-7b-inst"
    input_file = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-GraphMark_50.jsonl"

    # Auto-generate output filename based on input filename
    base_name = input_file.replace('.jsonl', '')
    output_file = f"{base_name}_paraphrased.jsonl"
    
    # Set LLM model
    LLM_MODEL = "llama-3-8b"
    
    print("Loading input JSONL file...")
    data = load_jsonl(input_file)
    print(f"Total items in input file: {len(data)}")
        
    # Initialize performance-optimized paraphrasing attack class (only once)
    print("Initializing optimized paraphrasing attack...")
    attack_processor = OptimizedParaphrasingAttack(
        llm_model=LLM_MODEL,
        ratio=0.50,
        device_id=None,
        pruning_ratio=0.2  # Same default value as models.py's insert_watermark
    )
    
    # Load existing output file if it exists
    paraphrased_data = []
    start_index = 0
    existing_count = 0
    
    if os.path.exists(output_file):
        try:
            paraphrased_data = load_jsonl(output_file)
            existing_count = len(paraphrased_data)
            start_index = existing_count
            print(f"Loaded existing output file with {existing_count} processed items")
        except Exception as e:
            print(f"Could not load existing output file: {e}")
            print("Starting fresh...")
            paraphrased_data = []
            existing_count = 0
            start_index = 0
    
    # Calculate how many more items to process to reach target total
    target_total = len(data)  # Process as many as total input data
    remaining_to_process = max(0, target_total - existing_count)
    
    print(f"Target total: {target_total}")
    print(f"Already processed: {existing_count}")
    print(f"Remaining to process: {remaining_to_process}")
    
    if remaining_to_process == 0:
        print(f"Already reached target of {target_total} items. No more processing needed.")
        exit(0)
    
    # Process remaining items using optimized processor
    for idx in range(start_index, start_index + remaining_to_process):
        print(f"\nProcessing item {idx + 1} of {target_total} (remaining: {remaining_to_process - (idx - start_index)})...")
        
        item = data[idx]
        
        # Process item with optimized processor
        new_item = attack_processor.process_item(item)
        
        # Add to paraphrased data
        paraphrased_data.append(new_item)
        
        # Save output file after each item (incremental saving)
        save_jsonl(paraphrased_data, output_file)
        
        print(f"Completed item {idx + 1}")
        print("=" * 80)
    
    print(f"All processing completed! Total items processed: {len(paraphrased_data)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 
import os
import json
import argparse
import torch
from models import subgraph_construction, LLM

# Default KG paths
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"

def load_data(file_path, max_samples=None):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    return data

def save_data_with_original_subgraph(data, output_path):
    """Save data with original subgraph information to JSONL file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"Data with original subgraph information saved to {output_path}")

def build_original_subgraphs(data, subgraph_constructor, max_samples=None):
    """Build original subgraphs for all data items"""
    print(f"Building original subgraphs for {len(data)} items...")
    
    processed_count = 0
    cached_count = 0
    error_count = 0
    
    for i, data_item in enumerate(data):
        if max_samples and i >= max_samples:
            break
            
        if i % 10 == 0:
            print(f"  Processing item {i+1}/{len(data)}")
        
        # Check if original subgraph already exists
        if 'original_subgraph_triples' in data_item and data_item['original_subgraph_triples']:
            cached_count += 1
            print(f"  ✓ Item {i+1}: Using cached original subgraph ({len(data_item['original_subgraph_triples'])} triplets)")
            continue
        
        # Build original subgraph from original text
        original_text = data_item.get('original_text', '')
        if not original_text:
            print(f"  ⚠ Item {i+1}: No original text found")
            error_count += 1
            continue
        
        try:
            # Construct subgraph from original text
            subgraph_result = subgraph_constructor.build_subgraph_from_text(original_text)
            constructed_triplets = subgraph_result.get('subgraph_triples', [])
            
            # Store original subgraph information
            data_item.update({
                'original_keywords': subgraph_result.get('keywords', []),
                'original_subgraph_triples': constructed_triplets,
                'original_subgraph_nodes': subgraph_result.get('subgraph_nodes', []),
                'original_seed_entities': subgraph_result.get('seed_entities', {}),
                'original_quality_info': subgraph_result.get('quality_info', {})
            })
            
            processed_count += 1
            print(f"  ✓ Item {i+1}: Built original subgraph with {len(constructed_triplets)} triplets")
            
        except Exception as e:
            print(f"  ✗ Item {i+1}: Error building subgraph: {e}")
            error_count += 1
            continue
    
    print(f"\nSubgraph construction summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Cached: {cached_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {processed_count + cached_count + error_count}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Build Original Subgraphs for KG Watermark Detection')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_path', type=str, help='Path to output JSONL file (default: input_path with _with_original_subgraph suffix)')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to process')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--model_name', type=str, help='LLM model name (auto-detect from data_path if not provided)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_path, args.max_samples)
    print(f"Loaded {len(data)} samples")
    
    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        # Auto-detect from data path
        data_filename = os.path.basename(args.data_path)
        if "Qwen-2.5-7b-inst" in data_filename:
            model_name = "Qwen2_5-7b_inst"
        elif "llama-3-8b-inst" in data_filename:
            model_name = "llama-3-8b-inst"
        elif "mistral-7b-inst" in data_filename:
            model_name = "mistral-7b-inst"
        else:
            model_name = "llama-3-8b"  # default fallback
            print(f"Warning: Could not determine model from data path, using default: {model_name}")
    
    print(f"Using LLM model: {model_name}")
    
    # Initialize LLM and subgraph constructor
    print("Initializing models...")
    llm_instance = LLM(model_name, device_id=args.device_id)
    subgraph_constructor = subgraph_construction(
        llm_instance, 
        ratio=0.1,
        kg_entity_path=KG_ENTITY_PATH, 
        kg_relation_path=KG_RELATION_PATH, 
        kg_triple_path=KG_TRIPLE_PATH
    )
    subgraph_constructor.load_kg(KG_ENTITY_PATH, KG_RELATION_PATH, KG_TRIPLE_PATH)
    print("Models initialized successfully")
    
    # Build original subgraphs
    data_with_subgraphs = build_original_subgraphs(data, subgraph_constructor, args.max_samples)
    
    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        # Create output path with _with_original_subgraph suffix
        base_path = args.data_path.replace('.jsonl', '')
        output_path = f"{base_path}_with_original_subgraph.jsonl"
    
    # Save data with original subgraph information
    print(f"\nSaving data with original subgraph information...")
    save_data_with_original_subgraph(data_with_subgraphs, output_path)
    
    print(f"\nCompleted! Original subgraphs saved to: {output_path}")

if __name__ == "__main__":
    main()

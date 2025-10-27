import json
import os
import sys
from models import subgraph_construction, LLM

# Default KG paths (same as in models.py)
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"

# LLM model name for demonstration (can be changed)
LLM_MODEL = "llama-3-8b-chat"

# Input and output file paths
INPUT_FILE = "/home/wooseok/KG_Mark/outputs/test/opengen_adaptive_pruning_paraphrased_100.jsonl"
OUTPUT_FILE = "/home/wooseok/KG_Mark/outputs/test/opengen_adaptive_pruning_paraphrased_with_subgraphs_100.jsonl"

def construct_subgraph_from_keywords(keywords, sg):
    """
    Given keywords and a subgraph_construction instance, construct a subgraph.
    Returns a dictionary with keywords, matched entities, subgraph nodes, and triples.
    """
    print(f"\n[Step 1] Using provided keywords: {keywords}")

    print("\n[Step 2] Matching entities...")
    matched_entities = sg.get_matching_entities(keywords)
    print(f"Matched entities: {matched_entities}")

    print("\n[Step 3] Getting entity embeddings...")
    entity_embeddings = sg.get_kepler_embeddings_for_matched_entities(matched_entities)
    print(f"Retrieved {len(entity_embeddings)} entity embeddings.")

    print("\n[Step 4] Constructing subgraph (semantic bridge)...")
    subgraph_nodes = sg.construct_subgraph_semantic_bridge(
        matched_entities, entity_embeddings, top_k=50, similarity_threshold=0.7, virtual_edge_ratio=0.1
    )
    print(f"Subgraph nodes: {len(subgraph_nodes)}")

    print("\n[Step 5] Extracting triples for subgraph...")
    subgraph_triples = sg.get_subgraph_triples(subgraph_nodes)
    print(f"Subgraph triples: {len(subgraph_triples)}")

    return {
        'keywords': keywords,
        'matched_entities': matched_entities,
        'subgraph_nodes': list(subgraph_nodes),
        'subgraph_triples': list(subgraph_triples.keys()),
        'num_subgraph_nodes': len(subgraph_nodes),
        'num_subgraph_triples': len(subgraph_triples)
    }

def extract_keywords_from_text(text, llm):
    """
    Extract keywords from text using LLM.
    Returns a list of keywords.
    """
    try:
        prompt = f"""Extract 3-5 key entities or concepts from the following text. 
        Focus on named entities, important concepts, and key terms.
        Return only the keywords separated by commas, no explanations.
        
        Text: {text}
        
        Keywords:"""
        
        response = llm.generate(prompt)
        keywords = [kw.strip() for kw in response.split(',')]
        return keywords[:5]  # Limit to 5 keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def main():
    print("Loading LLM and subgraph_construction model...")
    llm = LLM(LLM_MODEL)
    sg = subgraph_construction(
        llm,
        kg_entity_path=KG_ENTITY_PATH,
        kg_relation_path=KG_RELATION_PATH,
        kg_triple_path=KG_TRIPLE_PATH
    )
    
    print(f"Processing file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Process each entry in the JSONL file and append subgraph information
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line.strip())
                
                print(f"\n{'='*80}")
                print(f"Processing Entry {i+1}")
                print(f"{'='*80}")
                
                # Check if subgraph_triples already exists
                existing_subgraph_triples = data.get('subgraph_triples', [])
                
                if existing_subgraph_triples:
                    print(f"[Entry {i+1}] subgraph_triples already exists ({len(existing_subgraph_triples)} triples), skipping original subgraph construction")
                    # Keep existing subgraph data
                    original_subgraph_result = {
                        'keywords': data.get('keywords', []),
                        'matched_entities': data.get('matched_entities', []),
                        'subgraph_nodes': data.get('subgraph_nodes', []),
                        'subgraph_triples': existing_subgraph_triples,
                        'num_subgraph_nodes': data.get('num_subgraph_nodes', 0),
                        'num_subgraph_triples': len(existing_subgraph_triples)
                    }
                else:
                    # Extract original keywords from the entry
                    original_keywords = data.get('keywords', [])
                    if not original_keywords:
                        print(f"[Entry {i+1}] No original keywords found, skipping...")
                        # Still write the original data without subgraph info
                        f_out.write(line)
                        continue
                    
                    print(f"Original keywords: {original_keywords}")
                    
                    # Perform subgraph construction for original keywords
                    original_subgraph_result = construct_subgraph_from_keywords(original_keywords, sg)
                
                # Extract keywords from paraphrased_watermarked text (watermarked_text를 paraphrasing한 것이므로)
                paraphrased_watermarked_text = data.get('paraphrased_watermarked', '')
                if paraphrased_watermarked_text:
                    print(f"\nExtracting keywords from paraphrased_watermarked text...")
                    paraphrased_keywords = extract_keywords_from_text(paraphrased_watermarked_text, llm)
                    print(f"Paraphrased watermarked keywords: {paraphrased_keywords}")
                    
                    if paraphrased_keywords:
                        # Perform subgraph construction for paraphrased watermarked keywords
                        paraphrased_subgraph_result = construct_subgraph_from_keywords(paraphrased_keywords, sg)
                        
                        # Add paraphrased subgraph info with distinct field names
                        data.update({
                            'paraphrased_keywords': paraphrased_keywords,
                            'paraphrased_matched_entities': paraphrased_subgraph_result['matched_entities'],
                            'paraphrased_subgraph_nodes': paraphrased_subgraph_result['subgraph_nodes'],
                            'paraphrased_subgraph_triples': paraphrased_subgraph_result['subgraph_triples'],
                            'paraphrased_num_subgraph_nodes': paraphrased_subgraph_result['num_subgraph_nodes'],
                            'paraphrased_num_subgraph_triples': paraphrased_subgraph_result['num_subgraph_triples']
                        })
                    else:
                        print("No paraphrased watermarked keywords extracted, skipping paraphrased subgraph construction")
                else:
                    print("No paraphrased_watermarked text found, skipping paraphrased subgraph construction")
                
                # Append original subgraph information to the data (either existing or newly constructed)
                data.update(original_subgraph_result)
                
                # Write the enhanced data to output file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                print(f"✅ Entry {i+1} completed:")
                print(f"   - Original subgraph: {original_subgraph_result['num_subgraph_nodes']} nodes, {original_subgraph_result['num_subgraph_triples']} triples")
                if 'paraphrased_num_subgraph_triples' in data:
                    print(f"   - Paraphrased subgraph: {data['paraphrased_num_subgraph_nodes']} nodes, {data['paraphrased_num_subgraph_triples']} triples")
                
            except json.JSONDecodeError as e:
                print(f"[Entry {i+1}] JSON decode error: {e}")
                # Write the original line even if there's an error
                f_out.write(line)
                continue
            except Exception as e:
                print(f"[Entry {i+1}] Error processing entry: {e}")
                # Write the original line even if there's an error
                f_out.write(line)
                continue
    
    print(f"\n{'='*80}")
    print("Processing completed!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
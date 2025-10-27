import openai
import json
import sys
import os
from nltk.tokenize import sent_tokenize
from subgraph_construction import subgraph_construction

# Add parent directory to path to import config and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY
from models import LLM

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_jsonl(file_path):
    """Read a JSONL file and return a list of dicts."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

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
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases sentences while maintaining the original meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        paraphrased = response.choices[0].message.content.strip()
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

def construct_paraphrased_subgraph(paraphrased_text, constructor):
    """
    Paraphrased text에서 keyword를 추출하고 subgraph를 구성하여 triplets 반환
    
    Args:
        paraphrased_text (str): Paraphrased된 텍스트
        constructor (subgraph_construction): subgraph_construction 인스턴스
    
    Returns:
        dict: Paraphrased subgraph 정보 (models.py와 동일한 구조)
    """
    try:
        print("Constructing subgraph for paraphrased text...")
        
        # models.py와 동일한 방식으로 직접 위임 (성능 최적화)
        subgraph_info = constructor.build_subgraph_from_text(
            paraphrased_text, 
            enable_adaptive_pruning=True, 
            pruning_ratio=0.3  # models.py와 동일한 값
        )
        
        print(f"Paraphrased subgraph constructed: {len(subgraph_info['subgraph_triples'])} triplets")
        print(f"LLM extracted keywords: {subgraph_info['keywords']}")
        return subgraph_info
        
    except Exception as e:
        print(f"Error constructing paraphrased subgraph: {e}")
        return None

def select_paraphrased_triplets_for_watermarking(paraphrased_subgraph_info, original_keywords=None, ratio=0.15):
    """
    Paraphrased subgraph에서 keyword가 포함된 triplet들을 우선적으로 선택하고 ratio만큼만 반환
    
    Args:
        paraphrased_subgraph_info (dict): Paraphrased subgraph 정보
        original_keywords (list): 원본 텍스트의 keywords
        ratio (float): 선택할 triplet 비율 (0.0 ~ 1.0)
    
    Returns:
        list: 선택된 paraphrased triplets (keyword 우선순위 적용, ratio만큼만)
    """
    try:
        # Paraphrased subgraph의 triplets
        paraphrased_triplets = paraphrased_subgraph_info.get('subgraph_triples', [])
        paraphrased_keywords = paraphrased_subgraph_info.get('keywords', [])
        
        if not paraphrased_triplets:
            return []
        
        # 원본 keywords가 없으면 paraphrased keywords 사용
        if not original_keywords:
            original_keywords = paraphrased_keywords
        
        print(f"Original keywords: {original_keywords}")
        print(f"Paraphrased keywords: {paraphrased_keywords}")
        
        # paraphrased_keywords를 우선적으로 사용하고, 없으면 original_keywords 사용
        keywords_to_match = paraphrased_keywords if paraphrased_keywords else original_keywords
        print(f"Keywords to match: {keywords_to_match}")
        
        # Keyword 우선순위로 triplet 정렬
        keyword_priority_triplets = []
        non_keyword_triplets = []
        
        for triple in paraphrased_triplets:
            if len(triple) == 3:
                head_id, relation_id, tail_id = triple
                
                # Triple의 각 요소가 keyword와 관련이 있는지 확인
                is_keyword_related = False
                
                for keyword in keywords_to_match:
                    # 간단한 문자열 매칭 (더 정교한 매칭 로직으로 확장 가능)
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
        
        # Keyword가 포함된 triplet들을 먼저 선택하고, 나머지는 non-keyword triplet으로 채움
        all_triplets = keyword_priority_triplets + non_keyword_triplets
        
        # ratio만큼만 선택
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
    성능 최적화된 paraphrasing attack 클래스
    models.py와 동일한 방식으로 LLM 인스턴스와 subgraph_construction을 재사용
    """
    
    def __init__(self, llm_model, ratio=0.15, device_id=None):
        self.llm_model = llm_model
        self.ratio = ratio
        self.device_id = device_id
        
        # LLM 인스턴스 생성 (한 번만 생성하여 재사용)
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
        
        # subgraph_construction 초기화 (LLM 인스턴스 전달, 한 번만 생성)
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
        단일 아이템 처리 (성능 최적화)
        """
        original_text = item.get("original_text", "")
        watermarked_text = item.get("watermarked_text", "")
        
        # watermarked_text에 대해서만 paraphrasing 진행
        print("Paraphrasing watermarked text...")
        paraphrased_watermarked = paraphrase_text_sentence_by_sentence(watermarked_text)
        
        # Paraphrased text에서 새로운 subgraph 구성 (최적화된 constructor 사용)
        paraphrased_subgraph_info = construct_paraphrased_subgraph(paraphrased_watermarked, self.constructor)
        
        # Paraphrased subgraph에서 triplets 선택 (keyword 우선순위 적용)
        selected_paraphrased_triplets = []
        if paraphrased_subgraph_info:
            # 원본 keywords를 전달하여 keyword 우선순위로 triplet 선택
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
            "ratio": item.get("ratio", 0.15),
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
            # 기본 subgraph 정보
            new_item["paraphrased_subgraph_triples"] = paraphrased_subgraph_info.get("subgraph_triples", [])
            new_item["paraphrased_keywords"] = paraphrased_subgraph_info.get("keywords", [])
            
            # 추가 정보들 (선택적)
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
    # 파일 경로 설정
    # "gpt-4", "llama-3-8b", "llama-3-8b-inst", "mistral-7b-inst", "Qwen2_5-7b_inst"
    input_file = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_GraphMark_15.jsonl"
    output_file = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_GraphMark_paraphrased_15.jsonl"
    
    # LLM 모델 설정
    LLM_MODEL = "Qwen2_5-7b_inst"
    
    print("Loading input JSONL file...")
    data = load_jsonl(input_file)
    print(f"Total items in input file: {len(data)}")
    
    # 성능 최적화된 paraphrasing attack 클래스 초기화 (한 번만)
    print("Initializing optimized paraphrasing attack...")
    attack_processor = OptimizedParaphrasingAttack(
        llm_model=LLM_MODEL,
        ratio=0.15,
        device_id=None
    )
    
    # Load existing output file if it exists
    paraphrased_data = []
    start_index = 0
    
    if os.path.exists(output_file):
        try:
            paraphrased_data = load_jsonl(output_file)
            start_index = len(paraphrased_data)
            print(f"Loaded existing output file with {len(paraphrased_data)} items")
            print(f"Starting from index {start_index} (next item to process)")
        except Exception as e:
            print(f"Could not load existing output file: {e}")
            print("Starting fresh...")
            paraphrased_data = []
            start_index = 0
    
    # Process remaining items using optimized processor
    for idx in range(start_index, len(data)):
        print(f"\nProcessing item {idx + 1} of {len(data)} (remaining: {len(data) - idx})...")
        
        item = data[idx]
        
        # 최적화된 processor로 아이템 처리
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
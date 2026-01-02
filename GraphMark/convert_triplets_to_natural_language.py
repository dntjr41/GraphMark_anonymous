"""
Triplet ID를 자연어 형태로 변환하는 스크립트

JSONL 파일의 selected_triplets와 subgraph_triples를 자연어 형태로 변환합니다.
subgraph_construction.py의 load_kg 방식을 참고하여 구현했습니다.
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TripletConverter:
    """
    Triplet ID를 자연어 형태로 변환하는 클래스
    subgraph_construction.py의 load_kg 방식을 참고
    """
    
    def __init__(self, kg_entity_path: str = None, kg_relation_path: str = None):
        """
        Triplet 변환기 초기화
        
        Args:
            kg_entity_path: Entity 파일 경로 (기본값: kg/processed_wikidata5m/entities.txt)
            kg_relation_path: Relation 파일 경로 (기본값: kg/processed_wikidata5m/relations.txt)
        """
        # 기본 경로 설정 (subgraph_construction.py와 동일한 방식)
        if kg_entity_path is None:
            kg_entity_path = "kg/processed_wikidata5m/entities.txt"
        if kg_relation_path is None:
            kg_relation_path = "kg/processed_wikidata5m/relations.txt"
        
        # 절대 경로로 변환
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(kg_entity_path):
            kg_entity_path = os.path.join(project_root, kg_entity_path)
        if not os.path.isabs(kg_relation_path):
            kg_relation_path = os.path.join(project_root, kg_relation_path)
        
        print(f"Loading entity mappings from: {kg_entity_path}")
        print(f"Loading relation mappings from: {kg_relation_path}")
        
        # Entity와 Relation 데이터 로드 (subgraph_construction.py의 load_kg 방식 사용)
        self.entity = {}
        self.relation = {}
        
        self._load_kg(kg_entity_path, kg_relation_path)
        
        print(f"✅ Loaded {len(self.entity)} entities and {len(self.relation)} relations")
    
    def _load_kg(self, kg_entity_path: str, kg_relation_path: str):
        """
        KG 데이터 로드 (subgraph_construction.py의 load_kg 방식과 동일)
        """
        # Entity 로드
        if not os.path.exists(kg_entity_path):
            print(f"⚠️  Warning: Entity file not found: {kg_entity_path}")
        else:
            with open(kg_entity_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    entity_id = parts[0]
                    entity_name = parts[1:] if len(parts) > 1 else [entity_id]
                    # subgraph_construction.py와 동일한 형태로 저장
                    self.entity[entity_id] = {"entity": entity_name}
        
        # Relation 로드
        if not os.path.exists(kg_relation_path):
            print(f"⚠️  Warning: Relation file not found: {kg_relation_path}")
        else:
            with open(kg_relation_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    relation_id = parts[0]
                    relation_name = parts[1:] if len(parts) > 1 else [relation_id]
                    # subgraph_construction.py와 동일한 형태로 저장
                    self.relation[relation_id] = {"id": relation_id, "name": relation_name}
    
    def _select_best_name(self, item_id: str, data_dict: Dict, keywords: List[str] = None) -> str:
        """
        Entity 또는 Relation ID에서 가장 적합한 이름 선택
        models.py의 _select_best_name 방식을 참고
        
        Args:
            item_id: Entity 또는 Relation ID
            data_dict: Entity 또는 Relation 딕셔너리
            keywords: 키워드 리스트 (선택사항)
            
        Returns:
            선택된 이름
        """
        try:
            # 1. 데이터에서 이름 목록 추출
            if item_id not in data_dict:
                return str(item_id)
            
            item_data = data_dict[item_id]
            if isinstance(item_data, dict):
                # Entity의 경우: {"entity": [name1, name2, ...]}
                # Relation의 경우: {"id": relation_id, "name": [name1, name2, ...]}
                if "entity" in item_data:
                    names = item_data["entity"]
                elif "name" in item_data:
                    names = item_data["name"]
                else:
                    names = [str(item_id)]
            else:
                names = item_data if isinstance(item_data, list) else [str(item_id)]
            
            if not names:
                return str(item_id)
            
            # 2. 키워드가 있는 경우 키워드와 정확히 일치하는 이름 우선 선택
            if keywords:
                for keyword in keywords:
                    for name in names:
                        if isinstance(name, str) and keyword.lower() == name.lower():
                            return keyword
                    for name in names:
                        if isinstance(name, str) and (keyword.lower() in name.lower() or name.lower() in keyword.lower()):
                            return keyword
            
            # 3. 영어 이름 우선 필터링
            english_names = [name for name in names if isinstance(name, str) and self._is_english_text(name)]
            candidate_names = english_names if english_names else [name for name in names if isinstance(name, str)]
            
            if not candidate_names:
                return str(item_id)
            
            # 4. 품질 기준으로 최적 이름 선택
            best_name = candidate_names[0]
            for name in candidate_names:
                if self._is_better_name(name, best_name):
                    best_name = name
            
            return best_name
            
        except Exception as e:
            print(f"Error in _select_best_name for {item_id}: {e}")
            return str(item_id)
    
    def _is_english_text(self, text: str) -> bool:
        """텍스트가 영어인지 확인"""
        if not text or not isinstance(text, str):
            return False
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return False
        english_ratio = english_chars / total_chars
        return english_ratio >= 0.8
    
    def _is_better_name(self, name1: str, name2: str) -> bool:
        """두 이름 중 더 나은 이름 선택"""
        # 영어 우선
        is_english1 = self._is_english_text(name1)
        is_english2 = self._is_english_text(name2)
        if is_english1 and not is_english2:
            return True
        if is_english2 and not is_english1:
            return False
        
        # 적절한 길이 (너무 짧거나 길지 않음)
        len1, len2 = len(name1), len(name2)
        if 3 <= len1 <= 50 and not (3 <= len2 <= 50):
            return True
        if 3 <= len2 <= 50 and not (3 <= len1 <= 50):
            return False
        
        # 특수문자 최소화
        special1 = sum(1 for c in name1 if not c.isalnum() and c != ' ')
        special2 = sum(1 for c in name2 if not c.isalnum() and c != ' ')
        if special1 < special2:
            return True
        if special2 < special1:
            return False
        
        # 짧은 이름 우선 (동일 조건일 때)
        return len1 < len2
    
    def convert_triplet_to_natural_language(self, triplet: List[str], keywords: List[str] = None) -> Dict:
        """
        단일 triplet을 자연어 형태로 변환
        
        Args:
            triplet: [head_id, relation_id, tail_id] 형태의 triplet
            keywords: 키워드 리스트 (선택사항)
            
        Returns:
            {
                "head": "entity_name",
                "relation": "relation_name",
                "tail": "entity_name",
                "natural_language": "head relation tail"
            }
        """
        if len(triplet) < 3:
            return {
                "head": str(triplet[0]) if len(triplet) > 0 else "",
                "relation": str(triplet[1]) if len(triplet) > 1 else "",
                "tail": str(triplet[2]) if len(triplet) > 2 else "",
                "natural_language": " ".join(str(x) for x in triplet)
            }
        
        head_id, relation_id, tail_id = triplet[0], triplet[1], triplet[2]
        
        # subgraph_construction.py의 entity와 relation 딕셔너리 사용
        head_name = self._select_best_name(head_id, self.entity, keywords)
        relation_name = self._select_best_name(relation_id, self.relation, keywords)
        tail_name = self._select_best_name(tail_id, self.entity, keywords)
        
        natural_language = f"{head_name} {relation_name} {tail_name}"
        
        return {
            "head": head_name,
            "relation": relation_name,
            "tail": tail_name,
            "natural_language": natural_language
        }
    
    def convert_triplets_list(self, triplets: List[List[str]], keywords: List[str] = None) -> List[Dict]:
        """
        Triplet 리스트를 자연어 형태로 변환
        
        Args:
            triplets: Triplet 리스트
            keywords: 키워드 리스트 (선택사항)
            
        Returns:
            자연어 형태로 변환된 triplet 리스트
        """
        converted = []
        for triplet in triplets:
            converted.append(self.convert_triplet_to_natural_language(triplet, keywords))
        return converted
    
    def process_jsonl_file(self, 
                         input_file: str,
                         output_file: str,
                         convert_subgraph: bool = True,
                         convert_selected: bool = True):
        """
        JSONL 파일의 모든 샘플에 대해 triplet을 자연어로 변환
        
        Args:
            input_file: 입력 JSONL 파일 경로
            output_file: 출력 JSONL 파일 경로
            convert_subgraph: subgraph_triples도 변환할지 여부
            convert_selected: selected_triplets도 변환할지 여부
        """
        # 경로 정규화
        input_file = os.path.abspath(os.path.expanduser(input_file))
        output_file = os.path.abspath(os.path.expanduser(output_file))
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"\n{'='*80}")
        print(f"Converting triplets to natural language")
        print(f"{'='*80}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Convert subgraph_triples: {convert_subgraph}")
        print(f"Convert selected_triplets: {convert_selected}")
        print(f"{'='*80}\n")
        
        total_samples = 0
        processed_samples = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                total_samples += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue
                
                # 키워드 추출 (있는 경우)
                keywords = data.get('keywords', [])
                
                # 새 데이터 구조 생성
                new_data = {
                    "original_text": data.get("original_text", ""),
                    "watermarked_text": data.get("watermarked_text", ""),
                }
                
                # subgraph_triples 변환
                if convert_subgraph and "subgraph_triples" in data:
                    subgraph_triplets = data["subgraph_triples"]
                    if subgraph_triplets:
                        new_data["subgraph_triples"] = self.convert_triplets_list(subgraph_triplets, keywords)
                    else:
                        new_data["subgraph_triples"] = []
                
                # selected_triplets 변환
                if convert_selected and "selected_triplets" in data:
                    selected_triplets = data["selected_triplets"]
                    if selected_triplets:
                        new_data["selected_triplets"] = self.convert_triplets_list(selected_triplets, keywords)
                    else:
                        new_data["selected_triplets"] = []
                
                # 기타 필드 유지 (선택사항)
                for key in ["keywords", "ratio", "total_triplets", "used_triplets", 
                           "planned_modify", "planned_insert", "actual_modified_sentences",
                           "actual_inserted_sentences", "modification_ratio", "insertion_ratio",
                           "length_increase_ratio", "original_length", "watermarked_length"]:
                    if key in data:
                        new_data[key] = data[key]
                
                # JSONL로 저장
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                processed_samples += 1
                
                if processed_samples % 50 == 0:
                    print(f"Processed {processed_samples} samples...")
        
        print(f"\n{'='*80}")
        print(f"Conversion completed!")
        print(f"Total samples: {total_samples}")
        print(f"Processed samples: {processed_samples}")
        print(f"Output saved to: {output_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert triplet IDs to natural language format"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        '--kg_entity_path',
        type=str,
        default=None,
        help="Path to entity file (default: kg/processed_wikidata5m/entities.txt)"
    )
    parser.add_argument(
        '--kg_relation_path',
        type=str,
        default=None,
        help="Path to relation file (default: kg/processed_wikidata5m/relations.txt)"
    )
    parser.add_argument(
        '--convert_subgraph',
        action='store_true',
        default=True,
        help="Convert subgraph_triples to natural language (default: True)"
    )
    parser.add_argument(
        '--no_convert_subgraph',
        dest='convert_subgraph',
        action='store_false',
        help="Do not convert subgraph_triples"
    )
    parser.add_argument(
        '--convert_selected',
        action='store_true',
        default=True,
        help="Convert selected_triplets to natural language (default: True)"
    )
    parser.add_argument(
        '--no_convert_selected',
        dest='convert_selected',
        action='store_false',
        help="Do not convert selected_triplets"
    )
    
    args = parser.parse_args()
    
    # Triplet 변환기 초기화
    converter = TripletConverter(
        kg_entity_path=args.kg_entity_path,
        kg_relation_path=args.kg_relation_path
    )
    
    # 파일 처리
    converter.process_jsonl_file(
        input_file=args.input_file,
        output_file=args.output_file,
        convert_subgraph=args.convert_subgraph,
        convert_selected=args.convert_selected
    )
    
    print("✅ Conversion completed successfully!")


if __name__ == "__main__":
    main()


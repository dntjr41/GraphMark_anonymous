from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import torch
import spacy
from difflib import get_close_matches
# from models import LLM  # 순환 import 제거
from igraph import Graph
import numpy as np
import random

# igraph 기반 그래프 생성 함수
def build_igraph(triples):
    nodes = set()
    edges = []
    for (head, relation, tail) in triples:
        nodes.add(head)
        nodes.add(tail)
        edges.append((head, tail))
    node_list = list(nodes)
    node_idx = {node: idx for idx, node in enumerate(node_list)}
    g = Graph(directed=True)
    g.add_vertices(len(node_list))
    g.add_edges([(node_idx[head], node_idx[tail]) for (head, tail) in edges])
    return g, node_list, node_idx

class subgraph_construction():
    def __init__(self, llm, ratio=0.2, kg_entity_path="entities.txt", kg_relation_path="relations.txt", kg_triple_path="triples.txt", device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LLM은 이미 인스턴스로 전달되어야 함 (순환 import 방지)
        if hasattr(llm, 'generate'):  # LLM 인스턴스인지 확인
            self.llm = llm
        else:
            raise ValueError("LLM must be an instance with a 'generate' method, not a string")
        
        self.ratio = ratio
        self.topk = 3
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        import os
        # kg_entity_path가 절대 경로인지 확인하고 kg_root_path 계산
        if os.path.isabs(kg_entity_path):
            kg_root_path = os.path.dirname(kg_entity_path)
        else:
            # 상대 경로인 경우 현재 작업 디렉토리 기준으로 절대 경로 계산
            kg_root_path = os.path.abspath(os.path.dirname(kg_entity_path))
        
        print(f"Calculated kg_root_path: {kg_root_path}")
        self.load_pretrained_embeddings(kg_root_path)
        print("Pre-trained embeddings enabled (KEPLER always ON)")
        self.entity, self.relation, self.triple = self.load_kg(kg_entity_path, kg_relation_path, kg_triple_path)
    
    def load_pretrained_embeddings(self, kg_root_path):
        # Entity와 Relation embedding 로드 (새로 학습된 파일들 사용)
        self.entity_embeddings = np.load(f"{kg_root_path}/entity_embeddings_full.npy")
        self.relation_embeddings = np.load(f"{kg_root_path}/relation_embeddings_full.npy")
        
        # Entity ID와 이름 매핑 로드 (embedding 인덱스와 매핑)
        self.entity_id_to_name = {}
        self.entity_name_to_id = {}
        self.entity_id_to_idx = {}  # Entity ID를 embedding 인덱스로 매핑
        
        with open(f"{kg_root_path}/entities.txt", 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                entity_name = parts[1] if len(parts) > 1 else entity_id
                
                self.entity_id_to_name[entity_id] = entity_name
                self.entity_name_to_id[entity_name] = entity_id
                self.entity_id_to_idx[entity_id] = idx  # 인덱스 매핑
        
        print(f"Loaded {len(self.entity_id_to_name)} entities")
        print(f"New entity embeddings size: {self.entity_embeddings.shape[0]} entities")
        print(f"Entity mapping size: {len(self.entity_id_to_idx)} entities")
        
        # 매핑이 일치하는지 확인
        if len(self.entity_id_to_idx) != self.entity_embeddings.shape[0]:
            print(f"Warning: Entity mapping count ({len(self.entity_id_to_idx)}) != embedding count ({self.entity_embeddings.shape[0]})")
        else:
            print("✓ Entity mapping and embeddings are aligned!")
    
    def get_entity_embedding(self, entity_name):
        """Entity 이름으로 embedding 반환"""
        if self.entity_embeddings is None:
            return None
        
        try:
            # Entity 이름으로 ID 찾기
            if entity_name in self.entity_name_to_id:
                entity_id = self.entity_name_to_id[entity_name]
                # Entity ID를 인덱스로 매핑하여 embedding 반환
                if entity_id in self.entity_id_to_idx:
                    entity_idx = self.entity_id_to_idx[entity_id]
                    if entity_idx < len(self.entity_embeddings):
                        return self.entity_embeddings[entity_idx]
                    else:
                        print(f"Entity index {entity_idx} out of range for embeddings (max: {len(self.entity_embeddings)})")
                else:
                    print(f"Entity ID {entity_id} not found in index mapping")
            
            # 직접 매칭이 안되면 유사한 이름 찾기
            for name, entity_id in self.entity_name_to_id.items():
                if entity_name.lower() in name.lower() or name.lower() in entity_name.lower():
                    if entity_id in self.entity_id_to_idx:
                        entity_idx = self.entity_id_to_idx[entity_id]
                        if entity_idx < len(self.entity_embeddings):
                            return self.entity_embeddings[entity_idx]
            
            print(f"Entity '{entity_name}' not found in embeddings")
            return np.zeros(self.entity_embeddings.shape[1])
        except Exception as e:
            print(f"Error getting embedding for '{entity_name}': {e}")
            return np.zeros(self.entity_embeddings.shape[1])
    
    def load_kg(self, kg_entity_path, kg_relation_path, kg_triple_path):
        kg_entity, kg_relation, kg_triple = {}, {}, {}
        
        with open(kg_entity_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                entity_id = parts[0]
                entity_name = parts[1:] if len(parts) > 1 else [entity_id]
                
                kg_entity[entity_id] = {"entity": entity_name}
        
        with open(kg_relation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                relation_id = parts[0]
                relation_name = parts[1:] if len(parts) > 1 else [relation_id]
                
                kg_relation[relation_id] = {"id": relation_id, "name": relation_name}
                
        with open(kg_triple_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    head, relation, tail = parts[0], parts[1], parts[2]
                    kg_triple[(head, relation, tail)] = {"head": head, "relation": relation, "tail": tail}
        
        return kg_entity, kg_relation, kg_triple
    
    def extract_keywords_with_ner(self, text):
        """NER을 사용하여 키워드 추출 (fallback 방법)"""
        try:
            print(f"Using NER fallback for text: {text[:100]}...")
            
            # 텍스트 전처리
            if not text or len(text.strip()) < 10:
                print("Text too short for NER extraction")
                return ["text", "content", "information"][:self.topk]
            
            # spaCy 처리
            try:
                doc = self.nlp(text)
            except Exception as nlp_error:
                print(f"spaCy processing failed: {nlp_error}")
                # 간단한 fallback: 공백으로 분리하고 상위 단어들 선택
                words = text.lower().split()
                stop_words = set(stopwords.words("english"))
                filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
                word_counts = {}
                for word in filtered_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                return [word for word, _ in sorted_words[:self.topk]]
            
            # NER 엔티티 추출
            ner_keywords = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                    keyword = ent.text.strip()
                    if keyword and len(keyword) > 1:
                        ner_keywords.append(keyword.lower())
            
            # 중복 제거
            ner_keywords = list(set(ner_keywords))
            print(f"NER entities found: {ner_keywords}")
            
            # 부족한 경우 명사 추가
            if len(ner_keywords) < self.topk:
                nouns = []
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and len(token.text.strip()) > 2:
                        nouns.append(token.text.strip().lower())
                
                # 중복 제거하고 추가
                nouns = list(set(nouns))
                ner_keywords.extend(nouns)
                print(f"Added nouns: {nouns}")
            
            # Stop words 제거
            stop_words = set(stopwords.words("english"))
            filtered_keywords = [kw for kw in ner_keywords if kw not in stop_words]
            
            # 최종 키워드 선택
            final_keywords = filtered_keywords[:self.topk]
            
            # 부족한 경우 기본 키워드로 보완
            if len(final_keywords) < self.topk:
                default_keywords = ["information", "content", "text", "data", "document"]
                for default_kw in default_keywords:
                    if default_kw not in final_keywords and len(final_keywords) < self.topk:
                        final_keywords.append(default_kw)
            
            print(f"✓ NER extraction completed: {final_keywords}")
            return final_keywords
            
        except Exception as e:
            print(f"Error in NER extraction: {e}")
            # 최후의 fallback: 기본 키워드 반환
            default_keywords = ["text", "content", "information", "data", "document"]
            return default_keywords[:self.topk]
    
    def extract_keywords_with_llm(self, text):
        """
        LLM을 사용하여 키워드 추출 (더 안정적이고 일관성 있는 결과)
        """
        # 간단하고 명확한 프롬프트 사용
        prompt = f"""Extract {self.topk} keywords from this text. Return only keywords separated by commas.

        Text: {text}

        Keywords:"""

        try:
            print(f"Attempting LLM keyword extraction for text: {text[:100]}...")
            
            # LLM 호출 시도
            try:
                extracted_keywords = self.llm.generate(
                    prompt, 
                    max_tokens=50,  # 키워드만 필요하므로 토큰 수 최소화
                    temperature=0.1  # 낮은 temperature로 일관성 향상
                )
                print(f"LLM raw response: '{extracted_keywords}'")
            except Exception as llm_error:
                print(f"LLM generation failed: {llm_error}")
                print("Falling back to NER method")
                return self.extract_keywords_with_ner(text)
            
            # 응답 검증
            if not extracted_keywords or not isinstance(extracted_keywords, str):
                print("LLM returned invalid response, falling back to NER")
                return self.extract_keywords_with_ner(text)
            
            # 응답 정리
            extracted_keywords = extracted_keywords.strip()
            if not extracted_keywords:
                print("LLM returned empty response, falling back to NER")
                return self.extract_keywords_with_ner(text)
            
            # 키워드 추출 시도
            try:
                # 여러 줄 응답 처리
                lines = [line.strip() for line in extracted_keywords.split('\n') if line.strip()]
                
                # 키워드가 있는 줄 찾기
                keyword_line = None
                for line in reversed(lines):
                    if line and not any(marker in line.lower() for marker in ["text:", "keywords:", "extract", "analyze", "assistant"]):
                        keyword_line = line
                        break
                
                if not keyword_line:
                    print("No valid keyword line found in LLM response, falling back to NER")
                    return self.extract_keywords_with_ner(text)
                
                # 쉼표로 분리
                raw_keywords = []
                for keyword in keyword_line.split(","):
                    keyword = keyword.strip()
                    if keyword and len(keyword) > 0 and len(keyword) <= 50:
                        # 특수문자 제거
                        keyword = keyword.replace('"', '').replace("'", "").replace('(', '').replace(')', '')
                        if keyword:
                            raw_keywords.append(keyword)
                
                print(f"Parsed raw keywords: {raw_keywords}")
                
                if not raw_keywords:
                    print("No valid keywords parsed, falling back to NER")
                    return self.extract_keywords_with_ner(text)
                
                # 키워드 필터링
                stop_words = set(stopwords.words("english"))
                filtered_keywords = []
                
                for keyword in raw_keywords:
                    keyword_lower = keyword.lower()
                    # 유효한 키워드인지 확인
                    if (keyword_lower not in stop_words and 
                        len(keyword) > 1 and
                        not any(bad_word in keyword_lower for bad_word in ["extract", "keywords", "text", "analyze"])):
                        filtered_keywords.append(keyword)
                
                print(f"Filtered keywords: {filtered_keywords}")
                
                # 최종 키워드 선택
                if len(filtered_keywords) >= self.topk:
                    final_keywords = filtered_keywords[:self.topk]
                    print(f"✓ LLM extraction successful: {final_keywords}")
                    return final_keywords
                else:
                    print(f"LLM returned insufficient keywords ({len(filtered_keywords)}), supplementing with NER")
                    ner_keywords = self.extract_keywords_with_ner(text)
                    combined_keywords = list(set(filtered_keywords + ner_keywords))
                    final_keywords = combined_keywords[:self.topk]
                    print(f"✓ Combined keywords: {final_keywords}")
                    return final_keywords
                    
            except Exception as parse_error:
                print(f"Error parsing LLM response: {parse_error}")
                print("Falling back to NER method")
                return self.extract_keywords_with_ner(text)
                
        except Exception as e:
            print(f"Unexpected error in LLM keyword extraction: {e}")
            print("Falling back to NER method")
            return self.extract_keywords_with_ner(text)
    
    def get_matching_entities(self, keywords):
        matched_entities = {}
        for keyword in keywords:
            # 각 키워드별로 상위 2개의 엔티티 매칭
            candidates = []
            for entity_id, entity_data in self.entity.items():
                entity_name = entity_data["entity"]
                # get_close_matches는 리스트를 반환, n=5, cutoff=0.9
                similar_matches = get_close_matches(keyword, entity_name, n=5, cutoff=0.85)
                if similar_matches:
                    # difflib SequenceMatcher로 유사도 점수 계산
                    from difflib import SequenceMatcher
                    for candidate in similar_matches:
                        score = SequenceMatcher(None, keyword, candidate).ratio()
                        candidates.append((score, entity_id, entity_name, candidate))
                # 부분 일치도 고려 (더 유연한 매칭)
                if keyword.lower() in entity_name or any(word.lower() in entity_name for word in keyword.split()):
                    score = 0.9
                    candidates.append((score, entity_id, entity_name, keyword))
            
            # 점수 순으로 정렬하고 상위 1개만 선택 (빠른 처리)
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_entities = []
            
            for score, entity_id, entity_name, match in candidates[:1]:  # 상위 1개만
                if entity_id is not None:
                    top_entities.append((entity_id, entity_name))
                    break
            
            if top_entities:
                matched_entities[keyword] = top_entities
                print(f"Keyword '{keyword}' matched entity: {top_entities[0][0]}")
                
        return matched_entities
    
    def get_kepler_embeddings_for_matched_entities(self, matched_entities):
        """
        Matched entities에 대해 기존 학습된 embedding 반환
        Args:
            matched_entities (dict): Keyword와 매칭된 entity 정보
        Returns:
            dict: {entity_id: embedding} 형태의 딕셔너리
        """
        if self.entity_embeddings is None:
            print("Pre-trained embeddings not enabled")
            return {}
        
        entity_embeddings = {}
        
        print(f"Processing {len(matched_entities)} matched entities...")
        
        # Matched entities에서 entity 이름 추출하고 embedding 반환
        for keyword, entities in matched_entities.items():
            print(f"\nProcessing keyword: '{keyword}'")
            for entity_id, entity_name in entities:
                # entity_name이 리스트인 경우 (예: ["Barack", "Obama"])
                if isinstance(entity_name, list):
                    entity_name_str = " ".join(entity_name)
                else:
                    entity_name_str = str(entity_name)
                
                print(f"  Entity: {entity_id} -> {entity_name_str}")
                
                # 디버깅: entity_name_str이 매핑에 있는지 확인
                if entity_name_str in self.entity_name_to_id:
                    print(f"    Found in name_to_id mapping")
                else:
                    print(f"    NOT found in name_to_id mapping")
                
                # 직접 embedding 가져오기
                embedding = self.get_entity_embedding(entity_name_str)
                if embedding is not None and not np.allclose(embedding, 0):
                    entity_embeddings[entity_id] = embedding
                    print(f"    ✓ Retrieved embedding (shape: {embedding.shape})")
                else:
                    print(f"    ✗ No embedding found")
        
        print(f"\nRetrieved {len(entity_embeddings)} pre-trained embeddings")
        return entity_embeddings
    
    def get_entity_matching_with_kepler_embeddings(self, keywords):
        """
        1. Entity Matching: Keyword로 Wikidata5M entity를 찾고
        2. KEPLER Embedding: 각 Seed Node의 벡터 얻기
        """
        # 1. Entity Matching
        matched_entities = self.get_matching_entities(keywords)
        
        # 2. KEPLER Embedding 생성
        entity_embeddings = self.get_kepler_embeddings_for_matched_entities(matched_entities)
        
        return matched_entities, entity_embeddings
    
    def adaptive_pruning(self, subgraph_nodes, seed_entities, entity_embeddings, 
                        base_threshold=0.7, pruning_ratio=0.3, min_nodes=10):
        """
        Adaptive Pruning: Seed node보다 threshold가 낮은 node들에 대해 pruning 수행
        
        Args:
            subgraph_nodes (list): 현재 subgraph의 모든 노드들
            seed_entities (dict): Seed entity 정보
            entity_embeddings (dict): Seed entity embeddings
            base_threshold (float): 기본 similarity threshold
            pruning_ratio (float): Pruning 비율 (0.0 ~ 1.0)
            min_nodes (int): 최소 유지할 노드 수
        
        Returns:
            list: Pruning된 노드 리스트
        """
        print(f"\n=== Adaptive Pruning ===")
        print(f"Initial nodes: {len(subgraph_nodes)}")
        print(f"Base threshold: {base_threshold}")
        print(f"Pruning ratio: {pruning_ratio}")
        
        # 1. Seed nodes 추출 (pruning 대상에서 제외)
        seed_nodes = set()
        for keyword, entities in seed_entities.items():
            for entity_id, entity_name in entities:
                seed_nodes.add(entity_id)
        
        print(f"Seed nodes (protected): {len(seed_nodes)}")
        
        # 2. Non-seed nodes만 pruning 대상으로 선정
        non_seed_nodes = [node for node in subgraph_nodes if node not in seed_nodes]
        print(f"Non-seed nodes (pruning candidates): {len(non_seed_nodes)}")
        
        if len(non_seed_nodes) <= min_nodes:
            print(f"Not enough non-seed nodes for pruning (≤ {min_nodes})")
            return subgraph_nodes
        
        # 3. 각 non-seed node의 adaptive threshold 계산
        node_scores = self._calculate_adaptive_thresholds(
            non_seed_nodes, seed_entities, entity_embeddings, base_threshold
        )
        
        # 4. Adaptive threshold 기반으로 pruning 수행
        pruned_nodes = self._perform_adaptive_pruning(
            non_seed_nodes, node_scores, base_threshold, pruning_ratio, min_nodes
        )
        
        # 5. Seed nodes와 pruned nodes 합치기
        final_nodes = list(seed_nodes) + pruned_nodes
        
        print(f"Pruned nodes: {len(pruned_nodes)}/{len(non_seed_nodes)}")
        print(f"Final nodes after pruning: {len(final_nodes)}")
        
        return final_nodes
    
    def _calculate_adaptive_thresholds(self, non_seed_nodes, seed_entities, entity_embeddings, base_threshold):
        """
        각 non-seed node에 대해 adaptive threshold 계산
        
        Returns:
            dict: {node_id: (adaptive_threshold, similarity_score)}
        """
        node_scores = {}
        
        if not entity_embeddings:
            print("No seed embeddings available for adaptive threshold calculation")
            return node_scores
        
        print(f"Calculating adaptive thresholds for {len(non_seed_nodes)} nodes...")
        
        for node_id in non_seed_nodes:
            node_embedding = self.get_entity_embedding_by_id(node_id)
            if node_embedding is None or np.allclose(node_embedding, 0):
                continue
            
            # Seed embeddings와의 평균 유사도 계산
            similarities = []
            for seed_embedding in entity_embeddings.values():
                similarity = self._compute_cosine_similarity(node_embedding, seed_embedding)
                similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                # Adaptive threshold 계산:
                # - 평균 유사도가 높을수록 더 낮은 threshold 적용
                # - 최대 유사도도 고려하여 threshold 조정
                adaptive_threshold = base_threshold * (1 - avg_similarity * 0.3)
                adaptive_threshold = max(0.3, min(0.9, adaptive_threshold))  # 0.3 ~ 0.9 범위로 제한
                
                node_scores[node_id] = {
                    'adaptive_threshold': adaptive_threshold,
                    'avg_similarity': avg_similarity,
                    'max_similarity': max_similarity,
                    'base_threshold': base_threshold
                }
        
        print(f"Calculated adaptive thresholds for {len(node_scores)} nodes")
        return node_scores
    
    def _perform_adaptive_pruning(self, non_seed_nodes, node_scores, base_threshold, pruning_ratio, min_nodes):
        """
        Adaptive threshold 기반으로 pruning 수행
        """
        if not node_scores:
            return non_seed_nodes[:min_nodes]
        
        # 1. 각 node의 pruning score 계산
        pruning_scores = []
        for node_id in non_seed_nodes:
            if node_id in node_scores:
                score_info = node_scores[node_id]
                
                # Pruning score = (base_threshold - adaptive_threshold) * avg_similarity
                # 높은 pruning score = 더 쉽게 pruning됨
                pruning_score = (base_threshold - score_info['adaptive_threshold']) * score_info['avg_similarity']
                
                pruning_scores.append((node_id, pruning_score, score_info))
            else:
                # Score 정보가 없는 경우 기본값 사용
                pruning_scores.append((node_id, 0.0, None))
        
        # 2. Pruning score 순으로 정렬 (높은 score가 먼저 pruning됨)
        pruning_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Pruning 수행
        num_to_prune = int(len(non_seed_nodes) * pruning_ratio)
        num_to_keep = max(min_nodes, len(non_seed_nodes) - num_to_prune)
        
        kept_nodes = [node_id for node_id, _, _ in pruning_scores[:num_to_keep]]
        pruned_nodes = [node_id for node_id, _, _ in pruning_scores[num_to_keep:]]
        
        print(f"Pruning statistics:")
        print(f"  - Nodes to keep: {len(kept_nodes)}")
        print(f"  - Nodes to prune: {len(pruned_nodes)}")
        
        # 4. Pruning된 노드들의 정보 출력
        if pruned_nodes:
            print(f"  - Pruned nodes (top 5):")
            for i, node_id in enumerate(pruned_nodes[:5]):
                if node_id in node_scores:
                    score_info = node_scores[node_id]
                    print(f"    {i+1}. {node_id}: threshold={score_info['adaptive_threshold']:.3f}, "
                          f"avg_sim={score_info['avg_similarity']:.3f}")
        
        return kept_nodes
    
    def construct_subgraph_semantic_bridge(self, seed_entities, entity_embeddings, top_k=50, similarity_threshold=0.7, virtual_edge_ratio=0.1, enable_adaptive_pruning=True, pruning_ratio=0.3):
        """
        Semantic Bridge 전략: Seed Node 간의 의미적 연결성을 보장하는 Backbone Path 구성
        
        Args:
            seed_entities (dict): Seed entity 정보
            entity_embeddings (dict): Seed entity embeddings
            top_k (int): 최대 노드 수 (seed nodes 제외)
            similarity_threshold (float): 유사도 임계값
            virtual_edge_ratio (float): 가상 엣지 추가 비율
            enable_adaptive_pruning (bool): Adaptive pruning 활성화 여부
            pruning_ratio (float): Pruning 비율 (0.0 ~ 1.0)
        """
        print("Using Semantic Bridge strategy for subgraph construction")
        
        # 1. Seed nodes를 먼저 추가 (무조건 포함)
        seed_nodes = set()
        for keyword, entities in seed_entities.items():
            for entity_id, entity_name in entities:
                seed_nodes.add(entity_id)
        
        print(f"Seed nodes to include: {len(seed_nodes)}")
        
        if not entity_embeddings:
            print("No entity embeddings available, cannot use semantic bridge")
            return list(seed_nodes)
        
        # 2. Graph-based connections 찾기
        graph_nodes = self._find_graph_based_connections(seed_nodes)
        print(f"Found {len(graph_nodes)} graph-based connections")
        
        # 3. Semantic Bridge: Seed 간 의미적 연결성 보장
        semantic_bridge_nodes = self._create_semantic_bridges(
            seed_entities, entity_embeddings, similarity_threshold, virtual_edge_ratio
        )
        print(f"Found {len(semantic_bridge_nodes)} semantic bridge nodes")
        
        # 4. Seed nodes를 제외한 나머지 노드들에서 top_k개 선택
        non_seed_nodes = graph_nodes.union(semantic_bridge_nodes) - seed_nodes
        selected_non_seed_nodes = list(non_seed_nodes)[:top_k]
        
        # 5. Seed nodes와 선택된 non-seed nodes 합치기
        initial_subgraph_nodes = list(seed_nodes) + selected_non_seed_nodes
        
        print(f"Initial subgraph: {len(initial_subgraph_nodes)} nodes (including {len(seed_nodes)} seeds + {len(selected_non_seed_nodes)} others)")
        
        # 6. Adaptive Pruning 적용 (선택적)
        if enable_adaptive_pruning and len(initial_subgraph_nodes) > len(seed_nodes) + 5:
            print(f"\nApplying Adaptive Pruning...")
            final_subgraph_nodes = self.adaptive_pruning(
                initial_subgraph_nodes, seed_entities, entity_embeddings,
                base_threshold=similarity_threshold, pruning_ratio=pruning_ratio
            )
        else:
            print(f"Skipping Adaptive Pruning (disabled or insufficient nodes)")
            final_subgraph_nodes = initial_subgraph_nodes
        print(f"Final subgraph: {len(final_subgraph_nodes)} nodes")
        return final_subgraph_nodes
    
    def _find_graph_based_connections(self, seed_nodes, max_neighbors=10):
        """
        Graph-based 연결 찾기 (기존 KG 구조 활용)
        """
        graph_nodes = set()
        
        for seed_id in seed_nodes:
            # 1-hop neighbors
            neighbors_1 = self._find_directly_connected_entities(seed_id, max_neighbors=max_neighbors)
            graph_nodes.update(neighbors_1)
            
            # 2-hop neighbors (제한적)
            for neighbor in list(neighbors_1)[:3]:  # 상위 3개만
                neighbors_2 = self._find_directly_connected_entities(neighbor, max_neighbors=5)
                graph_nodes.update(neighbors_2)
        
        return graph_nodes
    
    def _find_directly_connected_entities(self, entity_id, max_neighbors=10):
        """
        주어진 entity와 직접 연결된 노드들을 찾기
        """
        connected_nodes = set()
        
        for (head, relation, tail) in self.triple.keys():
            if head == entity_id:
                connected_nodes.add(tail)
            elif tail == entity_id:
                connected_nodes.add(head)
        
        return list(connected_nodes)[:max_neighbors]
    
    def _create_semantic_bridges(self, seed_entities, entity_embeddings, similarity_threshold=0.7, virtual_edge_ratio=0.1):
        """
        Semantic Bridge 생성: Seed Node 간의 의미적 연결성 보장
        
        A. Seed Node 임베딩과 전체 엔티티 임베딩 간 유사도 검색
        B. 유사도 상위 k% 또는 임계값 θ 이상인 쌍을 "virtual edge"로 추가
        C. 최종적으로 top 10개만 선택
        """
        bridge_nodes = set()
        
        if not entity_embeddings or len(entity_embeddings) < 2:
            print("Not enough seed embeddings for semantic bridge")
            return bridge_nodes
        
        # 전체 entity 중에서 샘플링 (성능 최적화)
        all_entity_ids = list(self.entity.keys())
        sample_size = min(1000, len(all_entity_ids))  # 1000개 샘플링
        sampled_entities = random.sample(all_entity_ids, sample_size)
        
        print(f"Searching semantic bridges among {sample_size} entities...")
        
        # 각 seed embedding에 대해 유사한 entity들 찾기
        for seed_id, seed_embedding in entity_embeddings.items():
            similar_entities = self._find_semantically_similar_entities(
                seed_embedding, sampled_entities, similarity_threshold, top_k=20
            )
            bridge_nodes.update(similar_entities)
        
        # Seed 간 직접 연결을 위한 bridge entities 찾기
        seed_connectivity_bridges = self._find_seed_connectivity_bridges(
            entity_embeddings, sampled_entities, similarity_threshold
        )
        bridge_nodes.update(seed_connectivity_bridges)
        
        # 최종적으로 top 10개만 선택 (유사도 기반으로 정렬)
        final_bridge_nodes = self._select_top_bridge_nodes(bridge_nodes, entity_embeddings, top_k=10)
        
        print(f"Selected top {len(final_bridge_nodes)} bridge nodes from {len(bridge_nodes)} candidates")
        return final_bridge_nodes
    
    def _find_semantically_similar_entities(self, query_embedding, candidate_entities, threshold=0.7, top_k=20):
        """
        Query embedding과 의미적으로 유사한 entity들 찾기
        """
        similarities = []
        
        for entity_id in candidate_entities:
            entity_embedding = self.get_entity_embedding_by_id(entity_id)
            if entity_embedding is not None and not np.allclose(entity_embedding, 0):
                similarity = self._compute_cosine_similarity(query_embedding, entity_embedding)
                if similarity >= threshold:
                    similarities.append((entity_id, similarity))
        
        # 유사도 순으로 정렬하고 상위 k개 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in similarities[:top_k]]
    
    def _find_seed_connectivity_bridges(self, entity_embeddings, candidate_entities, similarity_threshold=0.7):
        """
        Seed nodes 간의 연결성을 보장하는 bridge entities 찾기
        """
        bridge_nodes = set()
        seed_embeddings = list(entity_embeddings.values())
        seed_ids = list(entity_embeddings.keys())
        
        # Seed embeddings 간의 유사도 계산
        for i, emb1 in enumerate(seed_embeddings):
            for j, emb2 in enumerate(seed_embeddings[i+1:], i+1):
                similarity = self._compute_cosine_similarity(emb1, emb2)
                if similarity >= similarity_threshold:
                    print(f"Seed nodes {i} and {j} are semantically similar (similarity: {similarity:.3f})")
                    
                    # 두 seed와 모두 유사한 중간 노드들 찾기
                    bridge_candidates = self._find_bridge_entities_between_seeds(
                        seed_ids[i], seed_ids[j], candidate_entities, similarity_threshold
                    )
                    bridge_nodes.update(bridge_candidates)
        
        return bridge_nodes
    
    def _find_bridge_entities_between_seeds(self, seed1_id, seed2_id, candidate_entities, similarity_threshold=0.7):
        """
        두 seed nodes를 연결하는 bridge entities 찾기
        """
        bridge_nodes = set()
        
        # 각 seed의 embedding
        seed1_embedding = self.get_entity_embedding_by_id(seed1_id)
        seed2_embedding = self.get_entity_embedding_by_id(seed2_id)
        
        if seed1_embedding is None or seed2_embedding is None:
            return bridge_nodes
        
        # 두 seed와 모두 유사한 중간 노드들 찾기
        for entity_id in candidate_entities:
            entity_embedding = self.get_entity_embedding_by_id(entity_id)
            if entity_embedding is not None:
                sim1 = self._compute_cosine_similarity(seed1_embedding, entity_embedding)
                sim2 = self._compute_cosine_similarity(seed2_embedding, entity_embedding)
                
                # 두 seed와 모두 유사한 노드들 (bridge 역할)
                if sim1 >= similarity_threshold and sim2 >= similarity_threshold:
                    bridge_nodes.add(entity_id)
        
        return bridge_nodes
    
    def get_entity_embedding_by_id(self, entity_id):
        """
        Entity ID로 직접 embedding 반환
        """
        if self.entity_embeddings is None:
            return None
        
        try:
            if entity_id in self.entity_id_to_idx:
                entity_idx = self.entity_id_to_idx[entity_id]
                if entity_idx < len(self.entity_embeddings):
                    return self.entity_embeddings[entity_idx]
        except Exception as e:
            print(f"Error getting embedding for entity ID {entity_id}: {e}")
        
        return None
    
    def _select_top_bridge_nodes(self, bridge_nodes, entity_embeddings, top_k=10):
        """
        Bridge nodes 중에서 가장 유사한 top_k개 선택
        """
        if not bridge_nodes:
            return set()
        
        # 각 bridge node의 평균 유사도 계산
        node_scores = []
        for node_id in bridge_nodes:
            node_embedding = self.get_entity_embedding_by_id(node_id)
            if node_embedding is not None and not np.allclose(node_embedding, 0):
                # 모든 seed embeddings와의 평균 유사도 계산
                similarities = []
                for seed_embedding in entity_embeddings.values():
                    similarity = self._compute_cosine_similarity(node_embedding, seed_embedding)
                    similarities.append(similarity)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    node_scores.append((node_id, avg_similarity))
        
        # 유사도 순으로 정렬하고 top_k개 선택
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [node_id for node_id, _ in node_scores[:top_k]]
        
        return set(top_nodes)
    
    def _compute_cosine_similarity(self, emb1, emb2):
        """
        두 embedding 간의 cosine similarity 계산
        """
        try:
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def get_subgraph_triples(self, subgraph_nodes):
        """
        Subgraph 노드들에 해당하는 triples 추출
        """
        subgraph_triples = []
        
        for (head, relation, tail) in self.triple.keys():
            if head in subgraph_nodes and tail in subgraph_nodes:
                subgraph_triples.append([head, relation, tail])
        
        print(f"Extracted {len(subgraph_triples)} triples for subgraph with {len(subgraph_nodes)} nodes")
        return subgraph_triples
    
    def analyze_subgraph_quality(self, subgraph_nodes, seed_entities):
        """
        Subgraph 품질 분석
        """
        print(f"\n=== Subgraph Quality Analysis ===")
        print(f"Total nodes: {len(subgraph_nodes)}")
        print(f"Seed entities: {len(seed_entities)}")
        
        # Seed entities가 subgraph에 포함되어 있는지 확인
        seed_ids = set()
        for entities in seed_entities.values():
            for entity_id, _ in entities:
                seed_ids.add(entity_id)
        
        included_seeds = seed_ids.intersection(set(subgraph_nodes))
        print(f"Seed entities included: {len(included_seeds)}/{len(seed_ids)}")
        
        # Seed coverage가 100%가 아니면 경고
        if len(included_seeds) != len(seed_ids):
            print(f"⚠️  WARNING: Not all seed entities are included in subgraph!")
            print(f"Missing seeds: {seed_ids - set(subgraph_nodes)}")
        else:
            print(f"✅ All seed entities are included in subgraph")
        
        # 연결성 분석
        connectivity = self._analyze_connectivity(subgraph_nodes)
        print(f"Average connectivity: {connectivity:.3f}")
        
        return {
            'total_nodes': len(subgraph_nodes),
            'seed_coverage': len(included_seeds) / len(seed_ids) if seed_ids else 0,
            'connectivity': connectivity,
            'seed_count': len(seed_ids),
            'included_seed_count': len(included_seeds)
        }
    
    def _analyze_connectivity(self, nodes):
        """
        노드들의 평균 연결성 계산
        """
        if len(nodes) < 2:
            return 0.0
        
        total_connections = 0
        node_set = set(nodes)
        
        for node in nodes:
            connections = 0
            for (head, relation, tail) in self.triple.keys():
                if head == node and tail in node_set:
                    connections += 1
                elif tail == node and head in node_set:
                    connections += 1
            total_connections += connections
        
        return total_connections / len(nodes)
    
    def build_subgraph_from_text(self, text, enable_adaptive_pruning=True, pruning_ratio=0.3):
        """
        텍스트 기반으로 subgraph 구성하는 통합 메서드
        
        Args:
            text (str): 입력 텍스트
            enable_adaptive_pruning (bool): Adaptive pruning 활성화 여부
            pruning_ratio (float): Pruning 비율 (0.0 ~ 1.0)
        
        Returns:
            dict: subgraph 정보를 담은 딕셔너리
        """
        try:
            print(f"Building subgraph from text: {text[:100]}...")
            
            # 키워드 추출 (LLM 사용 - 더 안정적이고 일관성 있는 결과)
            try:
                keywords = self.extract_keywords_with_llm(text)
                print(f"✓ Keywords extracted successfully: {keywords}")
            except Exception as keyword_error:
                print(f"❌ Keyword extraction failed: {keyword_error}")
                print("Using fallback keyword extraction...")
                # 최후의 fallback: 간단한 단어 분리
                words = text.lower().split()
                stop_words = set(stopwords.words("english"))
                keywords = [w for w in words if w not in stop_words and len(w) > 2][:self.topk]
                if not keywords:
                    keywords = ["text", "content", "information"][:self.topk]
                print(f"Fallback keywords: {keywords}")
            
            # 키워드 검증
            if not keywords or len(keywords) == 0:
                print("Warning: No keywords extracted, using default keywords")
                keywords = ["text", "content", "information"][:self.topk]
            
            # Entity 매칭 및 embedding 가져오기
            try:
                seed_entities = self.get_matching_entities(keywords)
                print(f"✓ Entity matching completed: {len(seed_entities)} entities found")
            except Exception as entity_error:
                print(f"❌ Entity matching failed: {entity_error}")
                seed_entities = {}
            
            try:
                entity_embeddings = self.get_kepler_embeddings_for_matched_entities(seed_entities)
                print(f"✓ Entity embeddings loaded: {len(entity_embeddings)} embeddings")
            except Exception as embedding_error:
                print(f"❌ Entity embedding loading failed: {embedding_error}")
                entity_embeddings = {}
            
            # Subgraph 구성 (Semantic Bridge 전략 사용)
            try:
                subgraph_nodes = self.construct_subgraph_semantic_bridge(
                    seed_entities, entity_embeddings, 
                    top_k=50, similarity_threshold=0.7, virtual_edge_ratio=0.1,
                    enable_adaptive_pruning=enable_adaptive_pruning, 
                    pruning_ratio=pruning_ratio
                )
                print(f"✓ Subgraph construction completed: {len(subgraph_nodes)} nodes")
            except Exception as subgraph_error:
                print(f"❌ Subgraph construction failed: {subgraph_error}")
                # 최소한의 subgraph 생성
                subgraph_nodes = list(seed_entities.keys()) if seed_entities else []
                print(f"Created minimal subgraph with {len(subgraph_nodes)} nodes")
            
            # Subgraph triples 추출
            try:
                subgraph_triples = self.get_subgraph_triples(subgraph_nodes)
                print(f"✓ Subgraph triples extracted: {len(subgraph_triples)} triples")
            except Exception as triple_error:
                print(f"❌ Triple extraction failed: {triple_error}")
                subgraph_triples = []
            
            # Subgraph 품질 분석
            try:
                quality_info = self.analyze_subgraph_quality(subgraph_nodes, seed_entities)
                print(f"✓ Quality analysis completed")
            except Exception as quality_error:
                print(f"❌ Quality analysis failed: {quality_error}")
                quality_info = {
                    'total_nodes': len(subgraph_nodes),
                    'seed_coverage': 0.0,
                    'connectivity': 0.0,
                    'seed_count': len(seed_entities),
                    'included_seed_count': 0
                }
            
            return {
                'keywords': keywords,
                'seed_entities': seed_entities,
                'subgraph_nodes': subgraph_nodes,
                'subgraph_triples': subgraph_triples,
                'entity_embeddings': entity_embeddings,
                'adaptive_pruning_enabled': enable_adaptive_pruning,
                'pruning_ratio': pruning_ratio,
                'quality_info': quality_info
            }
            
        except Exception as e:
            print(f"❌ Critical error in build_subgraph_from_text: {e}")
            print("Returning minimal subgraph...")
            
            # 최소한의 결과 반환
            return {
                'keywords': ["text", "content", "information"][:self.topk],
                'seed_entities': {},
                'subgraph_nodes': [],
                'subgraph_triples': [],
                'entity_embeddings': {},
                'adaptive_pruning_enabled': enable_adaptive_pruning,
                'pruning_ratio': pruning_ratio,
                'quality_info': {
                    'total_nodes': 0,
                    'seed_coverage': 0.0,
                    'connectivity': 0.0,
                    'seed_count': 0,
                    'included_seed_count': 0
                }
            }

# Example usage
if __name__ == "__main__":
    import time
    import json
    
    print("Testing Semantic Bridge subgraph construction...")
    
    llm_model = "llama-3-8b-chat"
    kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
    kg_entity_path = f"{kg_root_path}/entities.txt"
    kg_relation_path = f"{kg_root_path}/relations.txt"
    kg_triple_path = f"{kg_root_path}/triplets.txt"
    
    print(f"Loading KG from: {kg_root_path}")
    
    sg = subgraph_construction(llm_model, kg_entity_path=kg_entity_path, 
                               kg_relation_path=kg_relation_path, 
                               kg_triple_path=kg_triple_path)
    
    print(f"Loaded {len(sg.entity)} entities and {len(sg.relation)} relations")
    
    # Load test data from opengen_500.jsonl
    opengen_file = "data/opengen_500.jsonl"
    print(f"\nLoading test data from: {opengen_file}")
    
    # Read first 3 examples for testing
    test_examples = []
    with open(opengen_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only test first 3 examples
                break
            data = json.loads(line)
            # Combine prefix and targets
            combined_text = data["prefix"] + " " + " ".join(data["targets"])
            test_examples.append({
                'id': i,
                'text': combined_text,
                'prefix': data["prefix"],
                'targets': data["targets"]
            })
    
    print(f"Loaded {len(test_examples)} test examples")
    
    # Test each example
    for example in test_examples:
        print(f"\n{'='*80}")
        print(f"Testing Example {example['id']}")
        print(f"Text length: {len(example['text'])} characters")
        print(f"Prefix: {example['prefix'][:100]}...")
        print(f"Targets: {len(example['targets'])} targets")
        print(f"{'='*80}")
        
        # Extract keywords using NER only
        print("\nExtracting keywords...")
        keywords = sg.extract_keywords_with_ner(example['text'])
        
        print(f"NER keywords: {keywords}")
        
        # Get matched entities and embeddings
        print("\nMatching entities...")
        seed_entities = sg.get_matching_entities(keywords)
        
        print(f"Seed entities: {len(seed_entities)}")
        
        # Get embeddings
        print("\nGetting embeddings...")
        entity_embeddings = sg.get_kepler_embeddings_for_matched_entities(seed_entities)
        
        print(f"Embeddings: {len(entity_embeddings)}")
        
        # Test Semantic Bridge strategy
        print(f"\n{'='*60}")
        print(f"Testing Semantic Bridge Strategy for Example {example['id']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Test with NER keywords
        subgraph_nodes = sg.construct_subgraph_semantic_bridge(
            seed_entities, entity_embeddings, 
            top_k=50, similarity_threshold=0.7, virtual_edge_ratio=0.1
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Analyze quality
        quality = sg.analyze_subgraph_quality(subgraph_nodes, seed_entities)
        
        # Get triples
        triples = sg.get_subgraph_triples(subgraph_nodes)
        
        print(f"\n✅ Example {example['id']} completed in {execution_time:.2f} seconds")
        print(f"   Nodes: {len(subgraph_nodes)}, triples: {len(triples)}")
        print(f"   Seed coverage: {quality['seed_coverage']:.3f}")
        print(f"   Connectivity: {quality['connectivity']:.3f}")
        
        # Store results for this example
        example_results = {
            "example_id": example['id'],
            "execution_time": execution_time,
            "nodes": len(subgraph_nodes),
            "triples": len(triples),
            "quality": quality
        }
        
        # Save individual example results
        with open(f"semantic_bridge_example_{example['id']}_results.json", "w") as f:
            json.dump(example_results, f, indent=2)
        
        print(f"✅ Example {example['id']} results saved to semantic_bridge_example_{example['id']}_results.json")
    
    print(f"\n{'='*80}")
    print("All examples completed!")
    print(f"{'='*80}")
    
    # Summary of all results
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL EXAMPLES")
    print(f"{'='*80}")
    
    total_execution_time = 0
    total_nodes = 0
    total_triples = 0
    
    for example in test_examples:
        # Load results for this example
        try:
            with open(f"semantic_bridge_example_{example['id']}_results.json", "r") as f:
                example_results = json.load(f)
            
            total_execution_time += example_results['execution_time']
            total_nodes += example_results['nodes']
            total_triples += example_results['triples']
            
            print(f"Example {example['id']}: {example_results['execution_time']:.2f}s, "
                  f"Nodes: {example_results['nodes']}")
        except FileNotFoundError:
            print(f"Example {example['id']}: Results file not found")
    
    print(f"\nAVERAGE RESULTS:")
    print(f"  Average execution time: {total_execution_time/len(test_examples):.2f}s")
    print(f"  Average nodes: {total_nodes/len(test_examples):.1f}")
    print(f"  Average triples: {total_triples/len(test_examples):.1f}")
    
    print(f"\n{'='*80}")
    print("Semantic Bridge subgraph construction test completed!")
    print(f"{'='*80}")
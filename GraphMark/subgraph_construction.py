from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import torch
import spacy
import numpy as np
import random

class subgraph_construction():
    def __init__(self, llm=None, ratio=0.2, topk=5, kg_entity_path="entities.txt", kg_relation_path="relations.txt", kg_triple_path="triples.txt", device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LLM 설정 (테마 분석용)
        self.llm = llm
        self.ratio = ratio
        self.topk = topk
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load stopwords once
        self.stop_words = set(stopwords.words("english"))
        import os
        if os.path.isabs(kg_entity_path):
            kg_root_path = os.path.dirname(kg_entity_path)
        else:
            kg_root_path = os.path.abspath(os.path.dirname(kg_entity_path))
        
        print(f"Calculated kg_root_path: {kg_root_path}")
        self.load_pretrained_embeddings(kg_root_path)
        print("Pre-trained embeddings enabled (KEPLER always ON)")
        self.entity, self.relation, self.triple = self.load_kg(kg_entity_path, kg_relation_path, kg_triple_path)
    
    def load_pretrained_embeddings(self, kg_root_path):
        self.entity_embeddings = np.load(f"{kg_root_path}/entity_embeddings_full.npy")
        self.relation_embeddings = np.load(f"{kg_root_path}/relation_embeddings_full.npy")
        
        self.entity_id_to_name = {}
        self.entity_name_to_id = {}
        self.entity_id_to_idx = {}
        
        with open(f"{kg_root_path}/entities.txt", 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                entity_name = parts[1] if len(parts) > 1 else entity_id
                
                self.entity_id_to_name[entity_id] = entity_name
                self.entity_name_to_id[entity_name] = entity_id
                self.entity_id_to_idx[entity_id] = idx
        
        print(f"Loaded {len(self.entity_id_to_name)} entities")
        print(f"New entity embeddings size: {self.entity_embeddings.shape[0]} entities")
        print(f"Entity mapping size: {len(self.entity_id_to_idx)} entities")
        
        if len(self.entity_id_to_idx) != self.entity_embeddings.shape[0]:
            print(f"Warning: Entity mapping count ({len(self.entity_id_to_idx)}) != embedding count ({self.entity_embeddings.shape[0]})")
        else:
            print("✓ Entity mapping and embeddings are aligned!")
    
    def get_entity_embedding(self, entity_name):
        if self.entity_embeddings is None:
            return None
        
        try:
            if entity_name in self.entity_name_to_id:
                entity_id = self.entity_name_to_id[entity_name]
                if entity_id in self.entity_id_to_idx:
                    entity_idx = self.entity_id_to_idx[entity_id]
                    if entity_idx < len(self.entity_embeddings):
                        return self.entity_embeddings[entity_idx]
                    else:
                        print(f"Entity index {entity_idx} out of range for embeddings (max: {len(self.entity_embeddings)})")
                else:
                    print(f"Entity ID {entity_id} not found in index mapping")
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
        """Extract keywords using spaCy NER - preserve original case"""
        if not text or len(text.strip()) < 10:
            return ["text", "content", "information"][:self.topk]
        
        try:
            doc = self.nlp(text)
            keywords = []
            keywords_lower = []  # For duplicate checking
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC"]:
                    keyword = ent.text.strip()
                    keyword_lower = keyword.lower()
                    if keyword and len(keyword) > 1 and keyword_lower not in self.stop_words and keyword_lower not in keywords_lower:
                        keywords.append(keyword)
                        keywords_lower.append(keyword_lower)
            
            # Add nouns if not enough
            if len(keywords) < self.topk:
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                        keyword = token.text.strip()
                        keyword_lower = keyword.lower()
                        if keyword_lower not in self.stop_words and keyword_lower not in keywords_lower:
                            keywords.append(keyword)
                            keywords_lower.append(keyword_lower)
            
            return keywords[:self.topk] if keywords else ["text", "content", "information"][:self.topk]
        except Exception as e:
            print(f"Error in NER extraction: {e}")
            return ["text", "content", "information"][:self.topk]
    
    def extract_keywords_simple(self, text):
        """Extract keywords using NER, with semantic relevance filtering"""
        # Try NER first
        ner_keywords = self.extract_keywords_with_ner(text)
        
        # If LLM available and we need more keywords, try simple LLM extraction
        if self.llm and hasattr(self.llm, 'generate') and len(ner_keywords) < self.topk:
            try:
                llm_keywords = self._extract_keywords_with_llm(text)
                # Combine and deduplicate (case-insensitive)
                all_keywords = ner_keywords.copy()
                existing_lower = {kw.lower() for kw in all_keywords}
                
                for kw in llm_keywords:
                    if kw.lower() not in existing_lower:
                        all_keywords.append(kw)
                        existing_lower.add(kw.lower())
                
                candidate_keywords = all_keywords[:self.topk * 2]  # Get more candidates for filtering
            except Exception as e:
                print(f"LLM extraction failed: {e}")
                candidate_keywords = ner_keywords
        else:
            candidate_keywords = ner_keywords
        
        # Filter keywords by semantic relevance to text
        filtered_keywords = self._filter_keywords_by_relevance(candidate_keywords, text)
        
        # 최종 결과가 비어있으면 fallback 키워드 사용
        if not filtered_keywords:
            print(f"   ⚠️  No keywords after filtering, using fallback keywords")
            filtered_keywords = candidate_keywords[:self.topk] if candidate_keywords else ["text", "content", "information"][:self.topk]
        
        result = filtered_keywords[:self.topk]
        print(f"   ✅ Final keywords ({len(result)}): {result}")
        return result
    
    def extract_keywords_with_thematic_analysis(self, text):
        """Legacy function - use extract_keywords_simple instead"""
        return self.extract_keywords_simple(text)
    
    def _extract_keywords_with_llm(self, text):
        """Simple LLM-based keyword extraction"""
        try:
            prompt = f"Extract key words from: {text[:200]}\nKeywords (comma-separated):"
            response = self.llm.generate(prompt, max_tokens=50)
            keywords = self._parse_llm_keywords(response)
            keywords = self._filter_numeric_keywords(keywords)
            return keywords
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []
    
    
    def _parse_llm_keywords(self, response):
        """Parse keywords from LLM response - preserve original case"""
        try:
            response = response.strip()
            if not response:
                return []
            
            import re
            keywords = []
            
            # Try separators first
            for sep in [',', ';', '\n', '|']:
                if sep in response:
                    parts = [part.strip() for part in response.split(sep)]
                    for part in parts:
                        if part and len(part) > 1 and len(part) < 50:
                            clean_keyword = re.sub(r'[^\w\s-]', '', part).strip()
                            if clean_keyword and len(clean_keyword.split()) <= 3:
                                # Preserve original case
                                clean_keyword_lower = clean_keyword.lower()
                                if clean_keyword_lower not in self.stop_words:
                                    keywords.append(clean_keyword)
                    break
            
            # Fallback to word splitting
            if not keywords:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', response)
                for w in words:
                    if 3 <= len(w) <= 20:
                        w_lower = w.lower()
                        if w_lower not in self.stop_words:
                            keywords.append(w)  # Preserve case
            
            return keywords
            
        except Exception as e:
            print(f"Error parsing LLM keywords: {e}")
            return []
    
    def _filter_numeric_keywords(self, keywords):
        """Filter out keywords that are mostly numeric"""
        import re
        filtered = []
        for kw in keywords:
            # Remove keywords that are mostly numbers
            if not re.match(r'^\d+$', kw):
                filtered.append(kw)
        return filtered
    
    def _filter_keywords_by_relevance(self, keywords, text):
        """Filter keywords based on semantic relevance to text topic"""
        if not keywords:
            print(f"   ⚠️  No keywords to filter")
            return keywords
        
        # 키워드가 3개 이하이면 필터링하지 않고 그대로 반환 (너무 많이 걸러지는 것 방지)
        if len(keywords) <= 3:
            print(f"   📊 Skipping relevance filtering (too few keywords: {len(keywords)})")
            return keywords
        
        try:
            # Compute text embedding
            text_embedding = self.sbert_model.encode(text, convert_to_numpy=True)
            
            # Compute relevance scores for each keyword
            keyword_scores = []
            for keyword in keywords:
                # Compute keyword embedding
                keyword_embedding = self.sbert_model.encode(keyword, convert_to_numpy=True)
                
                # Compute cosine similarity
                similarity = np.dot(text_embedding, keyword_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(keyword_embedding)
                )
                
                keyword_scores.append((keyword, similarity))
            
            # Sort by relevance (higher = more relevant)
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by relevance threshold (keep top 70% most relevant, but at least top 3)
            threshold = np.median([score for _, score in keyword_scores])
            filtered_keywords = [kw for kw, score in keyword_scores if score >= threshold]
            
            # 최소한 top 3은 보장 (필터링 결과가 너무 적으면)
            min_keywords = min(3, len(keywords))
            if len(filtered_keywords) < min_keywords:
                filtered_keywords = [kw for kw, _ in keyword_scores[:min_keywords]]
                print(f"   ⚠️  Relevance filtering too aggressive, keeping top {min_keywords} keywords")
            
            print(f"   📊 Relevance filtering: {len(keywords)} -> {len(filtered_keywords)} keywords")
            print(f"   📊 Relevance threshold: {threshold:.3f}")
            
            return filtered_keywords
            
        except Exception as e:
            print(f"   ⚠️  Relevance filtering failed: {e}, using all keywords")
            return keywords
    
    def get_matching_entities(self, keywords):
        matched_entities = {}
        if not hasattr(self, '_entity_name_index'):
            print("Building entity name index for fast matching...")
            self._entity_name_index = self._build_entity_name_index()
            print(f"Index built with {len(self._entity_name_index)} entries")
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            best_match = self._fallback_entity_matching(keyword)
            if best_match:
                print(f"✅ Match: {best_match[0]} ('{best_match[1]}')")
                matched_entities[keyword] = [best_match]
            else:
                print(f"❌ No match found for '{keyword}'")
                
        print(f"\n📊 Final matching results: {len(matched_entities)}/{len(keywords)} keywords matched")
        for keyword, entities in matched_entities.items():
            entity_id, entity_name = entities[0]
            print(f"   '{keyword}' -> {entity_id} ('{entity_name}')")
                
        return matched_entities
    
    def _fallback_entity_matching(self, keyword):
        """Entity 매칭 (정확한 매칭 + 부분 매칭)"""
        keyword_lower = keyword.lower()
        
        # 정확한 매칭
        if keyword_lower in self._entity_name_index:
            entity_id, entity_name = self._entity_name_index[keyword_lower]
            return (entity_id, entity_name)
        
        # 부분 매칭
        candidates = []
        for indexed_name, (entity_id, entity_name) in self._entity_name_index.items():
            if (keyword_lower in indexed_name or 
                any(word in indexed_name for word in keyword_lower.split())):
                score = self._fast_similarity(keyword_lower, indexed_name)
                if score > 0.6:
                    candidates.append((score, entity_id, entity_name))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return (candidates[0][1], candidates[0][2])
        
        return None
    
    def _build_entity_name_index(self):
        """엔티티 이름 인덱스 구축 (최초 1회만 실행)"""
        index = {}
        
        for entity_id, entity_data in self.entity.items():
            entity_name_list = entity_data["entity"]
            
            # entity_name이 리스트인 경우 처리
            if isinstance(entity_name_list, list):
                entity_names = entity_name_list
            else:
                entity_names = [entity_name_list]
            
            # 각 entity 이름을 인덱스에 추가
            for entity_name in entity_names:
                if isinstance(entity_name, str) and len(entity_name.strip()) > 0:
                    name_lower = entity_name.lower().strip()
                    if name_lower not in index:  # 중복 방지
                        index[name_lower] = (entity_id, entity_name)
        
        return index
    
    def _fast_similarity(self, str1, str2):
        """빠른 유사도 계산 (SequenceMatcher 대신)"""
        if str1 == str2:
            return 1.0
        
        # Jaccard 유사도 (빠름)
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # 부분 일치 보너스
        if str1 in str2 or str2 in str1:
            jaccard += 0.3
        
        return min(1.0, jaccard)
    
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
                
                # 디버깅: entity_id가 매핑에 있는지 확인
                if entity_id in self.entity_id_to_name:
                    print(f"    Found in entity mapping")
                else:
                    print(f"    NOT found in entity mapping")
                
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
    
    def find_paths_between_entities(self, seed_entities, entity_embeddings, max_hops=1, max_neighbors_per_hop=5):
        """
        Step 3: Path Searching - Find paths between seed entities in the KG
        
        Args:
            seed_entities: Matched entities dictionary
            entity_embeddings: Entity embeddings
            max_hops: Maximum number of hops to explore (reduced for speed)
            max_neighbors_per_hop: Maximum neighbors to explore per hop (reduced for speed)
        
        Returns:
            set: Nodes found via path search
        """
        print("Searching for paths between seed entities...")
        
        # Get all seed entity IDs
        seed_ids = set()
        for keyword, entities in seed_entities.items():
            for entity_id, entity_name in entities:
                seed_ids.add(entity_id)
        
        print(f"Searching paths for {len(seed_ids)} seed entities...")
        
        # Start with seed nodes
        subgraph_nodes = set(seed_ids)
        seed_nodes_frozen = frozenset(seed_ids)
        
        # BFS: explore neighbors of seed nodes (limited to 1 hop)
        for hop in range(1, max_hops + 1):
            current_nodes = list(subgraph_nodes)
            new_nodes = set()
            
            for node_id in current_nodes:
                # Find directly connected entities (limited neighbors)
                neighbors = self._find_directly_connected_entities(node_id, max_neighbors_per_hop)
                
                # Add neighbors that are not seed nodes
                for neighbor in neighbors:
                    if neighbor not in seed_nodes_frozen:
                        new_nodes.add(neighbor)
            
            subgraph_nodes.update(new_nodes)
            print(f"  Hop {hop}: Found {len(new_nodes)} new nodes")
            
            # Early stop if too many nodes
            if len(subgraph_nodes) > 50:
                print(f"  Stopping early (reached {len(subgraph_nodes)} nodes)")
                break
        
        print(f"Total nodes found via path search: {len(subgraph_nodes)}")
        return subgraph_nodes
    
    def add_virtual_edges(self, seed_entities, entity_embeddings, existing_nodes, 
                         similarity_threshold=0.7, max_virtual_nodes=5):
        """
        Step 4: Virtual Edge Connection - Add bridge entities between disconnected seed entities
        
        Args:
            seed_entities: Matched entities
            entity_embeddings: Entity embeddings
            existing_nodes: Current subgraph nodes (set)
            similarity_threshold: Minimum similarity for virtual edges
            max_virtual_nodes: Maximum number of bridge nodes to add
        
        Returns:
            list: New nodes added via virtual edges (that are actually connected in KG)
        """
        print("Checking seed entity connectivity...")
        
        if not entity_embeddings or len(entity_embeddings) < 2:
            print("Not enough seed entities for virtual edges")
            return []
        
        # Get all seed entity IDs
        seed_ids = []
        for keyword, entities in seed_entities.items():
            for entity_id, entity_name in entities:
                seed_ids.append(entity_id)
        
        # Check if seed entities are connected in the current graph
        existing_nodes_set = set(existing_nodes)
        unconnected_pairs = []
        
        for i, seed1 in enumerate(seed_ids):
            for seed2 in seed_ids[i+1:]:
                has_path = self._check_simple_path(seed1, seed2, existing_nodes_set)
                if not has_path:
                    unconnected_pairs.append((seed1, seed2))
        
        print(f"Found {len(unconnected_pairs)} unconnected seed entity pairs")
        
        if not unconnected_pairs:
            print("All seed entities are already connected")
            return []
        
        # For each unconnected pair, find a bridge entity (minimal for speed)
        virtual_nodes = []
        sampled_entities = random.sample(list(self.entity.keys()), min(100, len(self.entity)))
        
        # Only process first 1 pair for speed
        if unconnected_pairs:
            seed1, seed2 = unconnected_pairs[0]
            bridge = self._find_bridge_between_two_seeds(seed1, seed2, sampled_entities, 
                                                         existing_nodes_set, similarity_threshold)
            if bridge:
                virtual_nodes.append(bridge)
                print(f"   ✓ Added bridge {bridge}")
        
        print(f"Added {len(virtual_nodes)} virtual bridge nodes")
        return virtual_nodes
    
    def _verify_bridge_connection(self, bridge_id, existing_nodes):
        """
        Check if bridge entity has connections to existing subgraph nodes
        (Currently disabled for speed - all triples are validated in get_subgraph_triples)
        """
        # Skip verification for speed - triples are validated later in get_subgraph_triples
        return True
    
    def _check_simple_path(self, entity1, entity2, nodes_in_graph):
        """Check if there's a path between two entities in current graph (simple check)"""
        if entity1 not in nodes_in_graph or entity2 not in nodes_in_graph:
            return False
        
        # Quick check: do they share any common neighbors?
        neighbors1 = self._get_one_hop_neighbors(entity1, nodes_in_graph)
        neighbors2 = self._get_one_hop_neighbors(entity2, nodes_in_graph)
        
        # If they share neighbors, they're connected
        if neighbors1.intersection(neighbors2):
            return True
        
        # If entity2 is in neighbors of entity1 or vice versa
        if entity2 in neighbors1 or entity1 in neighbors2:
            return True
        
        return False
    
    def _get_one_hop_neighbors(self, entity_id, nodes_in_graph):
        """Get one-hop neighbors of an entity"""
        neighbors = set()
        for (head, relation, tail) in self.triple.keys():
            if head == entity_id and tail in nodes_in_graph:
                neighbors.add(tail)
            elif tail == entity_id and head in nodes_in_graph:
                neighbors.add(head)
        return neighbors
    
    def _find_bridge_between_two_seeds(self, seed1, seed2, candidate_entities, existing_nodes, threshold=0.7):
        """
        Find one bridge entity between two seed entities
        Fast version: no connection verification, just similarity check
        """
        emb1 = self.get_entity_embedding_by_id(seed1)
        emb2 = self.get_entity_embedding_by_id(seed2)
        
        if emb1 is None or emb2 is None:
            return None
        
        best_candidate = None
        best_score = 0
        
        # Check up to 50 candidate entities (minimal for speed)
        for candidate_id in candidate_entities[:50]:
            if candidate_id in existing_nodes:
                continue
            
            candidate_emb = self.get_entity_embedding_by_id(candidate_id)
            if candidate_emb is not None:
                sim1 = self._compute_cosine_similarity(emb1, candidate_emb)
                sim2 = self._compute_cosine_similarity(emb2, candidate_emb)
                
                # Average similarity
                avg_sim = (sim1 + sim2) / 2
                
                if sim1 >= threshold and sim2 >= threshold and avg_sim > best_score:
                    best_score = avg_sim
                    best_candidate = candidate_id
        
        return best_candidate
    
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
        Optimized: skips connection check for speed
        """
        similarities = []
        
        # Only check first 300 candidates for speed
        for entity_id in candidate_entities[:300]:
            entity_embedding = self.get_entity_embedding_by_id(entity_id)
            if entity_embedding is not None and not np.allclose(entity_embedding, 0):
                similarity = self._compute_cosine_similarity(query_embedding, entity_embedding)
                if similarity >= threshold:
                    similarities.append((entity_id, similarity))
        
        # 유사도 순으로 정렬하고 상위 k개 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in similarities[:top_k]]
    
    def _entity_has_connections(self, entity_id):
        """
        Check if entity has connections in the KG
        (Currently disabled for speed - triples are validated in get_subgraph_triples)
        """
        # Skip verification for speed - triples are validated later in get_subgraph_triples
        return True
    
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
        Optimized: skips connection check for speed
        """
        bridge_nodes = set()
        
        # 각 seed의 embedding
        seed1_embedding = self.get_entity_embedding_by_id(seed1_id)
        seed2_embedding = self.get_entity_embedding_by_id(seed2_id)
        
        if seed1_embedding is None or seed2_embedding is None:
            return bridge_nodes
        
        # 두 seed와 모두 유사한 중간 노드들 찾기 (only check first 200)
        for entity_id in candidate_entities[:200]:
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
    
    def get_subgraph_triples(self, subgraph_nodes, seed_entities=None):
        """
        Subgraph 노드들에 해당하는 triples 추출 + 키워드 매칭된 entity 관련 triplet 보장
        """
        subgraph_triples = []
        
        # 1. 일반적인 subgraph triplets 추출
        for (head, relation, tail) in self.triple.keys():
            if head in subgraph_nodes and tail in subgraph_nodes:
                subgraph_triples.append([head, relation, tail])
        
        # 2. 키워드 entity와 관련된 triplet 추가 (각 Entity마다 최소 5개)
        if seed_entities:
            keyword_entity_ids = set()
            for keyword, entities in seed_entities.items():
                for entity_id, entity_name in entities:
                    keyword_entity_ids.add(entity_id)
            
            # 각 Entity별로 최소 5개씩 triplet 보장
            entity_triplet_counts = {}
            for entity_id in keyword_entity_ids:
                entity_triplet_counts[entity_id] = 0
            
            # 먼저 기존 subgraph_triples에서 키워드 entity들의 triplet 개수 계산
            for triple in subgraph_triples:
                if len(triple) >= 3:
                    head, relation, tail = triple
                    if head in keyword_entity_ids:
                        entity_triplet_counts[head] += 1
                    if tail in keyword_entity_ids:
                        entity_triplet_counts[tail] += 1
            
            # 키워드 entity와 관련된 triplet들 추가 (최소 5개씩 보장)
            added_count = 0
            for (head, relation, tail) in self.triple.keys():
                triple = [head, relation, tail]
                if triple not in subgraph_triples:
                    # head가 키워드 entity인 경우
                    if head in keyword_entity_ids and entity_triplet_counts[head] < 5:
                        subgraph_triples.append(triple)
                        entity_triplet_counts[head] += 1
                        added_count += 1
                    # tail이 키워드 entity인 경우
                    elif tail in keyword_entity_ids and entity_triplet_counts[tail] < 5:
                        subgraph_triples.append(triple)
                        entity_triplet_counts[tail] += 1
                        added_count += 1
            
            # 각 Entity별 triplet 개수 출력
            print(f"Entity triplet counts:")
            entities_with_insufficient_triplets = []
            for entity_id in keyword_entity_ids:
                count = entity_triplet_counts[entity_id]
                print(f"  {entity_id}: {count} triplets")
                if count < 5:
                    entities_with_insufficient_triplets.append((entity_id, count))
            
            if entities_with_insufficient_triplets:
                print(f"⚠️  Warning: {len(entities_with_insufficient_triplets)} entities have less than 5 triplets:")
                for entity_id, count in entities_with_insufficient_triplets:
                    print(f"    {entity_id}: {count} triplets")
            else:
                print(f"✅ All keyword entities have at least 5 triplets")
            
            print(f"Added {added_count} keyword-related triplets")
        
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
        print(f"\n{'='*80}")
        print(f"Building subgraph from text: {text[:100]}...")
        print(f"{'='*80}\n")
        
        # Step 1: Keyword Extraction
        print("Step 1: Keyword Extraction")
        print("-" * 80)
        keywords = self.extract_keywords_simple(text)
        print(f"Keywords: {keywords}")
        
        # 키워드가 비어있으면 경고하고 빈 리스트 반환
        if not keywords:
            print(f"⚠️  WARNING: No keywords extracted from text!")
            print(f"   This may cause subgraph construction to fail.")
            print(f"   Text length: {len(text)} characters")
            print(f"   Text preview: {text[:200]}...\n")
        else:
            print(f"✅ Extracted {len(keywords)} keywords\n")
        
        # Step 2: Entity Matching
        print("Step 2: Entity Matching")
        print("-" * 80)
        seed_entities, entity_embeddings = self.get_entity_matching_with_kepler_embeddings(keywords)
        print(f"Matched {len(seed_entities)} entities\n")
        
        # Step 3: Path Searching
        print("Step 3: Path Searching")
        print("-" * 80)
        subgraph_nodes_set = self.find_paths_between_entities(seed_entities, entity_embeddings)
        print(f"Found {len(subgraph_nodes_set)} nodes via path search\n")
        
        # Step 4: Virtual Edge Connection
        print("Step 4: Virtual Edge Connection")
        print("-" * 80)
        virtual_nodes = self.add_virtual_edges(seed_entities, entity_embeddings, subgraph_nodes_set)
        subgraph_nodes = list(subgraph_nodes_set) + virtual_nodes
        print(f"Added {len(virtual_nodes)} virtual nodes\n")
        
        # Step 5: Adaptive Pruning
        print("Step 5: Adaptive Pruning")
        print("-" * 80)
        if enable_adaptive_pruning:
            subgraph_nodes = self.adaptive_pruning(
                subgraph_nodes, seed_entities, entity_embeddings,
                base_threshold=0.7, pruning_ratio=pruning_ratio, min_nodes=10
            )
        print(f"Final subgraph: {len(subgraph_nodes)} nodes\n")
            
        # Get triples
        print("Extracting triples...")
        subgraph_triples = self.get_subgraph_triples(subgraph_nodes, seed_entities)
        
        # Quality analysis
        quality_info = self.analyze_subgraph_quality(subgraph_nodes, seed_entities)
        
        print(f"\n{'='*80}")
        print("Subgraph construction completed!")
        print(f"{'='*80}\n")
        
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
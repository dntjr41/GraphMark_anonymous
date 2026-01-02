import os
import torch
import re
import spacy
import time
import random
from typing import List, Dict, Optional
from transformers import BertTokenizer, BertModel
import numpy as np
from subgraph_construction import subgraph_construction
from transformers import RobertaModel, RobertaTokenizer
from llm_models import LLM, ChatGPT, Llama3_8B, Mistral7B, Qwen2_5_7B, KEPLEREmbedding

torch.manual_seed(42)
random.seed(42)

class KGWatermarker():
    def __init__(self, llm, ratio, topk=5, device_id=None, rarity_similarity_threshold: float = 0.6):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(llm, str):
            self.llm = LLM(llm, device_id=device_id)
        else:
            self.llm = llm
        self.ratio = ratio
        self.rarity_similarity_threshold = rarity_similarity_threshold
        self.nlp = spacy.load("en_core_web_sm")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.roberta_model.eval()
        
        kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
        self.constructor = subgraph_construction(
            llm=self.llm, ratio=ratio, topk=topk,
            kg_entity_path=f"{kg_root_path}/entities.txt", 
            kg_relation_path=f"{kg_root_path}/relations.txt", 
            kg_triple_path=f"{kg_root_path}/triplets.txt", 
            device_id=device_id)
        
        self.entity, self.relation, self.triple = self.constructor.load_kg(
            f"{kg_root_path}/entities.txt", 
            f"{kg_root_path}/relations.txt", 
            f"{kg_root_path}/triplets.txt")

    def _get_sentence_embedding(self, text):
        """RoBERTa sentence embedding via [CLS] (start token) representation."""        
        encoded = self.roberta_tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.roberta_model(**encoded)
        # CLS-equivalent: first token hidden state
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
        return cls_vec
    
    def _triplet_to_plain_sentence(self, triplet):
        """Create a simple textual representation of a triplet using KG labels."""
        if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
            return str(triplet)

        head, relation, tail = triplet[:3]
        head_name = self._select_best_name(head, self.entity)
        relation_name = self._select_best_name(relation, self.relation)
        tail_name = self._select_best_name(tail, self.entity)
        return f"{head_name} {relation_name} {tail_name}"

    def _filter_triplets_by_rarity(self, triplets, original_sentences, threshold=None, fallback_keep: int = None):
        """Select only triplets that are semantically distant from original sentences."""
        if not triplets:
            return triplets, []
        
        if threshold is None:
            threshold = getattr(self, 'rarity_similarity_threshold', 0.6)
        
        # Fallback keep 개수를 ratio 기반으로 조정 (최소 5개)
        if fallback_keep is None:
            fallback_keep = max(5, int(len(triplets) * 0.3))  # 최소 30% 또는 5개

        # Prepare sentence embeddings once
        sentence_embeddings = []
        if original_sentences:
            for sent in original_sentences:
                try:
                    emb = self._get_sentence_embedding(sent)
                    if emb is not None:
                        sentence_embeddings.append((sent, emb))
                except Exception:
                    continue

        if not sentence_embeddings:
            return triplets, []

        accepted = []
        scored_triplets = []

        for triplet in triplets:
            try:
                triplet_sentence = self._triplet_to_plain_sentence(triplet)
                triplet_emb = self._get_sentence_embedding(triplet_sentence)
                if triplet_emb is None:
                    continue

                similarities = [self._calculate_cosine_similarity(triplet_emb, sent_emb) for _, sent_emb in sentence_embeddings]
                max_sim = max(similarities) if similarities else 0.0

                scored_triplets.append((triplet, max_sim, triplet_sentence))

                if max_sim < threshold:
                    accepted.append((triplet, max_sim, triplet_sentence))
            except Exception:
                continue

        if accepted:
            return [triplet for triplet, _, _ in accepted], accepted

        # Fallback: keep the lowest similarity triplets to avoid empty selection
        # 더 많은 triplet을 보존하여 ratio 기반 선택이 가능하도록
        scored_triplets.sort(key=lambda item: item[1])
        fallback_count = min(fallback_keep, len(scored_triplets))
        fallback = scored_triplets[:fallback_count]
        print(f"   ⚠️ Rarity filtering: No triplets below threshold {threshold:.2f}, keeping {fallback_count} lowest similarity triplets")
        return [triplet for triplet, _, _ in fallback], fallback
    
    def build_subgraph_from_text(self, text, enable_adaptive_pruning=True, pruning_ratio=0.3):
        """Build subgraph from text using constructor"""
        return self.constructor.build_subgraph_from_text(text, enable_adaptive_pruning, pruning_ratio)
    
    def convert_triple_to_sentence(self, triple, keywords=None):
        head, relation, tail = triple
        head_name = self._select_best_name(head, self.entity, keywords)
        tail_name = self._select_best_name(tail, self.entity, keywords)
        relation_name = self._select_best_name(relation, self.relation)
        return self._create_fallback_sentence(head_name, relation_name, tail_name)
    
    def _select_best_name(self, item_id, data_dict, keywords=None):
        try:
            # 1. 데이터에서 이름 목록 추출
            if item_id not in data_dict:
                return str(item_id)
            
            entity_data = data_dict[item_id]
            if isinstance(entity_data, dict):
                names = entity_data.get("entity" if "entity" in entity_data else "name", [str(item_id)])
            else:
                names = entity_data if isinstance(entity_data, list) else [str(item_id)]
            
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
            print(f"Error in _select_best_name: {e}")
            return str(item_id)
    
    def _is_english_text(self, text):
        """텍스트가 영어인지 확인"""
        if not text or not isinstance(text, str):
            return False
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return False
        english_ratio = english_chars / total_chars
        return english_ratio >= 0.8
    
    def _is_better_name(self, name1, name2):
        """두 이름 중 더 나은 이름 선택 (영어 우선, 적절한 길이, 특수문자 최소화)"""
        # 1. 영어 우선
        is_english1 = self._is_english_text(name1)
        is_english2 = self._is_english_text(name2)
        if is_english1 and not is_english2:
            return True
        if is_english2 and not is_english1:
            return False
        
        # 2. 적절한 길이 (5-50자) 우선
        len1, len2 = len(name1), len(name2)
        if 5 <= len1 <= 50 and not (5 <= len2 <= 50):
            return True
        if 5 <= len2 <= 50 and not (5 <= len1 <= 50):
            return False
        
        # 3. 특수문자 최소화
        special1 = sum(1 for c in name1 if not c.isalnum() and c != ' ')
        special2 = sum(1 for c in name2 if not c.isalnum() and c != ' ')
        if special1 < special2:
            return True
        if special2 < special1:
            return False
        
        # 4. 적절한 단어 수 (1-5개) 우선
        words1, words2 = len(name1.split()), len(name2.split())
        if 1 <= words1 <= 5 and not (1 <= words2 <= 5):
            return True
        if 1 <= words2 <= 5 and not (1 <= words1 <= 5):
            return False
        
        # 5. 길이가 짧은 것 우선
        return len1 < len2
    
    def _is_entity_name_match(self, entity_id, keyword):
        """엔티티 ID의 이름이 키워드와 매칭되는지 확인"""
        try:
            if entity_id not in self.entity:
                return False
                
            entity_data = self.entity[entity_id]
            
            # 엔티티 데이터에서 이름 목록 추출
            if isinstance(entity_data, dict):
                entity_names = entity_data.get('name', list(entity_data.values()))
            else:
                entity_names = entity_data
            
            # 이름 목록이 리스트인 경우
            if isinstance(entity_names, list):
                for name in entity_names:
                    if isinstance(name, str):
                        if keyword.lower() == name.lower():
                            return True
                        if keyword.lower() in name.lower() or name.lower() in keyword.lower():
                            return True
                return False
            
            # 이름이 문자열인 경우
            elif isinstance(entity_names, str):
                if keyword.lower() == entity_names.lower():
                    return True
                if keyword.lower() in entity_names.lower() or entity_names.lower() in keyword.lower():
                    return True
                return False
            
            return False
            
        except Exception as e:
            print(f"Error in entity name matching: {e}")
            return False
    
    def insert_watermark(self, prefix, target, enable_adaptive_pruning=True, pruning_ratio=0.2):
        """Main watermarking function with improved LLM-based modification and insertion"""
        combined_text = f"{prefix} {target}"
        
        # Split text into sentences FIRST (needed for rarity filtering)
        doc = self.nlp(combined_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        original_sentences = sentences.copy()  # Reference to original
        
        # Build subgraph
        subgraph_info = self.build_subgraph_from_text(combined_text, enable_adaptive_pruning, pruning_ratio)
        
        # Select triplets (now sentences is defined)
        # 더 많은 triplet을 선택하기 위해 필터링 임계값을 완화
        selected_triplets, _, _ = self.select_triplets_for_watermarking(
            subgraph_info['subgraph_triples'], subgraph_info['keywords'], 
            main_topic=subgraph_info.get('main_topic', None), similarity_threshold=0.2,  # 0.3 -> 0.1로 완화
            original_sentences=sentences,
            rarity_threshold=0.75  # 0.6 -> 0.8로 완화 (높을수록 더 많은 triplet 통과)
        )
        
        # total_triplets는 subgraph_triples의 개수
        total_triplets = len(subgraph_info['subgraph_triples'])
        selected_triplets_count = len(selected_triplets)
        
        # Adaptive Ratio 적용: 문서 복잡도에 따른 Modify/Insert 비율 조정
        complexity_score = self._analyze_document_complexity(sentences)
        modify_ratio, insert_ratio = self._calculate_adaptive_ratio(complexity_score)
        
        # 수정 비율 제한 (ratio 기반, min: 3, max: 10 or 50% of sentences)
        # total_triplets의 ratio만큼 사용 (예: 25개 * 0.25 = 6.25 -> 6개, 최소 3개)
        ratio_based_count = int(total_triplets * self.ratio) if total_triplets > 0 else 0
        # 문장 수 기반 계산 (문장 수의 ratio만큼)
        sentence_based_count = int(len(sentences) * self.ratio)
        
        # 최소값: 3 (단, selected_triplets_count가 3 미만이면 selected_triplets_count)
        min_triplets = min(3, selected_triplets_count) if selected_triplets_count > 0 else 0
        # 최대값: 10 또는 문장 수의 50%
        max_triplets = min(10, int(len(sentences) * 0.5))
        
        # Ratio 기반 목표 개수 (total_triplets ratio와 sentence ratio 중 더 큰 값, 최소 3)
        target_by_ratio = max(ratio_based_count, sentence_based_count)
        target_by_ratio = max(min_triplets, target_by_ratio)  # 최소값 보장
        target_by_ratio = min(target_by_ratio, max_triplets, selected_triplets_count)  # 최대값 및 선택 가능 개수 제한
        
        max_allowed_triplets = target_by_ratio
        
        # 실제 사용할 triplet 수 제한
        actual_triplets = selected_triplets[:max_allowed_triplets]
        modify_count = max(1, int(len(actual_triplets) * modify_ratio))
        insert_count = len(actual_triplets) - modify_count
        
        print(f"📊 Document Complexity Analysis:")
        print(f"   Complexity Score: {complexity_score:.3f}")
        print(f"   Total Triplets (Subgraph): {total_triplets}")
        print(f"   Selected Triplets: {selected_triplets_count}")
        print(f"   Ratio-based count: {ratio_based_count} (from {total_triplets} * {self.ratio:.2f})")
        print(f"   Sentence-based count: {sentence_based_count} (from {len(sentences)} * {self.ratio:.2f})")
        print(f"   Allowed Triplets: {max_allowed_triplets} (min: {min_triplets}, max: {max_triplets} or 50% of sentences)")
        print(f"   Modify Ratio: {modify_ratio:.3f} ({modify_count} triplets)")
        print(f"   Insert Ratio: {insert_ratio:.3f} ({insert_count} triplets)")
        
        print(f"📊 Triplet distribution: {len(actual_triplets)} used -> {modify_count} modify, {insert_count} insert")
        
        # Triplet 선택 기준에 따라 Modify/Insert 분류
        modify_triplets, insert_triplets = self._select_triplets_for_modify_and_insert(
            actual_triplets, sentences, subgraph_info['keywords'], modify_count, insert_count
        )
        
        print(f"   Modify triplets: {len(modify_triplets)}")
        print(f"   Insert triplets: {len(insert_triplets)}")
        
        # Step 1: Modify sentences with selected triplets
        print(f"\n{'='*80}")
        print(f"STEP 1: MODIFY - Starting with {len(sentences)} original sentences")
        print(f"{'='*80}\n")
        
        modified_sentences, used_triplets = self.modify_sentences_with_keywords_and_triplets(
            sentences, modify_triplets, subgraph_info['keywords']
        )
        
        print(f"\n✅ MODIFY complete: {len(modified_sentences)} sentences (original: {len(sentences)})")
        
        # 원본 개수 보장 검증
        if len(modified_sentences) != len(sentences):
            print(f"⚠️  Warning: Sentence count changed during modification!")
            print(f"   Original: {len(sentences)}, Modified: {len(modified_sentences)}")
        
        # Step 2: Insert remaining triplets as new sentences (Modify에서 사용된 것 제외)
        remaining_for_insert = [t for t in insert_triplets if tuple(t) not in used_triplets]
        if remaining_for_insert:
            print(f"\n{'='*80}")
            print(f"STEP 2: INSERT - Starting with {len(modified_sentences)} sentences")
            print(f"{'='*80}\n")
            
            watermarked_sentences = self.insert_sentences_at_appropriate_positions(
                modified_sentences, remaining_for_insert, subgraph_info['keywords']
            )
            
            print(f"\n✅ INSERT complete: {len(watermarked_sentences)} total sentences")
        else:
            watermarked_sentences = modified_sentences
        
        # 원본 문장 보존을 위한 중복 제거
        print(f"🔍 Checking for duplicate sentences...")
        print(f"   Before duplicate removal: {len(watermarked_sentences)} sentences")
        watermarked_sentences = self._remove_duplicate_sentences_preserving_originals(
            watermarked_sentences, original_sentences
        )
        print(f"   After duplicate removal: {len(watermarked_sentences)} sentences")
        
        # Step 2.5: Verify and fix naturalness of sentences
        print(f"\n{'='*80}")
        print(f"STEP 2.5: NATURALNESS VERIFICATION - Checking sentence naturalness")
        print(f"{'='*80}\n")
        
        watermarked_sentences = self._verify_and_fix_naturalness(
            watermarked_sentences, original_sentences, subgraph_info['keywords']
        )
        
        print(f"\n✅ NATURALNESS VERIFICATION complete: {len(watermarked_sentences)} sentences")
        
        # Combine results
        watermarked_text = " ".join(watermarked_sentences)
        
        # Step 3: Verify entity preservation and retry if needed (Iterative)
        print(f"\n{'='*80}")
        print(f"STEP 3: VERIFICATION - Checking entity preservation")
        print(f"{'='*80}\n")
        
        max_retries = 5
        retry_count = 0
        verified_triplets = actual_triplets.copy()
        
        while retry_count < max_retries:
            verification_results = self._verify_triplet_entity_preservation(
                watermarked_text, verified_triplets
            )
            
            # Count successfully preserved triplets
            preserved_count = sum(1 for v in verification_results.values() if v["both_found"])
            total_to_verify = len(verified_triplets)
            preservation_rate = preserved_count / total_to_verify if total_to_verify > 0 else 0.0
            
            print(f"   Verification attempt {retry_count + 1}: {preserved_count}/{total_to_verify} triplets preserved ({preservation_rate*100:.1f}%)")
            
            # If preservation rate is acceptable (>= 50%) or all preserved, break
            if preservation_rate >= 0.8 or preserved_count == total_to_verify:
                print(f"   ✅ Entity preservation verified: {preserved_count}/{total_to_verify} triplets")
                break
            
            # Find failed triplets (not preserved) - convert tuple back to list
            failed_triplets = [list(t) for t, v in verification_results.items() if not v["both_found"]]
            
            if not failed_triplets or retry_count >= max_retries - 1:
                print(f"   ⚠️  Some triplets not fully preserved after {retry_count + 1} attempts")
                if failed_triplets:
                    print(f"   Failed triplets: {len(failed_triplets)}")
                break
            
            # Retry: Re-insert failed triplets
            print(f"   🔄 Retrying {len(failed_triplets)} failed triplets...")
            retry_count += 1
            
            # Re-insert failed triplets
            retry_inserted = self._retry_insert_failed_triplets(
                watermarked_sentences, failed_triplets, subgraph_info['keywords']
            )
            
            if retry_inserted:
                watermarked_sentences = retry_inserted
                watermarked_text = " ".join(watermarked_sentences)
                print(f"   ✅ Retry insertion complete: {len(watermarked_sentences)} sentences")
            else:
                print(f"   ⚠️  Retry insertion failed, keeping current result")
                break
        
        # 길이 증가 모니터링
        original_length = len(combined_text)
        watermarked_length = len(watermarked_text)
        length_increase_ratio = (watermarked_length - original_length) / original_length if original_length > 0 else 0
        
        # 길이 증가가 50%를 초과하면 경고 (더 유연한 기준)
        if length_increase_ratio > 0.5:
            print(f"   ⚠️  Warning: Document length increased by {length_increase_ratio:.1%} (max recommended: 50%)")
        elif length_increase_ratio > 0.3:
            print(f"   ℹ️  Info: Document length increased by {length_increase_ratio:.1%}")
        
        # 통계 계산
        actual_modified = sum(1 for i, (orig, mod) in enumerate(zip(sentences, modified_sentences)) if orig != mod)
        actual_inserted = len(insert_triplets)
        
        # modification_ratio와 insertion_ratio의 합이 100이 되도록 계산
        total_watermark_operations = actual_modified + actual_inserted
        if total_watermark_operations > 0:
            modification_ratio = (actual_modified / total_watermark_operations) * 100
            insertion_ratio = (actual_inserted / total_watermark_operations) * 100
        else:
            modification_ratio = 0.0
            insertion_ratio = 0.0
        
        return {
            "original_text": combined_text,
            "watermarked_text": watermarked_text,
            "keywords": subgraph_info['keywords'],
            "ratio": self.ratio,
            "total_triplets": total_triplets,
            "used_triplets": len(actual_triplets),
            "planned_modify": modify_count,
            "planned_insert": insert_count,
            "actual_modified_sentences": actual_modified,
            "actual_inserted_sentences": actual_inserted,
            "modification_ratio": modification_ratio,
            "insertion_ratio": insertion_ratio,
            "length_increase_ratio": length_increase_ratio,
            "original_length": original_length,
            "watermarked_length": watermarked_length,
            "subgraph_triples": subgraph_info['subgraph_triples'],
            "selected_triplets": actual_triplets
        }
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """두 벡터 간의 코사인 유사도 계산"""
        import numpy as np
        
        # 벡터 정규화
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 코사인 유사도 계산
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    
    def _analyze_document_complexity(self, sentences):
        """문서의 복잡도 분석"""
        if not sentences:
            return 0.5  # 기본값
        
        total_chars = 0
        total_words = 0
        total_sentences = len(sentences)
        complex_sentences = 0
        
        for sentence in sentences:
            # 기본 통계
            char_count = len(sentence)
            word_count = len(sentence.split())
            total_chars += char_count
            total_words += word_count
            
            # 복잡도 지표들
            is_complex = False
            
            # 1. 길이 기준 (문자 수)
            if char_count > 100:  # 100자 이상
                is_complex = True
            
            # 2. 단어 수 기준
            if word_count > 15:  # 15단어 이상
                is_complex = True
            
            # 3. 문법적 복잡도 (spaCy 분석)
            try:
                doc = self.nlp(sentence)
                
                # 복합문 (연결사가 있는 경우)
                conjunctions = [token for token in doc if token.pos_ == 'CCONJ']
                if len(conjunctions) > 0:
                    is_complex = True
                
                # 부사절 (관계사가 있는 경우)
                relative_clauses = [token for token in doc if token.dep_ in ['relcl', 'acl']]
                if len(relative_clauses) > 0:
                    is_complex = True
                
                # 구두점 복잡도 (쉼표, 세미콜론 등)
                punctuation_count = sum(1 for c in sentence if c in ';,:-')
                if punctuation_count > 2:
                    is_complex = True
                    
            except Exception:
                # spaCy 분석 실패 시 길이만으로 판단
                pass
            
            if is_complex:
                complex_sentences += 1
        
        # 복잡도 점수 계산 (0.0 ~ 1.0)
        avg_chars = total_chars / total_sentences if total_sentences > 0 else 0
        avg_words = total_words / total_sentences if total_sentences > 0 else 0
        complex_ratio = complex_sentences / total_sentences if total_sentences > 0 else 0
        
        # 가중 평균으로 최종 복잡도 계산
        complexity_score = (
            min(avg_chars / 150, 1.0) * 0.4 +  # 문자 수 (150자 기준)
            min(avg_words / 20, 1.0) * 0.3 +   # 단어 수 (20단어 기준)
            complex_ratio * 0.3                # 복잡한 문장 비율
        )
        
        return min(complexity_score, 1.0)
    
    def _calculate_adaptive_ratio(self, complexity_score):
        """복잡도에 따른 Modify/Insert 비율 계산"""
        # 복잡도가 낮으면 (짧고 간결) → Insert < Modify (0.3:0.7)
        # 복잡도가 높으면 (길고 복잡) → Insert > Modify (0.7:0.3)
        
        # 복잡도 0.0 ~ 1.0을 0.3 ~ 0.7로 매핑
        insert_ratio = 0.3 + (complexity_score * 0.4)
        modify_ratio = 1.0 - insert_ratio
        
        return modify_ratio, insert_ratio
    
    
    
    def _calculate_triplet_topic_similarity(self, triple, topic_embedding):
        """트리플렛과 메인 토픽 간의 유사도 계산 (enhanced for better theme matching)"""
        if topic_embedding is None:
            return 1.0  # 토픽이 없으면 모든 트리플렛 허용
        
        # Triplet의 entity name들을 직접 사용하여 더 정확한 유사도 계산
        try:
            h, r, t = triple
            
            # Entity와 relation name 가져오기
            h_name = self._select_best_name(h, self.entity)
            t_name = self._select_best_name(t, self.entity)
            r_name = self._select_best_name(r, self.relation)
            
            # Triple을 간단한 문장으로 표현
            if h_name and t_name and r_name:
                triple_text = f"{h_name} {r_name} {t_name}"
            else:
                triple_text = self.convert_triple_to_sentence(triple)
            
            if not triple_text:
                return 0.0
            
            sentence_embedding = self._get_sentence_embedding(triple_text)
            if sentence_embedding is None:
                return 0.0
            
            similarity = self._calculate_cosine_similarity(sentence_embedding, topic_embedding)
            return similarity
        except Exception as e:
            print(f"Error calculating triplet similarity: {e}")
            return 0.0
    
    def select_triplets_for_watermarking(self, subgraph_triples, keywords, keyword_triplets=None, main_topic=None,
                                         similarity_threshold=0.2, original_sentences=None, rarity_threshold=0.75):
        """Select triplets for watermarking based on keywords"""
        if not subgraph_triples:
            return [], 0, []
        
        triplets_list = list(subgraph_triples.values()) if isinstance(subgraph_triples, dict) else subgraph_triples
        
        # 메인 토픽 임베딩 계산 (전체 텍스트 사용)
        topic_embedding = None
        if main_topic:
            topic_embedding = self._get_sentence_embedding(main_topic)
        
        # 토픽 유사도 기반 필터링 (optional - lenient)
        filtered_triplets = []
        if topic_embedding is not None:
            for triple in triplets_list:
                similarity = self._calculate_triplet_topic_similarity(triple, topic_embedding)
                if similarity >= similarity_threshold:
                    filtered_triplets.append(triple)
        else:
            filtered_triplets = triplets_list
        
        print(f"📊 Theme filtering: {len(triplets_list)} -> {len(filtered_triplets)} triplets (threshold: {similarity_threshold if topic_embedding else 'N/A'})")
        
        # 필터링 결과가 너무 적으면 필터링 건너뛰기
        if len(filtered_triplets) < len(triplets_list) * 0.3:  # 30% 미만이면 필터링 건너뛰기
            print(f"   ⚠️ Too few triplets after theme filtering, using all triplets")
            filtered_triplets = triplets_list
        
        rarity_threshold = rarity_threshold if rarity_threshold is not None else 0.75
        rarity_filtered_triplets, rarity_details = self._filter_triplets_by_rarity(
            filtered_triplets, original_sentences, threshold=rarity_threshold
        )

        if rarity_details:
            rare_count = sum(1 for _, sim, _ in rarity_details if sim < rarity_threshold)
            print(f"🔎 Rarity filtering: kept {len(rarity_filtered_triplets)}/{len(filtered_triplets)} triplets below similarity {rarity_threshold:.2f}")
            # 필터링 결과가 너무 적으면 필터링 건너뛰기
            if len(rarity_filtered_triplets) < len(filtered_triplets) * 0.3:  # 30% 미만이면 필터링 건너뛰기
                print(f"   ⚠️ Too few triplets after rarity filtering, using theme-filtered triplets")
                rarity_filtered_triplets = filtered_triplets
            elif rare_count == 0:
                closest = ', '.join([
                    f"{trip[:3]} (sim={sim:.2f})" for trip, sim, _ in rarity_details[:3]
                ])
                print(f"   ⚠️ No triplets below threshold; using closest candidates: {closest}")
        else:
            print(f"🔎 Rarity filtering skipped (no embeddings available)")

        filtered_triplets = rarity_filtered_triplets
        
        # Ratio 기반 triplet 선택 (더 관대하게)
        # 원본 triplets_list의 개수를 기준으로 ratio 적용, 하지만 filtered_triplets 범위 내에서
        base_count = len(triplets_list)  # 원본 전체 개수
        ratio_based_count = max(3, int(base_count * self.ratio))  # Ratio 기반 목표 개수
        
        # Filtered triplets가 충분하면 ratio 기반 개수 사용, 부족하면 가능한 만큼 사용
        min_count = min(3, len(filtered_triplets))
        max_count = len(filtered_triplets)  # 필터링된 모든 triplet 사용 가능
        target_count = max(min_count, min(max_count, ratio_based_count))
        
        # 키워드별로 triplet 분류
        keyword_triplets_dict = {}
        other_triplets = []
        
        if filtered_triplets:
            for triple in filtered_triplets:
                if self._is_meaningful_triplet(triple, keywords):
                    # 어떤 키워드와 관련된지 찾기
                    matched_keyword = None
                    for keyword in keywords:
                        if self._is_triplet_related_to_keyword(triple, keyword):
                            if keyword not in keyword_triplets_dict:
                                keyword_triplets_dict[keyword] = []
                            keyword_triplets_dict[keyword].append(triple)
                            matched_keyword = keyword
                            break
                    
                    if not matched_keyword:
                        other_triplets.append(triple)
        
        # 키워드별로 최소 1개씩 선택
        selected_triplets = []
        used_triplets = set()
        
        # 각 키워드별로 최소 1개씩 선택
        for keyword in keywords:
            if keyword in keyword_triplets_dict and keyword_triplets_dict[keyword]:
                # 해당 키워드의 첫 번째 triplet 선택
                selected_triplet = keyword_triplets_dict[keyword][0]
                selected_triplets.append(selected_triplet)
                used_triplets.add(tuple(selected_triplet))
                print(f"   🎯 Selected triplet for keyword '{keyword}': {selected_triplet}")
        
        # 남은 triplet들 중에서 추가 선택 (filtered_triplets에서 가져옴)
        remaining_triplets = [t for t in filtered_triplets if tuple(t) not in used_triplets]
        
        # 목표 개수까지 추가 선택
        while len(selected_triplets) < target_count and remaining_triplets:
            selected_triplets.append(remaining_triplets.pop(0))
        
        print(f"🔍 Selected {len(selected_triplets)}/{target_count} triplets for watermarking")
        print(f"   - Keyword-related: {len([t for t in selected_triplets if any(self._is_triplet_related_to_keyword(t, k) for k in keywords)])}")
        print(f"   - Other: {len(selected_triplets) - len([t for t in selected_triplets if any(self._is_triplet_related_to_keyword(t, k) for k in keywords)])}")
        
        return selected_triplets, len(triplets_list), list(range(len(selected_triplets)))
    
    def _is_triplet_related_to_keyword(self, triple, keyword):
        """Triplet이 특정 키워드와 관련있는지 확인"""
        head, relation, tail = triple
        
        # Head나 Tail이 키워드와 매칭되는지 확인
        head_matches = self._is_entity_name_match(head, keyword)
        tail_matches = self._is_entity_name_match(tail, keyword)
        
        # 디버깅을 위한 출력
        if head_matches or tail_matches:
            print(f"   ✅ Triplet {triple} matches keyword '{keyword}' (head: {head_matches}, tail: {tail_matches})")
        
        return head_matches or tail_matches
    
    def _select_triplets_for_modify_and_insert(self, triplets, sentences, keywords, modify_count, insert_count):
        """
        Modify와 Insert에 적합한 triplet 선택 (더 관대한 기준)
        
        기준:
        - Modify: 키워드 관련 triplet 우선
        - Insert: 나머지 모든 triplet
        """
        if not triplets:
            return [], []
        
        # 1. 키워드 관련 triplet 분류
        keyword_related_triplets = []
        other_triplets = []
        
        for triple in triplets:
            is_keyword_related = any(self._is_triplet_related_to_keyword(triple, k) for k in keywords)
            
            if is_keyword_related:
                keyword_related_triplets.append(triple)
            else:
                other_triplets.append(triple)
        
        # 2. Modify용 triplet 선택: 키워드 관련 triplet 우선
        modify_triplets = keyword_related_triplets[:modify_count]
        
        # 부족하면 다른 triplet 추가
        if len(modify_triplets) < modify_count and other_triplets:
            remaining = modify_count - len(modify_triplets)
            modify_triplets.extend(other_triplets[:remaining])
        
        # 3. Insert용 triplet 선택: 나머지 모든 triplet
        remaining_triplets = [t for t in triplets if t not in modify_triplets]
        insert_triplets = remaining_triplets[:insert_count]
        
        print(f"   📊 Triplet classification: {len(keyword_related_triplets)} keyword-related, {len(other_triplets)} others")
        print(f"   📊 Assigned: {len(modify_triplets)} for Modify, {len(insert_triplets)} for Insert")
        
        return modify_triplets, insert_triplets
    
    def _is_informative_triplet(self, triple):
        """Informative한 관계인지 확인"""
        head, relation, tail = triple
        
        # Informative relations
        informative_relations = [
            'located in', 'in', 'at', 'from', 'originated in',
            'has', 'contains', 'includes', 'features',
            'part of', 'belongs to', 'member of',
            'founded by', 'created by', 'established by',
            'born in', 'works for', 'employed by'
        ]
        
        relation_lower = relation.lower()
        return any(inf_rel in relation_lower for inf_rel in informative_relations)
    
    def _is_meaningful_triplet(self, triple, keywords):
        """의미있는 triplet인지 확인 (enhanced for better theme matching)"""
        head, relation, tail = triple
        
        # 의미있는 관계인지 확인
        meaningful_relations = [
            'is a', 'is an', 'instance of', 'type of', 'class of',
            'has', 'contains', 'includes', 'features',
            'part of', 'belongs to', 'member of', 'component of',
            'located in', 'in', 'at', 'situated in',
            'founded by', 'created by', 'established by', 'started by',
            'works for', 'employed by', 'works at',
            'born in', 'from', 'originated in',
            'developer', 'developed by', 'created by',
            'maker', 'manufacturer', 'producer'
        ]
        
        relation_lower = relation.lower()
        is_meaningful_relation = any(meaningful_rel in relation_lower for meaningful_rel in meaningful_relations)
        
        # 키워드와 관련된 triplet 우선 선택
        is_keyword_related = False
        for keyword in keywords:
            if (self._is_entity_name_match(head, keyword) or 
                self._is_entity_name_match(tail, keyword)):
                is_keyword_related = True
                break
        
        # 더 엄격하게: 키워드 관련이거나 의미있는 관계이면 허용하되, 키워드 관련 triplet 우선
        return is_meaningful_relation or is_keyword_related
    
    def modify_sentences_with_keywords_and_triplets(self, sentences, triplets, keywords):
        """LLM-based sentence modification with triplet integration (RAG/CoT approach)"""
        if not triplets or not sentences or not keywords:
            return sentences, set()
        
        print(f"📝 Modifying sentences with RAG/CoT approach...")
        print(f"   Original sentences: {len(sentences)}")
        
        modified_sentences = []
        used_triplets = set()
        available_triplets = triplets.copy()
        modification_indices = []  # Track which sentences were modified
        
        for i, sentence in enumerate(sentences):
            # Find relevant triplet for this sentence
            best_triplet = self._find_relevant_triplet_for_sentence(sentence, available_triplets)
            
            if best_triplet:
                # LLM-based modification with RAG approach
                modified_sentence = self._llm_modify_sentence_with_triplet(sentence, best_triplet, keywords)
                
                if modified_sentence and modified_sentence != sentence:
                    # Only replace if successfully modified
                    modified_sentences.append(modified_sentence)
                    modification_indices.append(i)
                    used_triplets.add(tuple(best_triplet))
                    available_triplets.remove(best_triplet)
                    print(f"   ✅ Modified sentence {i+1}")
                else:
                    # Keep original if modification failed or same
                    modified_sentences.append(sentence)
                    print(f"   ⚪ Kept original sentence {i+1} (no modification or failed)")
            else:
                # Keep original if no triplet matched
                modified_sentences.append(sentence)
        
        print(f"   ✅ Modified {len(modification_indices)}/{len(sentences)} sentences")
        print(f"   Final sentences: {len(modified_sentences)} (should be {len(sentences)})")
        return modified_sentences, used_triplets
    
    def _llm_modify_sentence_with_triplet(self, sentence, triplet, keywords):
        """Use LLM to naturally integrate triplet into sentence (RAG approach)"""
        try:
            h, r, t = triplet
            h_name = self._select_best_name(h, self.entity)
            r_name = self._select_best_name(r, self.relation)
            t_name = self._select_best_name(t, self.entity)
            
            # RAG-style prompt for natural integration with grammar and proper noun preservation
            prompt = f"""Modify the following sentence to naturally include the given fact while preserving ALL original content and ensuring grammatical correctness.

Original sentence: {sentence}
Fact to integrate: ({h_name}, {r_name}, {t_name})

Document context: {', '.join(keywords)}

CRITICAL REQUIREMENTS:
- Preserve ALL original words and content from the original sentence
- Keep proper nouns capitalized correctly (e.g., Apple, California, New York)
- Maintain grammatical correctness and sentence flow
- The fact should be integrated smoothly without changing sentence structure
- Do not add unnecessary words or change core meaning
- Ensure the sentence sounds natural and professional
- Output ONLY the complete modified sentence, nothing else

Modified sentence:"""

            response = self.llm.generate(prompt, max_tokens=80, temperature=0.3)
            modified = response.strip().replace('Modified sentence:', '').strip() if response else ""
            
            # 빈 응답 체크 및 fallback
            if not modified or len(modified) < 10:
                if not modified:
                    print(f"   ⚠️  LLM returned empty response for modification, using fallback")
                # Fallback: 간단한 수정 방식 사용
                modified = self._simple_modify_sentence(sentence, triplet, keywords)
                if not modified or modified == sentence:
                    # Fallback도 실패하면 원본 반환
                    return sentence
            
            # Preserve proper nouns from original
            modified = self._preserve_proper_nouns(sentence, modified)
            
            # Quality check
            if len(modified) > 300:
                # 너무 길면 fallback 사용
                modified = self._simple_modify_sentence(sentence, triplet, keywords)
                if not modified or modified == sentence:
                    return sentence
            
            if not modified.endswith(('.', '!', '?')):
                modified += '.'
            
            return modified
            
        except Exception as e:
            print(f"   ⚠️  LLM modification failed: {e}, using fallback")
            # Fallback: 간단한 수정 방식 사용
            try:
                modified = self._simple_modify_sentence(sentence, triplet, keywords)
                if modified and modified != sentence:
                    return modified
            except:
                pass
            return sentence
    
    def _preserve_proper_nouns(self, original, modified):
        """Preserve proper nouns from original sentence"""
        import re
        try:
            # Extract proper nouns from original (capitalized words)
            doc_original = self.nlp(original)
            proper_nouns = set()
            for token in doc_original:
                if token.text[0].isupper() and len(token.text) > 1:
                    proper_nouns.add(token.text)
            
            # Replace in modified if different
            doc_modified = self.nlp(modified)
            words = modified.split()
            result_words = []
            
            for word in words:
                # Remove punctuation for comparison
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word in proper_nouns:
                    # Preserve original capitalization
                    original_word = [w for w in proper_nouns if w.lower() == clean_word.lower()]
                    if original_word:
                        # Keep the original capitalization
                        if word[0].isupper():
                            result_words.append(original_word[0] + word[len(clean_word):])
                        else:
                            result_words.append(word)
                    else:
                        result_words.append(word)
                else:
                    result_words.append(word)
            
            return ' '.join(result_words)
        except:
            return modified
    
    def _find_relevant_triplet_for_sentence(self, sentence, triplets):
        """Find relevant triplet for a sentence using semantic similarity"""
        if not triplets:
            return None
        
        try:
            # Get sentence embedding
            sent_embed = self._get_sentence_embedding(sentence)
            if sent_embed is None:
                return triplets[0] if triplets else None
            
            best_triplet = None
            best_sim = 0.1  # Lower threshold to allow more matches
            
            for triplet in triplets:
                if len(triplet) >= 3:
                    # Create triplet text for embedding
                    triplet_text = f"{self._select_best_name(triplet[0], self.entity)} {self._select_best_name(triplet[1], self.relation)} {self._select_best_name(triplet[2], self.entity)}"
                    trip_embed = self._get_sentence_embedding(triplet_text)
                    
                    if trip_embed is not None:
                        sim = self._calculate_cosine_similarity(sent_embed, trip_embed)
                        if sim > best_sim:
                            best_sim = sim
                            best_triplet = triplet
            
            # Fallback: If no triplet found with similarity threshold, use first available triplet
            if best_triplet is None and triplets:
                print(f"   ⚠️  No triplet found with similarity > 0.1, using first available triplet")
                return triplets[0]
            
            return best_triplet
        except Exception as e:
            return triplets[0] if triplets else None
    
    def _find_best_triplet_for_sentence(self, sentence, triplets, sentence_keywords):
        """문장에 가장 적합한 triplet 찾기 (더 관대한 조건)"""
        if not triplets:
            return None
    
        best_triplet = None
        best_score = -1
        
        for triplet in triplets:
            if len(triplet) >= 3:
                head, relation, tail = triplet
                
                # 기본 점수 (모든 triplet에 기본 점수 부여)
                score = 1
                
                # 키워드와의 관련성 점수 계산 (있는 경우에만)
                if sentence_keywords:
                    for keyword in sentence_keywords:
                        if keyword.lower() in head.lower() or keyword.lower() in tail.lower():
                            score += 2
                        if keyword.lower() in relation.lower():
                            score += 1
                
                # 문장 내용과의 관련성 확인 (더 관대한 매칭)
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in head.lower().split()):
                    score += 1
                if any(word in sentence_lower for word in tail.lower().split()):
                    score += 1
                
                # 부분 매칭도 고려
                if any(word in sentence_lower for word in head.lower().split() if len(word) > 3):
                    score += 0.5
                if any(word in sentence_lower for word in tail.lower().split() if len(word) > 3):
                    score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_triplet = triplet
        
        # 더 관대한 조건: 점수가 0.5 이상이면 사용
        return best_triplet if best_score >= 0.5 else None
    
    def _simple_modify_sentence(self, sentence, triplet, keywords):
        """간단한 문장 수정 (길이 제한)"""
        try:
            head, relation, tail = triplet
            
            # 이름 선택
            head_name = self._select_best_name(head, self.entity)
            relation_name = self._select_best_name(relation, self.relation)
            tail_name = self._select_best_name(tail, self.entity)
            
            if not all([head_name, relation_name, tail_name]):
                return sentence
            
            # 간단한 삽입 패턴들 (길이 제한)
            if relation_name.lower() in ['located', 'based', 'headquartered']:
                # 위치 정보 추가
                if sentence.endswith('.'):
                    return sentence[:-1] + f", {relation_name} in {tail_name}."
                else:
                    return sentence + f", {relation_name} in {tail_name}."
            
            elif relation_name.lower() in ['founded', 'established', 'created']:
                # 설립 정보 추가
                if sentence.endswith('.'):
                    return sentence[:-1] + f", {relation_name} in {tail_name}."
                else:
                    return sentence + f", {relation_name} in {tail_name}."
            
            elif relation_name.lower() in ['developer', 'producer', 'manufacturer']:
                # 개발/제조 정보 추가
                if sentence.endswith('.'):
                    return sentence[:-1] + f" ({relation_name} by {tail_name})."
                else:
                    return sentence + f" ({relation_name} by {tail_name})."
            
            else:
                # 기본 패턴
                if sentence.endswith('.'):
                    return sentence[:-1] + f" ({relation_name} {tail_name})."
                else:
                    return sentence + f" ({relation_name} {tail_name})."
                    
        except Exception as e:
            print(f"Error in simple sentence modification: {e}")
            return sentence
    
    def _llm_modify_sentences_with_context(self, sentences, triplets, keywords):
        """LLM을 사용하여 문장들을 문맥에 맞게 지능적으로 수정"""
        try:
            # triplet들을 자연스러운 문장으로 변환
            triplet_sentences = []
            for triplet in triplets:
                sentence = self.convert_triplet_to_sentence_with_llm(triplet, keywords)
                triplet_sentences.append(sentence)
            
            # 현재 문서의 문맥 정보 수집
            context_info = self._analyze_document_context(sentences, keywords)
            
            # LLM 프롬프트 생성
            prompt = self._create_sentence_modification_prompt(sentences, triplet_sentences, context_info, keywords)
            
            # LLM 호출
            response = self.llm.generate(
                prompt, 
                max_tokens=600,
                temperature=0.6,
                do_sample=True
            )
            
            # 응답에서 수정된 문장들 추출
            modified_sentences = self._parse_llm_modification_response(response, sentences)
            
            # 사용된 triplet들 추적 (간단한 휴리스틱)
            used_triplets = set()
            for i, (orig, mod) in enumerate(zip(sentences, modified_sentences)):
                if orig != mod:
                    # 수정된 문장에 사용된 triplet 찾기
                    for j, triplet in enumerate(triplets):
                        if j < len(triplet_sentences):
                            triplet_text = triplet_sentences[j]
                            if any(word in mod.lower() for word in triplet_text.lower().split()[:3]):
                                used_triplets.add(tuple(triplet))
            
            print(f"   ✅ Successfully modified {len([i for i, (orig, mod) in enumerate(zip(sentences, modified_sentences)) if orig != mod])} sentences using LLM")
            return modified_sentences, used_triplets
                
        except Exception as e:
            print(f"   ❌ LLM modification failed: {e}")
            # 실패 시 원본 문장 반환
            return sentences, set()
    
    def _create_sentence_modification_prompt(self, sentences, triplet_sentences, context_info, keywords):
        """문장 수정을 위한 LLM 프롬프트 생성"""
        sentences_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
        triplets_text = "\n".join([f"- {s}" for s in triplet_sentences])
        
        prompt = f"""Enhance these sentences by naturally integrating relevant facts while preserving original meaning.

        SENTENCES:
        {sentences_text}

        FACTS TO INTEGRATE:
        {triplets_text}

        TOPICS: {', '.join(keywords)}

        RULES:
        - Keep original meaning intact
        - Add relevant facts naturally using connectors like "which", "that", "additionally"
        - Only modify sentences where facts fit naturally
        - Keep additions concise (max 15 words per addition)
        - Maintain consistent tone

        MODIFIED SENTENCES:"""
        
        return prompt
    
    def _parse_llm_modification_response(self, response, original_sentences):
        """LLM 수정 응답에서 문장들을 추출"""
        try:
            # 응답에서 번호가 매겨진 문장들 추출
            lines = response.strip().split('\n')
            modified_sentences = []
            
            for line in lines:
                line = line.strip()
                # 번호가 매겨진 문장 패턴 찾기 (예: "1. 문장내용")
                if re.match(r'^\d+\.\s+', line):
                    sentence = re.sub(r'^\d+\.\s+', '', line)
                    if sentence and len(sentence) > 5:  # 최소 길이 확인
                        modified_sentences.append(sentence)
            
            # 추출된 문장 수가 원본과 다르면 원본 반환
            if len(modified_sentences) != len(original_sentences):
                print(f"   ⚠️  LLM response parsing incomplete, using original sentences")
                return original_sentences
            
            return modified_sentences
            
        except Exception as e:
            print(f"   ⚠️  Error parsing LLM modification response: {e}")
            return original_sentences
    
    def _calculate_triplet_sentence_similarity(self, triplet, sentence):
        """Triplet의 head, relation, tail과 문장 간의 유사도 계산"""
        head, relation, tail = triplet
        
        # 각 구성요소의 이름 가져오기
        head_name = self._select_best_name(head, self.entity)
        relation_name = self._select_best_name(relation, self.relation)
        tail_name = self._select_best_name(tail, self.entity)
        
        # 각 구성요소와 문장의 유사도 계산
        head_embedding = self._get_sentence_embedding(head_name)
        relation_embedding = self._get_sentence_embedding(relation_name)
        tail_embedding = self._get_sentence_embedding(tail_name)
        sentence_embedding = self._get_sentence_embedding(sentence)
        
        if not all([head_embedding is not None, relation_embedding is not None, 
                   tail_embedding is not None, sentence_embedding is not None]):
            return 0.0
        
        # 각 구성요소와의 유사도 계산 후 평균
        head_sim = self._calculate_cosine_similarity(head_embedding, sentence_embedding)
        relation_sim = self._calculate_cosine_similarity(relation_embedding, sentence_embedding)
        tail_sim = self._calculate_cosine_similarity(tail_embedding, sentence_embedding)
        
        # 가중 평균 (head와 tail에 더 높은 가중치)
        avg_similarity = (head_sim * 0.4 + relation_sim * 0.2 + tail_sim * 0.4)
        return avg_similarity
    
    
    
    def convert_triplet_to_sentence_with_llm(self, triplet, keywords=None):
        """RAG 스타일로 triplet을 자연스러운 문장으로 변환"""
        if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
            return str(triplet)
        
        head, relation, tail = triplet
        
        # 각 구성요소의 이름 가져오기
        head_name = self._select_best_name(head, self.entity, keywords)
        relation_name = self._select_best_name(relation, self.relation)
        tail_name = self._select_best_name(tail, self.entity, keywords)
        
        # RAG 스타일 프롬프트 생성
        prompt = f"""Convert this knowledge into a natural, informative sentence.

SUBJECT: {head_name}
RELATION: {relation_name}
OBJECT: {tail_name}

CONTEXT: Document about {', '.join(keywords) if keywords else 'general topics'}

Create a single, grammatically correct sentence (max 20 words) that sounds natural and professional.

SENTENCE:"""

        try:
            # LLM 호출
            response = self.llm.generate(
                prompt,
                max_tokens=40,
                temperature=0.7,
                do_sample=True
            )
            
            # 응답에서 문장 추출 및 정리
            generated_sentence = response.strip() if response else ""
            
            # 빈 응답 또는 기본 품질 검사 (길이 제한 강화)
            if not generated_sentence or len(generated_sentence) < 10 or len(generated_sentence) > 150:
                if not generated_sentence:
                    print(f"   ⚠️  LLM returned empty response, using fallback")
                return self._create_fallback_sentence(head_name, relation_name, tail_name)
            
            # 단어 수 제한 (20단어 이하)
            word_count = len(generated_sentence.split())
            if word_count > 20:
                # 20단어로 자르고 자연스럽게 마무리
                words = generated_sentence.split()[:20]
                generated_sentence = ' '.join(words)
                if not generated_sentence.endswith(('.', '!', '?')):
                    generated_sentence += '.'
            
            # 문장이 제대로 끝나지 않으면 마침표 추가
            if not generated_sentence.endswith(('.', '!', '?')):
                generated_sentence += '.'
            
            return generated_sentence
            
        except Exception as e:
            print(f"   Error in LLM triplet conversion: {e}")
            return self._create_fallback_sentence(head_name, relation_name, tail_name)
    
    def _create_fallback_sentence(self, head_name, relation_name, tail_name):
        """LLM 실패 시 사용할 기본 문장 패턴 (간결 버전)"""
        relation_lower = relation_name.lower()
        
        # 이름 길이 제한 (10자 이하)
        if len(head_name) > 10:
            head_name = head_name[:10] + "..."
        if len(tail_name) > 10:
            tail_name = tail_name[:10] + "..."
        
        # 기본 패턴들 (간결하게)
        if 'is a' in relation_lower or 'instance of' in relation_lower:
            sentence = f"{head_name} is a {tail_name}."
        elif 'has' in relation_lower or 'contains' in relation_lower:
            sentence = f"{head_name} has {tail_name}."
        elif 'located in' in relation_lower or 'in' in relation_lower:
            sentence = f"{head_name} is in {tail_name}."
        elif 'founded by' in relation_lower or 'created by' in relation_lower:
            sentence = f"{head_name} was founded by {tail_name}."
        elif 'part of' in relation_lower:
            sentence = f"{head_name} is part of {tail_name}."
        else:
            sentence = f"{head_name} {relation_name} {tail_name}."
        
        # 최종 길이 제한 (50자 이하)
        if len(sentence) > 50:
            sentence = f"{head_name} is related to {tail_name}."
        
        return sentence
    
    def _is_unnatural_sentence(self, sentence):
        """문장이 부자연스러운지 판단 (동적 방법)"""
        if not sentence or len(sentence) < 5:
            return True
        
        # 1. 문법적 구조 검사
        if not self._has_valid_grammatical_structure(sentence):
            return True
        
        # 2. 반복되는 단어나 구문 검사
        if self._has_repetitive_patterns(sentence):
            return True
        
        # 3. 의미적 일관성 검사
        if not self._has_semantic_coherence(sentence):
            return True
        
        # 4. 문장 길이와 복잡도 검사
        if not self._has_appropriate_length_and_complexity(sentence):
            return True
        
        return False
    
    def _has_valid_grammatical_structure(self, sentence):
        """문법적 구조가 유효한지 검사"""
        try:
            doc = self.nlp(sentence)
            
            # 문장이 제대로 시작하는지
            if not sentence[0].isupper():
                return False
            
            # 주어와 동사가 있는지
            has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
            has_verb = any(token.pos_ == 'VERB' for token in doc)
            
            if not has_subject or not has_verb:
                return False
            
            # 너무 많은 특수문자
            special_chars = sum(1 for c in sentence if not c.isalnum() and c != ' ' and c != '.' and c != ',' and c != '!' and c != '?')
            if special_chars > len(sentence) * 0.15:  # 15% 이상이 특수문자
                return False
            
            return True
        except:
            return False
    
    def _has_repetitive_patterns(self, sentence):
        """반복되는 패턴이 있는지 검사"""
        words = sentence.lower().split()
        
        # 같은 단어가 3번 이상 반복
        word_counts = {}
        for word in words:
            if len(word) > 2:  # 2글자 이상인 단어만
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] >= 3:
                    return True
        
        # 연속된 같은 단어 (예: "apple apple apple")
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
        
        # 비슷한 단어가 연속으로 나오는 패턴 (예: "california state california")
        for i in range(len(words) - 2):
            if words[i] == words[i+2] and words[i+1] in ['state', 'is', 'a', 'an', 'the']:
                return True
        
        return False
    
    def _has_semantic_coherence(self, sentence):
        """의미적 일관성이 있는지 검사"""
        try:
            doc = self.nlp(sentence)
            
            # 명사와 동사의 관계가 적절한지
            nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            verbs = [token.text for token in doc if token.pos_ == 'VERB']
            
            # 명사가 너무 많고 동사가 적으면 부자연스러움
            if len(nouns) > 5 and len(verbs) < 2:
                return False
            
            # 의미가 모호한 단어 조합 검사 (일반적인 문법 오류)
            sentence_lower = sentence.lower()
            ambiguous_patterns = [
                'is a lists',  # 문법 오류
                'type of technolog',  # 철자 오류
                'pull media company',  # 의미 모호
            ]
            
            for pattern in ambiguous_patterns:
                if pattern in sentence_lower:
                    return False
            
            return True
        except:
            return True  # 에러 시 통과
    
    def _has_appropriate_length_and_complexity(self, sentence):
        """적절한 길이와 복잡도를 가지는지 검사"""
        # 너무 짧거나 긴 문장
        if len(sentence) < 10 or len(sentence) > 300:
            return False
        
        # 단어 수가 적절한지
        word_count = len(sentence.split())
        if word_count < 3 or word_count > 50:
            return False
        
        # 문장이 제대로 끝나는지
        if not sentence.rstrip().endswith(('.', '!', '?')):
            return False
        
        return True
    
    def _verify_and_fix_naturalness(self, sentences, original_sentences, keywords):
        """문장들의 자연스러움을 검증하고 수정 (원본 문장만 되돌리기, 새로 삽입된 문장은 수정)"""
        if not sentences:
            return sentences
        
        print(f"🔍 Verifying naturalness of {len(sentences)} sentences...")
        
        verified_sentences = []
        original_set = set(s.strip().lower() for s in original_sentences) if original_sentences else set()
        unnatural_count = 0
        fixed_count = 0
        reverted_count = 0
        
        for i, sentence in enumerate(sentences):
            is_unnatural = self._is_unnatural_sentence(sentence)
            
            if is_unnatural:
                unnatural_count += 1
                sentence_normalized = sentence.strip().lower()
                
                # 원본 문장이고 인덱스가 원본 범위 내인 경우만 원본으로 되돌리기
                if i < len(original_sentences) and sentence_normalized in original_set:
                    # 원본에서 정확히 일치하는 문장 찾기
                    found_original = False
                    for orig in original_sentences:
                        if orig.strip().lower() == sentence_normalized:
                            verified_sentences.append(orig)
                            reverted_count += 1
                            print(f"   🔄 Reverted to original (sentence {i+1}): {sentence[:50]}...")
                            found_original = True
                            break
                    
                    if not found_original:
                        # 인덱스로 원본 문장 가져오기
                        verified_sentences.append(original_sentences[i])
                        reverted_count += 1
                        print(f"   🔄 Reverted to original by index (sentence {i+1})")
                else:
                    # 새로 삽입된 문장이거나 원본이 아닌 경우 LLM으로 수정 시도
                    fixed = self._fix_unnatural_sentence_with_llm(sentence, keywords)
                    if fixed and not self._is_unnatural_sentence(fixed):
                        verified_sentences.append(fixed)
                        fixed_count += 1
                        print(f"   ✅ Fixed with LLM (sentence {i+1}): {sentence[:50]}... → {fixed[:50]}...")
                    else:
                        # 수정 실패 시 원본 유지 (삽입된 문장은 유지)
                        verified_sentences.append(sentence)
                        print(f"   ⚠️  Could not fix (sentence {i+1}), keeping as-is")
            else:
                # 자연스러운 문장은 그대로 유지
                verified_sentences.append(sentence)
        
        if unnatural_count > 0:
            print(f"   📊 Naturalness verification results:")
            print(f"      - Unnatural sentences detected: {unnatural_count}")
            print(f"      - Fixed with LLM: {fixed_count}")
            print(f"      - Reverted to original: {reverted_count}")
            print(f"      - Remaining unnatural: {unnatural_count - fixed_count - reverted_count}")
        else:
            print(f"   ✅ All sentences passed naturalness verification")
        
        return verified_sentences
    
    def _fix_unnatural_sentence_with_llm(self, sentence, keywords):
        """LLM을 사용하여 부자연스러운 문장을 수정"""
        try:
            prompt = f"""Fix the following sentence to make it more natural and grammatically correct while preserving its core meaning.

Sentence to fix: {sentence}

Document context: {', '.join(keywords) if keywords else 'general topics'}

REQUIREMENTS:
- Fix grammatical errors and awkward phrasing
- Make the sentence sound natural and professional
- Preserve the core meaning and key information
- Ensure proper capitalization and punctuation
- Output ONLY the corrected sentence, nothing else

Corrected sentence:"""

            response = self.llm.generate(prompt, max_tokens=60, temperature=0.3)
            fixed = response.strip() if response else ""
            
            # 빈 응답 또는 품질 검사
            if not fixed or len(fixed) < 5:
                return None
            
            # 너무 긴 경우도 제외
            if len(fixed) > 300:
                return None
            
            # 문장 끝 마침표 확인
            if not fixed.endswith(('.', '!', '?')):
                fixed += '.'
            
            return fixed
            
        except Exception as e:
            print(f"   ⚠️  Error fixing sentence with LLM: {e}")
            return None
    
    def _fix_grammar_with_llm(self, sentences):
        """간단한 규칙 기반 문법 수정 (LLM 대신)"""
        if not sentences:
            return sentences
        
        corrected_sentences = []
        
        for sentence in sentences:
            try:
                corrected = sentence
                
                # 1. 관사 오류 수정 (a/an)
                import re
                # "an" + 자음으로 시작하는 단어 → "a"
                corrected = re.sub(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])', r'a \1', corrected, flags=re.IGNORECASE)
                
                # "a" + 모음으로 시작하는 단어 → "an" (단, 대문자 단어는 제외)
                corrected = re.sub(r'\ba\s+([aeiouAEIOU][a-z])', r'an \1', corrected)
                
                # 2. 대문자 문제 수정 (NER 활용)
                try:
                    # 기존 self.nlp를 사용하여 고유명사 식별
                    doc = self.nlp(corrected)
                    
                    # NER 엔티티들을 추출
                    entities = set()
                    for ent in doc.ents:
                        # 엔티티의 각 단어를 추가
                        for token in ent.text.split():
                            entities.add(token.strip())
                    
                    words = corrected.split()
                    fixed_words = []
                    
                    for i, word in enumerate(words):
                        # 전체 대문자 단어 처리
                        if word.isupper() and len(word) > 2:
                            # NER에서 식별된 엔티티인지 확인
                            if word in entities:
                                # 고유명사는 그대로 유지
                                fixed_words.append(word)
                            elif i == 0:
                                # 문장 첫 단어는 첫 글자만 대문자
                                fixed_words.append(word.capitalize())
                            else:
                                # 문장 중간의 일반 단어는 소문자화
                                fixed_words.append(word.lower())
                        # 부분 대문자 단어 (예: "SOFTWARE", "Linux")
                        elif word.isupper() and len(word) > 2 and word.lower() not in ['i', 'a', 'an', 'the']:
                            # NER에서 식별된 엔티티인지 확인
                            if word in entities:
                                # 고유명사는 그대로 유지
                                fixed_words.append(word)
                            elif i == 0:
                                # 문장 첫 단어는 첫 글자만 대문자
                                fixed_words.append(word.capitalize())
                            else:
                                # 문장 중간의 일반 단어는 소문자화
                                fixed_words.append(word.lower())
                        else:
                            fixed_words.append(word)
                            
                except Exception as e:
                    # NER 실패 시 기본 규칙 적용
                    print(f"   ⚠️  NER failed, using basic rules: {e}")
                    words = corrected.split()
                    fixed_words = []
                    
                    for i, word in enumerate(words):
                        if word.isupper() and len(word) > 2:
                            if i == 0:
                                fixed_words.append(word.capitalize())
                            else:
                                fixed_words.append(word.lower())
                        else:
                            fixed_words.append(word)
                
                corrected = ' '.join(fixed_words)
                
                # 3. 중복 공백 제거
                corrected = re.sub(r'\s+', ' ', corrected)
                
                # 4. 문장 끝 마침표 확인
                corrected = corrected.strip()
                if corrected and not corrected.endswith(('.', '!', '?')):
                    corrected += '.'
                
                corrected_sentences.append(corrected)
                
                if corrected != sentence:
                    print(f"   ✅ Fixed: {sentence[:30]}... → {corrected[:30]}...")
                    
            except Exception as e:
                corrected_sentences.append(sentence)
                print(f"   ⚠️  Error fixing sentence: {e}")
        
        print(f"   ✅ Grammar fixed {len([s for s in corrected_sentences if s != sentences[corrected_sentences.index(s)]])} sentences")
        return corrected_sentences
    
    def _remove_duplicate_sentences_preserving_originals(self, watermarked_sentences, original_sentences):
        """중복 문장 제거 - 원본 문장들은 절대 삭제하지 않음"""
        if not watermarked_sentences:
            return watermarked_sentences
        
        # 원본 문장들의 수정 버전도 추적
        # 실제로는 모든 문장을 유지하는 것이 가장 안전
        
        seen = set()
        unique_sentences = []
        duplicates_removed = 0
        
        for sentence in watermarked_sentences:
            normalized = sentence.strip().lower()
            
            if normalized not in seen:
                unique_sentences.append(sentence)
                seen.add(normalized)
            else:
                # 중복 발견
                duplicates_removed += 1
                print(f"   🗑️  Removed duplicate (attempt to preserve original content)")
        
        if duplicates_removed > 0:
            print(f"   ✅ Removed {duplicates_removed} duplicate sentences")
            print(f"   Final: {len(unique_sentences)} unique sentences (original: {len(original_sentences)})")
        else:
            print(f"   ✓ No duplicates found")
        
        return unique_sentences
    
    def _remove_duplicate_sentences(self, sentences):
        """중복 문장 제거 (순서 유지) - legacy"""
        if not sentences:
            return sentences
        
        seen = set()
        unique_sentences = []
        duplicates_removed = 0
        
        for sentence in sentences:
            # 문장을 정규화하여 비교 (공백, 대소문자 무시)
            normalized = sentence.strip().lower()
            
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)
            else:
                duplicates_removed += 1
                print(f"   🗑️  Removed duplicate: {sentence[:50]}...")
        
        if duplicates_removed > 0:
            print(f"   ✅ Removed {duplicates_removed} duplicate sentences")
        
        return unique_sentences
    
    def insert_sentences_at_appropriate_positions(self, sentences, triplets, keywords):
        """Insert sentences using LLM with context-aware bridge generation (CoT approach)"""
        if not triplets or not sentences:
            return sentences
        
        print(f"➕ Inserting {len(triplets)} triplets with context-aware bridging...")
        print(f"   Original sentences: {len(sentences)}")
        
        # Start with all original sentences
        result_sentences = sentences.copy()
        original_count = len(sentences)
        
        for i, triplet in enumerate(triplets):
            if len(triplet) >= 3:
                # Find best insertion position
                position = self._find_best_insertion_position(result_sentences, triplet, keywords)
                
                # Generate bridge sentence with context
                bridge_sentence = self._llm_generate_bridge_sentence(
                    result_sentences, position, triplet, keywords
                )
                
                if bridge_sentence and bridge_sentence.strip():
                    result_sentences.insert(position, bridge_sentence)
                    print(f"   ✅ Inserted triplet {i+1} at position {position}: {bridge_sentence[:50]}...")
                else:
                    print(f"   ⚠️  Skipped triplet {i+1} (empty or None bridge sentence)")
        
        print(f"   Final sentences: {len(result_sentences)} (original: {original_count}, inserted: {len(result_sentences) - original_count})")
        return result_sentences
    
    def _llm_generate_bridge_sentence(self, sentences, position, triplet, keywords):
        """Generate natural bridge sentence between context (CoT approach)"""
        try:
            h, r, t = triplet
            h_name = self._select_best_name(h, self.entity)
            r_name = self._select_best_name(r, self.relation)
            t_name = self._select_best_name(t, self.entity)
            
            # Get context
            prev_context = " ".join(sentences[max(0, position-2):position]) if position > 0 else ""
            next_context = " ".join(sentences[position:min(len(sentences), position+2)]) if position < len(sentences) else ""
            
            # CoT-style prompt for bridge generation with grammar preservation
            prompt = f"""Generate a grammatically correct connecting sentence that smoothly bridges between two paragraphs while incorporating the given fact.

Previous context: {prev_context if prev_context else '(beginning of document)'}
Next context: {next_context if next_context else '(end of document)'}
Fact to incorporate: ({h_name}, {r_name}, {t_name})

Document context: {', '.join(keywords)}

REQUIREMENTS:
- Create a natural transition sentence (15-25 words)
- Ensure proper grammar and sentence structure
- Capitalize proper nouns correctly ({h_name}, {t_name} should keep their original capitalization)
- Use appropriate connectors (Furthermore, Additionally, Moreover, etc.)
- Make sure the sentence sounds professional and natural
- The sentence should connect previous and next contexts smoothly
- Incorporate the fact naturally as part of the sentence
- Output ONLY the bridge sentence, nothing else

Bridge sentence:"""

            response = self.llm.generate(prompt, max_tokens=50, temperature=0.5)
            bridge = response.strip().replace('Bridge sentence:', '').strip() if response else ""
            
            # 빈 응답 체크 및 fallback
            if not bridge or len(bridge) < 10:
                if not bridge:
                    print(f"   ⚠️  LLM returned empty response for bridge sentence, using fallback")
                # Fallback: 간단한 triplet 문장 생성
                bridge = self._create_simple_triplet_sentence(triplet, keywords)
                if not bridge:
                    return None
            
            # Quality check
            if len(bridge) > 150:
                return None
            
            if not bridge.endswith(('.', '!', '?')):
                bridge += '.'
            
            # Preserve capitalization of proper nouns
            if h_name:
                bridge = re.sub(r'\b' + re.escape(h_name.lower()) + r'\b', h_name, bridge, flags=re.IGNORECASE)
            if t_name:
                bridge = re.sub(r'\b' + re.escape(t_name.lower()) + r'\b', t_name, bridge, flags=re.IGNORECASE)
            
            return bridge
            
        except Exception as e:
            print(f"   ⚠️  Bridge generation failed: {e}, using fallback")
            # Fallback: 간단한 triplet 문장 생성
            try:
                bridge = self._create_simple_triplet_sentence(triplet, keywords)
                return bridge
            except:
                return None
    
    def _find_best_insertion_position(self, sentences, triplet, keywords):
        """Find best position to insert sentence based on semantic similarity"""
        if not sentences:
            return 0
        
        # Try to find semantically relevant position
        triplet_text = f"{self._select_best_name(triplet[0], self.entity)} {self._select_best_name(triplet[1], self.relation)} {self._select_best_name(triplet[2], self.entity)}"
        
        try:
            triplet_embedding = self._get_sentence_embedding(triplet_text)
            if triplet_embedding is not None:
                best_pos = 0
                best_sim = -1
                
                for i in range(len(sentences)):
                    sent_embed = self._get_sentence_embedding(sentences[i])
                    if sent_embed is not None:
                        sim = self._calculate_cosine_similarity(triplet_embedding, sent_embed)
                        if sim > best_sim:
                            best_sim = sim
                            best_pos = i + 1
                
                return best_pos if best_sim > 0.2 else len(sentences) // 2
        except:
            pass
        
        # Default to middle
        return len(sentences) // 2
    
    def _create_simple_triplet_sentence(self, triplet, keywords):
        """간단한 triplet 문장 생성 (길이 제한)"""
        try:
            head, relation, tail = triplet
            
            # 이름 선택
            head_name = self._select_best_name(head, self.entity)
            relation_name = self._select_best_name(relation, self.relation)
            tail_name = self._select_best_name(tail, self.entity)
            
            if not all([head_name, relation_name, tail_name]):
                return None
            
            # 간단한 문장 패턴들
            if relation_name.lower() in ['located', 'based', 'headquartered']:
                return f"{head_name} is {relation_name} in {tail_name}."
            elif relation_name.lower() in ['founded', 'established', 'created']:
                return f"{head_name} was {relation_name} in {tail_name}."
            elif relation_name.lower() in ['developer', 'producer', 'manufacturer']:
                return f"{head_name} is a {relation_name} of {tail_name}."
            elif relation_name.lower() in ['part', 'member', 'component']:
                return f"{head_name} is part of {tail_name}."
            else:
                return f"{head_name} {relation_name} {tail_name}."
                
        except Exception as e:
            print(f"Error creating simple triplet sentence: {e}")
            return None
    
    def _find_simple_insert_position(self, sentences, triplet):
        """간단한 삽입 위치 찾기"""
        if not sentences:
            return 0
        
        # 문장 중간 위치에 삽입 (너무 앞이나 뒤는 피함)
        if len(sentences) <= 2:
            return len(sentences)  # 맨 뒤에 삽입
        else:
            # 중간 위치들 중에서 선택
            middle_positions = [len(sentences) // 2, len(sentences) // 2 + 1]
            return middle_positions[0]  # 첫 번째 중간 위치
    
    def _llm_integrate_triplets_with_context(self, sentences, triplets, keywords):
        """LLM을 사용하여 triplet들을 문맥에 맞게 지능적으로 통합"""
        try:
            # triplet들을 자연스러운 문장으로 변환
            triplet_sentences = []
            for triplet in triplets:
                sentence = self.convert_triplet_to_sentence_with_llm(triplet, keywords)
                triplet_sentences.append(sentence)
            
            # 현재 문서의 문맥 정보 수집
            context_info = self._analyze_document_context(sentences, keywords)
            
            # LLM 프롬프트 생성
            prompt = self._create_context_integration_prompt(sentences, triplet_sentences, context_info, keywords)
            
            # LLM 호출
            response = self.llm.generate(
                prompt,
                max_tokens=500,
                temperature=0.7,
                do_sample=True
            )
            
            # 응답에서 문장들 추출
            integrated_sentences = self._parse_llm_response(response, sentences)
            
            print(f"   ✅ Successfully integrated {len(triplet_sentences)} triplets using LLM")
            return integrated_sentences
            
        except Exception as e:
            print(f"   ❌ LLM integration failed: {e}")
            # 실패 시 기본 방식으로 fallback
            return self._fallback_integration(sentences, triplets, keywords)
    
    def _analyze_document_context(self, sentences, keywords):
        """문서의 문맥 정보 분석"""
        context = {
            'total_sentences': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'document_style': 'formal',
            'main_topics': keywords,
            'sentence_types': []
        }
        
        # 간단한 문장 유형 분석 (LLM이 더 정확하게 처리)
        for sentence in sentences:
            # 기본적인 문장 유형 분류
            if any(word in sentence.lower() for word in ['founded', 'established', 'created', 'developed']):
                context['sentence_types'].append('factual')
            elif any(word in sentence.lower() for word in ['is a', 'are a', 'was a', 'were a']):
                context['sentence_types'].append('descriptive')
            else:
                context['sentence_types'].append('general')
        
        return context
    
    def _create_context_integration_prompt(self, sentences, triplet_sentences, context_info, keywords):
        """문맥 통합을 위한 LLM 프롬프트 생성"""
        sentences_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
        triplets_text = "\n".join([f"- {s}" for s in triplet_sentences])
        
        prompt = f"""Integrate new factual information into this document naturally while maintaining flow and coherence.

DOCUMENT:
{sentences_text}

NEW FACTS:
{triplets_text}

TOPICS: {', '.join(keywords)}

INTEGRATION:
- Place facts near related content
- Use natural connectors ("Additionally", "Furthermore", "Moreover")
- Keep new sentences concise (max 25 words each)
- Maintain document flow and readability
- Number each sentence in output

INTEGRATED DOCUMENT:"""

        return prompt
    
    def _parse_llm_response(self, response, original_sentences):
        """LLM 응답에서 문장들을 추출"""
        try:
            # 응답에서 번호가 매겨진 문장들 추출
            lines = response.strip().split('\n')
            integrated_sentences = []
            
            for line in lines:
                line = line.strip()
                # 번호가 매겨진 문장 패턴 찾기 (예: "1. 문장내용")
                if re.match(r'^\d+\.\s+', line):
                    sentence = re.sub(r'^\d+\.\s+', '', line)
                    if sentence and len(sentence) > 5:  # 최소 길이 확인
                        integrated_sentences.append(sentence)
            
            # 추출된 문장이 너무 적으면 원본 + 새 문장으로 fallback
            if len(integrated_sentences) < len(original_sentences) * 0.8:
                print(f"   ⚠️  LLM response parsing incomplete, using fallback")
                return original_sentences
            
            return integrated_sentences
            
        except Exception as e:
            print(f"   ⚠️  Error parsing LLM response: {e}")
            return original_sentences
    
    def _fallback_integration(self, sentences, triplets, keywords):
        """LLM 실패 시 사용할 기본 통합 방식"""
        result_sentences = sentences.copy()
        
        for triplet in triplets:
            sentence = self.convert_triplet_to_sentence_with_llm(triplet, keywords)
            if sentence:
                # 간단한 유사도 기반 삽입
                best_position = self._find_best_insertion_position_with_similarity(result_sentences, sentence)
                result_sentences.insert(best_position, sentence)
        
        return result_sentences
    
    
    def _find_best_insertion_position_with_similarity(self, sentences, triplet_sentence):
        """Semantic similarity를 기반으로 최적의 삽입 위치 찾기 (개선된 버전)"""
        if not sentences:
            return 0
        
        # RoBERTa를 사용한 semantic similarity 계산
        triplet_embedding = self._get_sentence_embedding(triplet_sentence)
        if triplet_embedding is None:
            # RoBERTa 실패 시 중간 위치에 삽입
            return len(sentences) // 2
        
        best_position = 0
        best_similarity = -1.0
        
        # 각 문장과의 유사도 계산
        for i, sentence in enumerate(sentences):
            sentence_embedding = self._get_sentence_embedding(sentence)
            if sentence_embedding is None:
                continue
            
            # 코사인 유사도 계산
            similarity = self._calculate_cosine_similarity(triplet_embedding, sentence_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = i + 1  # 문장 뒤에 삽입
        
        # 유사도가 너무 낮으면 중간 위치에 삽입
        if best_similarity < 0.1:
            return len(sentences) // 2
        
        print(f"   Best similarity: {best_similarity:.3f} at position {best_position}")
        return best_position
    
    def _verify_triplet_entity_preservation(self, text: str, triplets: List[List[str]]) -> Dict:
        """
        Verify that Head and Tail entities of triplets are preserved in the text
        
        Args:
            text: Watermarked text to verify
            triplets: List of triplets to verify
        
        Returns:
            Dictionary mapping triplet tuples to verification results
        """
        verification_results = {}
        
        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for triplet in triplets:
            if len(triplet) < 3:
                continue
            
            head_id, relation_id, tail_id = triplet[0], triplet[1], triplet[2]
            triplet_tuple = tuple(triplet)
            
            # Get entity names
            head_names = self._get_entity_names_for_verification(head_id)
            tail_names = self._get_entity_names_for_verification(tail_id)
            
            # Check if both head and tail are found in the same sentence
            head_found = False
            tail_found = False
            both_found = False
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check head entity
                head_matched = False
                for head_name in head_names:
                    if head_name and isinstance(head_name, str):
                        if head_name.lower() in sentence_lower:
                            head_found = True
                            head_matched = True
                            break
                
                # Check tail entity
                tail_matched = False
                for tail_name in tail_names:
                    if tail_name and isinstance(tail_name, str):
                        if tail_name.lower() in sentence_lower:
                            tail_found = True
                            tail_matched = True
                            break
                
                # If both found in same sentence
                if head_matched and tail_matched:
                    both_found = True
                    break
            
            verification_results[triplet_tuple] = {
                "head_found": head_found,
                "tail_found": tail_found,
                "both_found": both_found,
                "head_names": head_names,
                "tail_names": tail_names
            }
        
        return verification_results
    
    def _get_entity_names_for_verification(self, entity_id: str) -> List[str]:
        """Get all entity names for verification (similar to watermark_detection)"""
        names = []
        if entity_id in self.entity:
            entity_data = self.entity[entity_id]
            if isinstance(entity_data, dict):
                names.extend(entity_data.get("entity", []))
            elif isinstance(entity_data, list):
                names.extend(entity_data)
        
        # If no names found, use entity_id as fallback
        if not names:
            names = [str(entity_id)]
        
        # Filter to only English strings
        english_names = [name for name in names if isinstance(name, str) and self._is_english_text(name)]
        return english_names if english_names else [name for name in names if isinstance(name, str)]
    
    def _retry_insert_failed_triplets(self, sentences: List[str], failed_triplets: List[List[str]], 
                                     keywords: List[str]) -> Optional[List[str]]:
        """
        Retry inserting failed triplets with more explicit entity name preservation
        
        Args:
            sentences: Current watermarked sentences
            failed_triplets: Triplets that failed verification
            keywords: Document keywords
        
        Returns:
            Updated sentences with retried triplets, or None if failed
        """
        if not failed_triplets:
            return sentences
        
        result_sentences = sentences.copy()
        
        for triplet in failed_triplets:
            if len(triplet) < 3:
                continue
            
            head_id, relation_id, tail_id = triplet[0], triplet[1], triplet[2]
            
            # Get entity names explicitly
            head_names = self._get_entity_names_for_verification(head_id)
            tail_names = self._get_entity_names_for_verification(tail_id)
            
            # Select best names (prefer shorter, more common names)
            head_name = head_names[0] if head_names else str(head_id)
            tail_name = tail_names[0] if tail_names else str(tail_id)
            relation_name = self._select_best_name(relation_id, self.relation)
            
            # Generate sentence with explicit entity names
            try:
                # More explicit prompt to ensure entity names are preserved
                prompt = f"""Create a natural sentence that explicitly includes the following entities and their relationship.

Entity 1 (Head): {head_name}
Relation: {relation_name}
Entity 2 (Tail): {tail_name}

Document context: {', '.join(keywords)}

REQUIREMENTS:
- MUST include both "{head_name}" and "{tail_name}" in the sentence
- Use the exact entity names provided (do not paraphrase or replace them)
- Create a grammatically correct sentence (15-25 words)
- Make it sound natural and professional
- Output ONLY the sentence, nothing else

Sentence:"""

                response = self.llm.generate(prompt, max_tokens=50, temperature=0.3)
                new_sentence = response.strip()
                
                # Verify entity names are in the generated sentence
                new_sentence_lower = new_sentence.lower()
                head_in_sentence = head_name.lower() in new_sentence_lower
                tail_in_sentence = tail_name.lower() in new_sentence_lower
                
                if head_in_sentence and tail_in_sentence:
                    # Quality check
                    if len(new_sentence) < 10 or len(new_sentence) > 200:
                        continue
                    
                    if not new_sentence.endswith(('.', '!', '?')):
                        new_sentence += '.'
                    
                    # Insert at appropriate position
                    position = self._find_best_insertion_position(result_sentences, triplet, keywords)
                    result_sentences.insert(position, new_sentence)
                    print(f"      ✅ Retried triplet: {head_name} ... {tail_name}")
                else:
                    print(f"      ⚠️  Retry failed: entities not preserved in generated sentence")
                    
            except Exception as e:
                print(f"      ⚠️  Retry insertion error: {e}")
                continue
        
        return result_sentences if len(result_sentences) > len(sentences) else None
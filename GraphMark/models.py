import os
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import json
import re
import spacy
import nltk
import time
import random
from collections import Counter
from torch.nn.functional import cosine_similarity
from openai import OpenAI
from together import Together
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from scipy.stats import kendalltau
import numpy as np
import pickle
import tiktoken
import pdb
from functools import lru_cache
from subgraph_construction import subgraph_construction
from transformers import RobertaModel, RobertaTokenizer

torch.manual_seed(42)
random.seed(42)

class LLM():
    def __init__(self, model, device_id=None):
        if "gpt" in model:
            self.model = ChatGPT(model)
        elif model == "llama-3-8b":
            self.model = Llama3_8B(device_id=device_id)
        elif model == "llama-3-8b-inst":
            self.model = Llama3_8B_Chat(device_id=device_id)
        elif model == "mistral-7b-inst":
            self.model = Mistral_7B_Inst(device_id=device_id)
        elif model == "Qwen2_5-7b_inst":
            self.model = Qwen2_5_7B_Inst(device_id=device_id)
    
    def generate(self, prompt, max_tokens=600, temperature=1.0):
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=float(temperature))

class ChatGPT():
    def __init__(self, llm):
        # config.py에서 API 키 읽기
        try:
            import sys
            sys.path.append('/home/wooseok/KG_Mark')
            from config import OPENAI_API_KEY
            print("OpenAI API key loaded from config.py")
        except ImportError:
            print("Warning: config.py not found. Please set OPENAI_API_KEY environment variable.")
            import os
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in config.py or environment variables")
        
        self.llm = llm
        print(f"Loading {llm}...")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def obtain_response(self, prompt, max_tokens, temperature, seed=42):
        response = None
        num_attemps = 0
        messages = []
        messages.append({"role": "user", "content": prompt})
        while response is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed)
            except Exception as e:
                if num_attemps == 5:
                    print(f"Attempt {num_attemps} failed, breaking...")
                    return None
                print(e)
                num_attemps += 1
                print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response.choices[0].message.content.strip()
    
    def generate(self, prompt, max_tokens, temperature):
        return self.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature)

class Llama3_8B():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return output_text

class Llama3_8B_Chat():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        # max_tokens를 max_new_tokens로 변환
        if max_tokens != 512:  # 기본값이 아닌 경우
            max_new_tokens = max_tokens
        else:
            max_new_tokens = max_tokens
            
        generation_args = {
            "eos_token_id": eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text


class Mistral_7B_Inst():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text

class Qwen2_5_7B_Inst():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text

    
class KEPLEREmbedding():
    def __init__(self, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # KEPLER 사전학습 모델 로드 (Wikipedia + Wikidata5M 기반)
        self.MODEL_NAME = "thunlp/KEPLER"
        print(f"Loading KEPLER model: {self.MODEL_NAME}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()  # 추론 모드
        self.model = self.model.to(self.device)
        
        print(f"KEPLER model loaded on device: {self.device}")
    
    def get_entity_embedding(self, entity_name):
        """
        Entity 이름을 통해 KEPLER embedding 생성
        Args:
            entity_name (str): Entity 이름 (예: "Neil Armstrong")
        Returns:
            numpy.ndarray: 768차원 embedding 벡터
        """
        try:
            # 입력: "Neil Armstrong" 같은 Entity 이름
            inputs = self.tokenizer(entity_name, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # [CLS] 토큰 임베딩 사용 (768차원)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{entity_name}': {e}")
            # 에러 시 zero embedding 반환
            return np.zeros(768)
    
    def get_entity_embeddings_batch(self, entity_names, batch_size=8):
        """
        여러 entity 이름에 대해 배치로 embedding 생성
        Args:
            entity_names (list): Entity 이름 리스트
            batch_size (int): 배치 크기
        Returns:
            dict: {entity_name: embedding} 형태의 딕셔너리
        """
        embeddings = {}
        
        for i in range(0, len(entity_names), batch_size):
            batch_names = entity_names[i:i+batch_size]
            try:
                # 배치로 토크나이징
                inputs = self.tokenizer(batch_names, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 각 entity의 [CLS] 토큰 embedding 추출
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                for j, entity_name in enumerate(batch_names):
                    embeddings[entity_name] = batch_embeddings[j]
                    
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # 에러 시 개별 처리
                for entity_name in batch_names:
                    embeddings[entity_name] = self.get_entity_embedding(entity_name)
        
        return embeddings
    
    def compute_similarity(self, emb1, emb2):
        """두 embedding 간의 코사인 유사도 계산"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def find_similar_entities(self, query_embedding, entity_embeddings, top_k=5):
        """Query embedding과 가장 유사한 entity들 찾기"""
        similarities = []
        for entity_name, entity_emb in entity_embeddings.items():
            sim = self.compute_similarity(query_embedding, entity_emb)
            similarities.append((entity_name, sim))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class KGWatermarker():
    def __init__(self, llm, ratio, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LLM 인스턴스 생성
        self.llm = LLM(llm, device_id=device_id)
        self.ratio = ratio  # triplet 활용 비율
        self.nlp = spacy.load("en_core_web_sm")
        
        # KG 데이터 로드
        kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
        kg_entity_path = f"{kg_root_path}/entities.txt"
        kg_relation_path = f"{kg_root_path}/relations.txt"
        kg_triple_path = f"{kg_root_path}/triplets.txt"
        
        # subgraph_construction 초기화 (LLM 인스턴스 전달)
        self.constructor = subgraph_construction(
            self.llm, ratio=ratio, 
            kg_entity_path=kg_entity_path, 
            kg_relation_path=kg_relation_path, 
            kg_triple_path=kg_triple_path, 
            device_id=device_id
        )
        
        self.entity, self.relation, self.triple = self.constructor.load_kg(
            kg_entity_path, kg_relation_path, kg_triple_path
        )
    
    def build_subgraph_from_text(self, text, enable_adaptive_pruning=True, pruning_ratio=0.3):
        return self.constructor.build_subgraph_from_text(text, enable_adaptive_pruning, pruning_ratio)
    
    def convert_triple_to_sentence(self, triple):
        """Triple을 자연스러운 문장으로 변환"""
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            return str(triple)
        
        head, relation, tail = triple
        head_name = self.entity.get(head, {}).get("entity", [head])[0] if head in self.entity else head
        tail_name = self.entity.get(tail, {}).get("entity", [tail])[0] if tail in self.entity else tail
        relation_name = self.relation.get(relation, {}).get("name", [relation])[0] if relation in self.relation else relation
        
        patterns = [
            f"{head_name} {relation_name} {tail_name}.",
            f"{head_name} is known for {relation_name} {tail_name}.",
            f"The {relation_name} of {head_name} is {tail_name}.",
            f"{head_name} has {relation_name} {tail_name}."
        ]
        
        return random.choice(patterns)
    
    def select_triplets_for_watermarking(self, subgraph_triples, keywords):
        """Ratio 기반으로 watermarking에 사용할 triplets 선택"""
        total_triplets = len(subgraph_triples)
        if total_triplets == 0:
            return [], 0, []
        
        target_count = max(1, int(total_triplets * self.ratio))
        
        # 간단한 선택: 상위 target_count개 선택
        if isinstance(subgraph_triples, dict):
            selected_keys = list(subgraph_triples.keys())[:target_count]
            selected_triplets = [subgraph_triples[k] for k in selected_keys]
            return selected_triplets, total_triplets, selected_keys
        else:
            selected_triplets = subgraph_triples[:target_count]
            selected_indices = list(range(target_count))
            return selected_triplets, total_triplets, selected_indices
    
    def _is_english_text(self, text):
        """텍스트가 영어인지 판별"""
        if not text or not isinstance(text, str):
            return False
        
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        total_chars = sum(1 for char in text if char.isalpha())
        return total_chars > 0 and english_chars / total_chars >= 0.8
    

    
    def _find_related_entity_relation_for_keyword(self, keyword, selected_triplets):
        """키워드와 관련된 triplet에서 추가 정보(Entity/Relation) 찾기"""
        for triple in selected_triplets:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
                
            head, relation, tail = triple
            head_name = self.entity.get(head, {}).get("entity", [head])[0] if head in self.entity else head
            tail_name = self.entity.get(tail, {}).get("entity", [tail])[0] if tail in self.entity else tail
            relation_name = self.relation.get(relation, {}).get("name", [relation])[0] if relation in self.relation else relation
            
            # 키워드가 head, tail, relation 중 어디에 있는지 확인
            keyword_lower = keyword.lower()
            
            if keyword_lower in head_name.lower():
                # 키워드가 head에 있으면 tail과 relation 정보 반환
                if self._is_english_text(tail_name) and self._is_english_text(relation_name):
                    return {"type": "head_match", "relation": relation_name, "tail": tail_name}
                    
            elif keyword_lower in tail_name.lower():
                # 키워드가 tail에 있으면 head와 relation 정보 반환
                if self._is_english_text(head_name) and self._is_english_text(relation_name):
                    return {"type": "tail_match", "head": head_name, "relation": relation_name}
                    
            elif keyword_lower in relation_name.lower():
                # 키워드가 relation에 있으면 head와 tail 정보 반환
                if self._is_english_text(head_name) and self._is_english_text(tail_name):
                    return {"type": "relation_match", "head": head_name, "tail": tail_name}
        
        return None
    
    def _insert_entity_relation_with_pos_check(self, keyword_token, additional_info, doc, keyword_index):
        """POS Tagging을 확인하여 Entity/Relation을 적절한 자리에 삽입"""
        if not additional_info:
            return None
        
        # 키워드 토큰의 POS tag 확인
        keyword_pos = keyword_token.pos_
        
        # 추가 정보를 자연스럽게 삽입
        if additional_info["type"] == "head_match":
            # 키워드가 head에 있는 경우: relation + tail을 삽입
            relation = additional_info["relation"]
            tail = additional_info["tail"]
            
            # 키워드가 명사(NN)인 경우: "which is a [relation] of [tail]" 형태로 삽입
            if keyword_pos in ["NOUN", "PROPN"]:
                return f"which is a {relation} of {tail}"
            # 키워드가 다른 POS인 경우: "with {relation} {tail}" 형태로 삽입
            else:
                return f"with {relation} {tail}"
                
        elif additional_info["type"] == "tail_match":
            # 키워드가 tail에 있는 경우: head + relation을 삽입
            head = additional_info["head"]
            relation = additional_info["relation"]
            
            # 키워드가 명사(NN)인 경우: "where {head} has {relation}" 형태로 삽입
            if keyword_pos in ["NOUN", "PROPN"]:
                return f"where {head} has {relation}"
            # 키워드가 다른 POS인 경우: "from {head} with {relation}" 형태로 삽입
            else:
                return f"from {head} with {relation}"
                
        elif additional_info["type"] == "relation_match":
            # 키워드가 relation에 있는 경우: head + tail을 삽입
            head = additional_info["head"]
            tail = additional_info["tail"]
            
            # 키워드가 동사(VERB)인 경우: "involving {head} and {tail}" 형태로 삽입
            if keyword_pos == "VERB":
                return f"involving {head} and {tail}"
            # 키워드가 다른 POS인 경우: "between {head} and {tail}" 형태로 삽입
            else:
                return f"between {head} and {tail}"
        
        return None
    

    
    def watermark_sentence(self, sentence, keywords, selected_triplets, should_modify=False):
        """단일 문장에 워터마킹 적용 - Keyword는 유지하고 관련 Entity/Relation을 적절한 자리에 삽입"""
        if not should_modify:
            return sentence, []
        
        # 키워드가 포함된 문장인지 확인
        if not any(k.lower() in sentence.lower() for k in keywords):
            return sentence, []
        
        # Keyword는 그대로 두고, 관련 Entity/Relation을 문장에 삽입
        doc = self.nlp(sentence)
        modified_tokens = []
        modified = False
        
        # 키워드가 포함된 토큰들을 찾아서 해당 위치에 관련 정보 삽입
        for i, token in enumerate(doc):
            if any(k.lower() in token.text.lower() for k in keywords):
                # 키워드는 그대로 유지
                modified_tokens.append(token.text)
                
                # 해당 키워드와 관련된 triplet에서 추가 정보 찾기
                additional_info = self._find_related_entity_relation_for_keyword(token.text, selected_triplets)
                if additional_info:
                    # POS Tagging을 확인하여 적절한 자리에 삽입
                    insertion_text = self._insert_entity_relation_with_pos_check(token, additional_info, doc, i)
                    if insertion_text:
                        modified_tokens.append(insertion_text)
                        modified = True
            else:
                modified_tokens.append(token.text)
        
        modified_sentence = " ".join(modified_tokens)
        
        # 추가 정보 삽입이 실패한 경우 빈 리스트 반환 (문장 삽입 제거)
        inserted_sentences = []
        
        return modified_sentence, inserted_sentences
    
    def insert_watermark(self, prefix, target, enable_adaptive_pruning=True, pruning_ratio=0.3):
        """워터마크 삽입 메인 함수"""
        combined_text = f"{prefix} {target}"
        
        # Subgraph 구성
        subgraph_info = self.build_subgraph_from_text(combined_text, enable_adaptive_pruning, pruning_ratio)
        
        # Triplets 선택
        selected_triplets, total_triplets, _ = self.select_triplets_for_watermarking(
            subgraph_info['subgraph_triples'], subgraph_info['keywords']
        )
        
        # 문장별 워터마킹
        doc = self.nlp(combined_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # 키워드가 포함된 문장들 찾기
        keyword_sentences = [s for s in sentences if any(k.lower() in s.lower() for k in subgraph_info['keywords'])]
        
        # selected_triplets 개수만큼만 수정
        num_to_modify = min(len(selected_triplets), len(keyword_sentences))
        sentences_to_modify = random.sample(keyword_sentences, num_to_modify) if num_to_modify > 0 else []
        
        modified_sentences = []
        inserted_sentences = []
        
        for sentence in sentences:
            should_modify = sentence in sentences_to_modify
            modified_sentence, new_sentences = self.watermark_sentence(
                sentence, subgraph_info['keywords'], selected_triplets, should_modify
            )
            modified_sentences.append(modified_sentence)
            inserted_sentences.extend(new_sentences)
        
        # 결과 조합
        watermarked_text = " ".join(modified_sentences + inserted_sentences)
        
        return {
            "original_text": prefix + " " + target,
            "watermarked_text": watermarked_text,
            "keywords": subgraph_info['keywords'],
            "ratio": self.ratio,
            "total_triplets": total_triplets,
            "used_triplets": len(selected_triplets),
            "triplet_usage_ratio": len(selected_triplets) / total_triplets if total_triplets > 0 else 0,
            "modified_sentences": sum(1 for i, s in enumerate(modified_sentences) if s != sentences[i]),
            "inserted_sentences": len(inserted_sentences),
            "pruning_ratio": pruning_ratio,
            "subgraph_triples": subgraph_info['subgraph_triples'],
            "selected_triplets": selected_triplets
        }
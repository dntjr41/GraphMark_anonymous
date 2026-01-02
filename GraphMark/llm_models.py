"""
LLM 모델 관련 클래스들
- LLM: 기본 LLM 래퍼 클래스
- ChatGPT: OpenAI ChatGPT API 클래스
- Llama3_8B: Llama 3 8B 모델 클래스
- Mistral7B: Mistral 7B 모델 클래스
- Qwen2_5_7B: Qwen 2.5 7B 모델 클래스
"""

import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import os
import numpy as np
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import OPENAI_API_KEY as CONFIG_API_KEY
except ImportError:
    CONFIG_API_KEY = None


class LLM:
    """LLM 팩토리 클래스 - 모델 이름에 따라 적절한 LLM 클래스 인스턴스 생성"""
    def __init__(self, model, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = model
        
        # 모델 이름에 따라 적절한 클래스 인스턴스 생성
        if "gpt-3.5-turbo" in model.lower():
            self.model = ChatGPT(model)
        elif model == "gpt-4":
            self.model = ChatGPT4(model)
        elif model == "llama-3-8b":
            self.model = Llama3_8B()
        elif model == "llama-3-8b-inst":
            self.model = Llama3_8B_Chat()
        elif model == "mistral-7b-inst":
            self.model = Mistral7B()
        elif model == "Qwen2_5-7b_inst":
            self.model = Qwen2_5_7B()
        else:
            # 기본값으로 Llama3_8B 사용
            self.model = Llama3_8B()
    
    def generate(self, prompt, max_tokens=600, temperature=1.0, **kwargs):
        """텍스트 생성 - 내부 모델의 generate 메서드 호출"""
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)


class ChatGPT:
    """OpenAI ChatGPT API 클래스"""
    def __init__(self, llm):
        self.model_name = "gpt-3.5-turbo"
        # Try to get API key in order: 1) environment variable, 2) config.py, 3) key file
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # If not found, try config.py
        if not self.api_key and CONFIG_API_KEY:
            self.api_key = CONFIG_API_KEY
            print(f"✅ Loaded OpenAI API key from config.py")
        
        # If still not found, try to read from key file
        if not self.api_key:
            key_file_paths = [
                "/home/wooseok/KG_Mark/openai_key_wooseok.txt",
                "/home/wooseok/KG_Mark/openai_key.txt",
                "openai_key_wooseok.txt",
                "openai_key.txt"
            ]
            for key_file_path in key_file_paths:
                if os.path.exists(key_file_path):
                    try:
                        with open(key_file_path, 'r') as f:
                            self.api_key = f.read().strip()
                        print(f"✅ Loaded OpenAI API key from: {key_file_path}")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to read key file {key_file_path}: {e}")
                        continue
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable, config.py, or key file is required")
        
        openai.api_key = self.api_key
    
    def obtain_response(self, prompt, max_tokens, temperature, seed=42):
        """OpenAI API를 통한 응답 생성"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in ChatGPT API call: {e}")
            return "Error in generation"
    
    def generate(self, prompt, max_tokens, temperature):
        """텍스트 생성"""
        return self.obtain_response(prompt, max_tokens, temperature)

class ChatGPT4:
    """OpenAI ChatGPT 4 API 클래스"""
    def __init__(self, llm):
        self.model_name = "gpt-4"
        self.actual_model_name = None  # Will be set after first API call
        # Try to get API key in order: 1) environment variable, 2) config.py, 3) key file
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # If not found, try config.py
        if not self.api_key and CONFIG_API_KEY:
            self.api_key = CONFIG_API_KEY
            print(f"✅ Loaded OpenAI API key from config.py")
        
        # If still not found, try to read from key file
        if not self.api_key:
            key_file_paths = [
                "/home/wooseok/KG_Mark/openai_key_wooseok.txt",
                "/home/wooseok/KG_Mark/openai_key.txt",
                "openai_key_wooseok.txt",
                "openai_key.txt"
            ]
            for key_file_path in key_file_paths:
                if os.path.exists(key_file_path):
                    try:
                        with open(key_file_path, 'r') as f:
                            self.api_key = f.read().strip()
                        print(f"✅ Loaded OpenAI API key from: {key_file_path}")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to read key file {key_file_path}: {e}")
                        continue
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable, config.py, or key file is required")
        
        openai.api_key = self.api_key
        print(f"🔍 GPT-4 Model Checkpoint (requested): {self.model_name}")
        
        # Test API call to get actual model version
        try:
            test_response = openai.ChatCompletion.create(
                model="chatgpt-4o-latest",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            if hasattr(test_response, 'model'):
                self.actual_model_name = test_response.model
                print(f"🔍 GPT-4 Model Checkpoint (actual): {self.actual_model_name}")
        except Exception as e:
            print(f"⚠️  Could not verify GPT-4 model version: {e}")
            print(f"   Will check on first actual API call")
    
    def obtain_response(self, prompt, max_tokens, temperature, seed=42):
        """OpenAI API를 통한 응답 생성"""
        try:
            model_to_use = "chatgpt-4o-latest"
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            # 응답에서 실제 사용된 모델 정보 확인 및 저장
            if hasattr(response, 'model'):
                actual_model = response.model
                if self.actual_model_name is None:
                    # 첫 번째 호출 시에만 출력
                    self.actual_model_name = actual_model
                    print(f"🔍 GPT-4 Model Checkpoint (actual): {actual_model}")
                elif self.actual_model_name != actual_model:
                    # 모델이 변경된 경우에만 출력
                    self.actual_model_name = actual_model
                    print(f"🔍 GPT-4 Model Checkpoint (actual): {actual_model}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in ChatGPT 4 API call: {e}")
            return "Error in generation"
    
    def generate(self, prompt, max_tokens, temperature):
        """텍스트 생성"""
        return self.obtain_response(prompt, max_tokens, temperature)

class Llama3_8B:
    """Llama 3 8B 모델 클래스"""
    def __init__(self, half=False):
        self.model_name = "meta-llama/Llama-3-8B"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        print("Loading Llama 3 8B model...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            torch_dtype=torch.float16 if half else torch.float32,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self,
                 prompt,
                 max_tokens=600,
                 temperature=0.7,
                 top_p=0.9,
                 top_k=40,
                 repetition_penalty=1.2,
                 do_sample=True,
                 **kwargs):
        """Llama 3 8B를 사용한 텍스트 생성"""
        try:
            # 입력 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 파라미터 설정
            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # 안전한 파라미터만 사용
            safe_generation_args = {k: v for k, v in generation_args.items() 
                                  if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k', 
                                          'repetition_penalty', 'do_sample', 'pad_token_id', 
                                          'eos_token_id', 'use_cache']}
            
            # LogitsProcessor 추가 (repetition_penalty가 1.0이 아닌 경우)
            logits_processor = None
            if repetition_penalty != 1.0:
                from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
                logits_processor = LogitsProcessorList([
                    RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                ])
            
            if logits_processor:
                generation_args["logits_processor"] = logits_processor
                
            print(f"   DEBUG: generation_args = {generation_args}")
            
            # transformers generate 메서드 사용 (generation_config 수정)
            try:
                with torch.inference_mode():
                    # generation_config를 수정해서 안전한 파라미터 설정
                    original_config = self.model.generation_config
                    self.model.generation_config.do_sample = False
                    self.model.generation_config.num_return_sequences = 1
                    self.model.generation_config.num_beams = 1
                    self.model.generation_config.temperature = 1.0
                    self.model.generation_config.top_p = 1.0
                    self.model.generation_config.top_k = 50
                    self.model.generation_config.repetition_penalty = 1.0
                    self.model.generation_config.length_penalty = 1.0
                    self.model.generation_config.early_stopping = False
                    self.model.generation_config.max_length = None
                    self.model.generation_config.min_length = None
                    self.model.generation_config.bad_words_ids = None
                    self.model.generation_config.force_words_ids = None
                    self.model.generation_config.suppress_tokens = None
                    self.model.generation_config.begin_suppress_tokens = None
                    self.model.generation_config.forced_eos_token_id = None
                    self.model.generation_config.exponential_decay_length_penalty = None
                    self.model.generation_config.renormalize_logits = False
                    self.model.generation_config.constraints = None
                    self.model.generation_config.output_attentions = False
                    self.model.generation_config.output_hidden_states = False
                    self.model.generation_config.output_scores = False
                    self.model.generation_config.return_dict_in_generate = False
                    self.model.generation_config.generation_kwargs = {}
                    
                    # 안전한 generation args만 사용
                    safe_generation_args = {
                        'max_new_tokens': generation_args.get('max_new_tokens'),
                        'temperature': generation_args.get('temperature'),
                        'top_p': generation_args.get('top_p'),
                        'top_k': generation_args.get('top_k'),
                        'repetition_penalty': generation_args.get('repetition_penalty'),
                        'do_sample': generation_args.get('do_sample'),
                        'pad_token_id': generation_args.get('pad_token_id'),
                        'eos_token_id': generation_args.get('eos_token_id')
                    }
                    
                    print(f"   DEBUG: Using transformers generate with safe args: {safe_generation_args}")
                    
                    outputs = self.model.generate(**inputs, **safe_generation_args)
                    output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                    
                    print(f"   DEBUG: Generated text: {output_text[:100]}...")
                    return output_text
                    
            except Exception as e:
                print(f"   DEBUG: Transformers generate failed: {e}")
                import traceback
                print(f"   DEBUG: Traceback: {traceback.format_exc()}")
                return "Error in generation"
                
        except Exception as e:
            print(f"Error in Llama3_8B generation: {e}")
            return "Error in generation"

class Llama3_8B_Chat():
    def __init__(self, half=False):
        # CUDA_VISIBLE_DEVICES 환경변수를 고려한 device 설정
        if torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES가 설정된 경우, 항상 cuda:0을 사용 (가시적인 첫 번째 GPU)
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        if half:
            print("Loading half precision model...")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
            self.model = self.model.to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            self.model = self.model.to(self.device)
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        typical_p=None,
        temperature=1,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        # Llama3 Instruct 모델에 최적화된 generation 파라미터
        generation_args = {
            "eos_token_id": eos_token_id,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        }
        
        # num_return_sequences가 1보다 클 때만 추가
        if num_return_sequences > 1:
            generation_args["num_return_sequences"] = num_return_sequences
        
        # do_sample=True일 때만 sampling 파라미터들 추가
        if do_sample:
            generation_args.update({
                "temperature": temperature,
            })
            
            # top_k와 top_p는 일시적으로 비활성화 (에러 방지)
            # if top_k > 0:
            #     generation_args["top_k"] = top_k
            # if top_p > 0:
            #     generation_args["top_p"] = top_p
        
        # typical_p는 선택적으로만 추가
        if typical_p is not None and do_sample:
            generation_args["typical_p"] = typical_p
            
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
            
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text

class Mistral7B:
    """Mistral 7B 모델 클래스"""
    def __init__(self, half=False):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        print("Loading Mistral 7B model...")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16 if half else torch.float32,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self,
                 prompt,
                 max_tokens=600,
                 temperature=1.0,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1,
                 do_sample=True,
                 **kwargs):
        """Mistral 7B를 사용한 텍스트 생성"""
        try:
            # 입력 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 파라미터 설정
            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # 안전한 파라미터만 사용
            safe_generation_args = {k: v for k, v in generation_args.items() 
                                  if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k', 
                                          'repetition_penalty', 'do_sample', 'pad_token_id', 
                                          'eos_token_id', 'use_cache']}
            
            # LogitsProcessor 추가 (repetition_penalty가 1.0이 아닌 경우)
            logits_processor = None
            if repetition_penalty != 1.0:
                from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
                logits_processor = LogitsProcessorList([
                    RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                ])
            
            if logits_processor:
                generation_args["logits_processor"] = logits_processor
                
            print(f"   DEBUG: generation_args = {generation_args}")
            
            # transformers generate 메서드 사용
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **safe_generation_args)
                    output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                    
                    print(f"   DEBUG: Generated text: {output_text[:100]}...")
                    return output_text
                    
            except Exception as e:
                print(f"   DEBUG: Transformers generate failed: {e}")
                import traceback
                print(f"   DEBUG: Traceback: {traceback.format_exc()}")
                return "Error in generation"
                
        except Exception as e:
            print(f"Error in Mistral7B generation: {e}")
            return "Error in generation"


class Qwen2_5_7B:
    """Qwen 2.5 7B 모델 클래스"""
    def __init__(self, half=False):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        print("Loading Qwen 2.5 7B model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float16 if half else torch.float32,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self,
                 prompt,
                 max_tokens=600,
                 temperature=1.0,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1,
                 do_sample=True,
                 **kwargs):
        """Qwen 2.5 7B Instruct를 사용한 텍스트 생성 (chat template 사용)"""
        try:
            # Qwen Instruct 모델은 chat template 사용
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            # Chat template 적용
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # 생성 파라미터 설정
            generation_args = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # LogitsProcessor 추가 (repetition_penalty가 1.0이 아닌 경우)
            if repetition_penalty != 1.0:
                from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
                logits_processor = LogitsProcessorList([
                    RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                ])
                generation_args["logits_processor"] = logits_processor
            
            # transformers generate 메서드 사용
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(inputs, **generation_args)
                    # 입력 길이 이후의 부분만 디코딩
                    input_length = inputs.shape[-1]
                    output_text = self.tokenizer.decode(
                        outputs[0][input_length:], 
                        skip_special_tokens=True
                    )
                    
                    # 빈 응답 체크
                    if not output_text or len(output_text.strip()) < 3:
                        print(f"   ⚠️  Qwen generated empty or very short response, returning fallback")
                        return ""
                    
                    return output_text.strip()
                    
            except Exception as e:
                print(f"   ⚠️  Qwen generation failed: {e}")
                import traceback
                traceback.print_exc()
                return ""
                
        except Exception as e:
            print(f"Error in Qwen2_5_7B generation: {e}")
            import traceback
            traceback.print_exc()
            return ""


class KEPLEREmbedding:
    """KEPLER 임베딩 모델 클래스"""
    def __init__(self, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.MODEL_NAME = "thunlp/KEPLER"
        print(f"Loading KEPLER model: {self.MODEL_NAME}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"KEPLER model loaded on device: {self.device}")
    
    def get_entity_embedding(self, entity_name):
        try:
            inputs = self.tokenizer(entity_name, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{entity_name}': {e}")
            return np.zeros(768)

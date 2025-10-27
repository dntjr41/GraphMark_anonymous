import json
import math
import torch
import openai
import re
import statistics
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk import ngrams

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 1. JSONL 파일 읽기 함수
def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 2. Perplexity 계산 함수 (LLaMA3 8B Instruct 사용)
def calculate_perplexity(text, model, tokenizer):
    """주어진 텍스트의 Perplexity를 계산"""
    if not text:  # 텍스트가 비어 있는 경우
        return float('inf')  # 무한대로 설정
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    return perplexity

# 3. Log Diversity 계산 함수
def calculate_log_diversity(text, n=2):
    """주어진 텍스트의 Log Diversity를 계산 (n-gram 기반)"""
    if not text:  # 텍스트가 비어 있는 경우
        return 0
    tokens = text.split()
    n_grams = list(ngrams(tokens, n))
    unique_n_grams = set(n_grams)
    if len(unique_n_grams) == 0:
        return 0
    return math.log(len(unique_n_grams))

# 4. GPT를 사용한 텍스트 품질 평가 함수
def evaluate_text_with_gpt(original_text, watermarked_text, metric, model_name="gpt-4-turbo"):
    """
    Compare original_text and watermarked_text using the given metric ('relevance', 'coherence', 'interestingness').
    Returns (original_score, watermarked_score).
    """
    if metric == "relevance":
        prompt = (
            f"Compare the following two texts and evaluate their relevance to the intended meaning on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "coherence":
        prompt = (
            f"Compare the coherence of the following two texts on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "interestingness":
        prompt = (
            f"Compare how interesting the following two texts are on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    else:
        raise ValueError("Invalid metric provided.")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert text evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )
        result_text = response.choices[0].message.content
        if result_text:
            result_text = result_text.strip()
            try:
                match = re.search(r"\[Original:\s*(\d+),\s*Watermarked:\s*(\d+)\]", result_text)
                if match:
                    orig_score = int(match.group(1))
                    water_score = int(match.group(2))
                    return orig_score, water_score
            except Exception as e:
                print(f"Score extraction failed: {e}")
    except Exception as e:
        print(f"GPT API call failed for {metric}: {e}")
    
    return 0, 0

# 5. 통합 메인 함수
def main():
    # 파일 경로 설정
    file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_GraphMark_15.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/test/test.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_blackbox.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_exp.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_expedit.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_kgw.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_postmark-12.jsonl"
    # file_path = "/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b-inst_unigram.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    total_items = len(data)
    print(f"총 {total_items}개 항목을 처리합니다.")
    
    # LLaMA3 8B Instruct 모델과 토크나이저 로드
    print("LLaMA3 8B Instruct 모델을 로드 중...")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # GPU/CPU 자동 매핑
            torch_dtype=torch.float16  # 메모리 효율성을 위해 float16 사용
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # LLaMA3 모델의 경우 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("LLaMA3 8B Instruct 모델 로드 완료!")
        
    except Exception as e:
        print(f"Error loading LLaMA3 8B Instruct model: {e}")
        print("Please check the model path and ensure all required files are present.")
        return
    
    # 각 텍스트 유형별 메트릭 저장용 리스트
    perplexities_original = []
    perplexities_watermarked = []
    log_diversities_original = []
    log_diversities_watermarked = []
    
    # GPT 품질 평가 결과 저장용 딕셔너리
    relevance_result = {"original_better": 0, "watermarked_better": 0, "tie": 0}
    coherence_result = {"original_better": 0, "watermarked_better": 0, "tie": 0}
    interestingness_result = {"original_better": 0, "watermarked_better": 0, "tie": 0}
    
    # 각 항목 처리
    for idx, item in enumerate(data):
        print(f"\n항목 {idx+1}/{total_items} 처리 중...")
        
        original_text = item.get("original_text", "")
        watermarked_text = item.get("watermarked_text", "")
        
        # 1. Perplexity 계산 (LLaMA3 8B Instruct 사용)
        print("  - Perplexity 계산 중...")
        perplexity_original = calculate_perplexity(original_text, model, tokenizer)
        perplexity_watermarked = calculate_perplexity(watermarked_text, model, tokenizer)
        
        # 2. Log Diversity 계산
        print("  - Log Diversity 계산 중...")
        log_diversity_original = calculate_log_diversity(original_text)
        log_diversity_watermarked = calculate_log_diversity(watermarked_text)
        
        # 3. GPT 품질 평가 (GPT-4 Turbo 사용)
        print("  - GPT 품질 평가 중...")
        relevance_original, relevance_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "relevance")
        coherence_original, coherence_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "coherence")
        interestingness_original, interestingness_watermarked = evaluate_text_with_gpt(original_text, watermarked_text, "interestingness")
        
        # 결과 저장
        perplexities_original.append(perplexity_original)
        perplexities_watermarked.append(perplexity_watermarked)
        log_diversities_original.append(log_diversity_original)
        log_diversities_watermarked.append(log_diversity_watermarked)
        
        # GPT 품질 평가 결과 집계
        if relevance_original > relevance_watermarked:
            relevance_result["original_better"] += 1
        elif relevance_original < relevance_watermarked:
            relevance_result["watermarked_better"] += 1
        else:
            relevance_result["tie"] += 1

        if coherence_original > coherence_watermarked:
            coherence_result["original_better"] += 1
        elif coherence_original < coherence_watermarked:
            coherence_result["watermarked_better"] += 1
        else:
            coherence_result["tie"] += 1

        if interestingness_original > interestingness_watermarked:
            interestingness_result["original_better"] += 1
        elif interestingness_original < interestingness_watermarked:
            interestingness_result["watermarked_better"] += 1
        else:
            interestingness_result["tie"] += 1
        
        # 현재 항목 결과 출력
        print(f"    - Perplexity: Original={perplexity_original:.2f}, Watermarked={perplexity_watermarked:.2f}")
        print(f"    - Log Diversity: Original={log_diversity_original:.2f}, Watermarked={log_diversity_watermarked:.2f}")
        print(f"    - Relevance: Original={relevance_original}, Watermarked={relevance_watermarked}")
        print(f"    - Coherence: Original={coherence_original}, Watermarked={coherence_watermarked}")
        print(f"    - Interestingness: Original={interestingness_original}, Watermarked={interestingness_watermarked}")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
    # 1. Perplexity 결과
    print("\n1. PERPLEXITY ANALYSIS (LLaMA3 8B Instruct)")
    print("-" * 50)
    avg_perplexity_original = statistics.mean(perplexities_original)
    avg_perplexity_watermarked = statistics.mean(perplexities_watermarked)
    print(f"Original Text Average Perplexity: {avg_perplexity_original:.2f}")
    print(f"Watermarked Text Average Perplexity: {avg_perplexity_watermarked:.2f}")
    print(f"Perplexity Difference: {avg_perplexity_watermarked - avg_perplexity_original:.2f}")
    
    # 2. Log Diversity 결과
    print("\n2. LOG DIVERSITY ANALYSIS")
    print("-" * 50)
    avg_log_diversity_original = statistics.mean(log_diversities_original)
    avg_log_diversity_watermarked = statistics.mean(log_diversities_watermarked)
    print(f"Original Text Average Log Diversity: {avg_log_diversity_original:.2f}")
    print(f"Watermarked Text Average Log Diversity: {avg_log_diversity_watermarked:.2f}")
    print(f"Log Diversity Difference: {avg_log_diversity_watermarked - avg_log_diversity_original:.2f}")
    
    # 3. GPT 품질 평가 결과
    print("\n3. GPT QUALITY EVALUATION (GPT-4 Turbo)")
    print("-" * 50)
    print("Relevance:")
    print(f"  Original Better: {relevance_result['original_better'] / total_items * 100:.2f}%")
    print(f"  Watermarked Better: {relevance_result['watermarked_better'] / total_items * 100:.2f}%")
    print(f"  Tie: {relevance_result['tie'] / total_items * 100:.2f}%")

    print("\nCoherence:")
    print(f"  Original Better: {coherence_result['original_better'] / total_items * 100:.2f}%")
    print(f"  Watermarked Better: {coherence_result['watermarked_better'] / total_items * 100:.2f}%")
    print(f"  Tie: {coherence_result['tie'] / total_items * 100:.2f}%")

    print("\nInterestingness:")
    print(f"  Original Better: {interestingness_result['original_better'] / total_items * 100:.2f}%")
    print(f"  Watermarked Better: {interestingness_result['watermarked_better'] / total_items * 100:.2f}%")
    print(f"  Tie: {interestingness_result['tie'] / total_items * 100:.2f}%")
    
    # 4. 종합 분석
    print("\n4. COMPREHENSIVE ANALYSIS")
    print("-" * 50)
    
    # Perplexity 품질 (낮을수록 좋음)
    if avg_perplexity_watermarked <= avg_perplexity_original * 1.1:  # 10% 이내 차이
        print("✓ Perplexity: Watermarked text maintains good language model quality")
    else:
        print("⚠ Perplexity: Watermarked text shows degraded language model quality")
    
    # Log Diversity 품질 (높을수록 좋음)
    if avg_log_diversity_watermarked >= avg_log_diversity_original * 0.9:  # 10% 이내 차이
        print("✓ Log Diversity: Watermarked text maintains good lexical diversity")
    else:
        print("⚠ Log Diversity: Watermarked text shows reduced lexical diversity")
    
    # GPT 품질 평가 종합
    total_original_better = (relevance_result['original_better'] + 
                           coherence_result['original_better'] + 
                           interestingness_result['original_better'])
    total_watermarked_better = (relevance_result['watermarked_better'] + 
                               coherence_result['watermarked_better'] + 
                               interestingness_result['watermarked_better'])
    
    if total_watermarked_better > total_original_better:
        print("✓ Overall Quality: Watermarked text is rated higher in quality metrics")
    elif total_watermarked_better < total_original_better:
        print("⚠ Overall Quality: Original text is rated higher in quality metrics")
    else:
        print("✓ Overall Quality: Both texts show similar quality levels")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)

# 메인 함수 실행
if __name__ == "__main__":
    main()

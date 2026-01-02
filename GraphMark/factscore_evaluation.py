"""
FactScore Evaluation Script
공식 FactScore 패키지를 사용하여 텍스트의 사실성 정확도를 평가합니다.

FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
Min et al., 2023

사용 전 준비사항:
1. pip install --upgrade factscore
2. python -m spacy download en_core_web_sm
3. python -m factscore.download_data --llama_7B_HF_path "llama-7B"
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add local factscore repository to path (pip install 없이 사용)
factscore_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "factscore_repo")
if os.path.exists(factscore_repo_path):
    sys.path.insert(0, factscore_repo_path)
    print(f"✅ Using local factscore repository: {factscore_repo_path}")
else:
    print(f"⚠️  Warning: factscore_repo not found at {factscore_repo_path}")
    print(f"   Please clone: git clone https://github.com/shmsw25/FActScore.git factscore_repo")


class FactScoreEvaluator:
    """
    FactScore 평가를 수행하는 클래스
    공식 FactScore 패키지를 사용하여 Wikipedia 기준으로 사실성을 평가합니다.
    """
    
    def __init__(self, 
                 openai_key: Optional[str] = None,
                 gpt3_cache_dir: Optional[str] = None,
                 model_name: str = "retrieval+llama",
                 llama_model_path: Optional[str] = None,
                 device_id: Optional[int] = None):
        """
        FactScore 평가기 초기화
        
        Args:
            openai_key: OpenAI API 키 (GPT-3 사용 시 필요)
            gpt3_cache_dir: GPT-3 캐시 디렉토리
            model_name: 사용할 모델 ("retrieval+llama", "retrieval+chatgpt", "retrieval+llama+npm" 등)
            llama_model_path: LLaMA 모델 경로 (예: "meta-llama/Meta-Llama-3-8B-Instruct" 또는 로컬 경로)
            device_id: GPU device ID (None이면 CUDA_VISIBLE_DEVICES 사용)
        """
        print("Initializing FactScore Evaluator with official package...")
        
        # FactScore 패키지가 로컬 저장소에서 사용 가능한지 확인
        try:
            from factscore.factscorer import FactScorer
            self.FactScorer = FactScorer
            self.use_official_factscore = True
            print("✅ FactScore module found (from local repository)")
        except ImportError as e:
            raise ImportError(
                f"FactScore module not found. Error: {e}\n"
                "Please clone the repository:\n"
                "  cd /home/wooseok/KG_Mark\n"
                "  git clone https://github.com/shmsw25/FActScore.git factscore_repo\n\n"
                "Then install dependencies (optional, if needed):\n"
                "  pip install sentence_transformers transformers pysqlite3 openai rank_bm25 spacy\n"
                "  python -m spacy download en_core_web_sm"
            )
        
        # FactScorer 초기화
        try:
            print(f"Initializing FactScorer with model: {model_name}...")
            
            # openai_key 처리: 파일 경로 또는 직접 키
            openai_key_path = openai_key
            if openai_key and not os.path.exists(openai_key):
                # 키가 파일 경로가 아니면 임시 파일 생성
                temp_key_file = os.path.join(os.path.expanduser("~"), ".openai_key_temp.txt")
                with open(temp_key_file, 'w') as f:
                    f.write(openai_key)
                openai_key_path = temp_key_file
            
            # gpt3_cache_dir이 None이면 기본값 사용
            if gpt3_cache_dir is None:
                gpt3_cache_dir = ".cache/factscore"
            
            # llama 모델 경로 설정
            model_dir = gpt3_cache_dir
            if llama_model_path and "llama" in model_name:
                # llama-3-8b-inst 사용 시 모델 경로 지정
                # FactScorer는 inst-llama-7B를 하드코딩하므로, 
                # llama-3-8b-inst를 사용하려면 FactScorer를 수정하거나 래핑 필요
                print(f"  Using custom LLaMA model: {llama_model_path}")
                # 모델 경로를 model_dir에 저장 (나중에 사용)
                self.llama_model_path = llama_model_path
            else:
                self.llama_model_path = None
            
            # FactScorer 초기화
            self.scorer = FactScorer(
                model_name=model_name,
                openai_key=openai_key_path if openai_key_path else "api.key",
                model_dir=model_dir,
                cache_dir=gpt3_cache_dir
            )
            
            # llama-3-8b-inst를 사용하는 경우 CLM 모델 경로 수정
            if llama_model_path and "llama" in model_name and hasattr(self.scorer, 'lm') and self.scorer.lm:
                print(f"  Replacing LLaMA model with: {llama_model_path}")
                # CLM을 새로 초기화
                from factscore.clm import CLM
                cache_file = os.path.join(gpt3_cache_dir, "llama-3-8b-inst.pkl")
                self.scorer.lm = CLM("llama-3-8b-inst", model_dir=llama_model_path, cache_file=cache_file)
                print("  ✅ LLaMA model replaced successfully!")
            
            print("✅ FactScorer initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing FactScorer: {e}")
            import traceback
            traceback.print_exc()
            print("\nPlease ensure:")
            print("  1. factscore_repo is cloned: git clone https://github.com/shmsw25/FActScore.git factscore_repo")
            print("  2. Dependencies are installed (if needed)")
            print("  3. SpaCy model is downloaded: python -m spacy download en_core_web_sm")
            if llama_model_path:
                print(f"  4. LLaMA model is available at: {llama_model_path}")
            raise
    
    def calculate_factscore(self, text: str, topic: str = None) -> Dict:
        """
        텍스트의 FactScore를 계산합니다 (공식 FactScore 패키지 사용).
        
        Args:
            text: 평가할 텍스트
            topic: 텍스트의 주제/엔티티 (예: "Albert Einstein", None이면 "general" 사용)
            
        Returns:
            FactScore 결과 딕셔너리
        """
        if not text:
            return {
                "score": 0.0,
                "init_score": 0.0,
                "num_facts": 0,
                "num_accurate": 0,
                "num_verifiable": 0,
                "details": []
            }
        
        try:
            # FactScorer.get_score()는 topics와 generations 리스트를 받음
            # 단일 텍스트의 경우 리스트로 변환
            topics = [topic if topic else "general"]
            generations = [text]
            
            # get_score 호출 (gamma=10은 기본값, length penalty 적용)
            result = self.scorer.get_score(
                topics=topics,
                generations=generations,
                gamma=10,
                verbose=False
            )
            
            # result는 dict: {"score", "init_score", "respond_ratio", "decisions", "num_facts_per_response"}
            score = result.get("score", 0.0)
            init_score = result.get("init_score", score)
            decisions = result.get("decisions", [])
            
            # decisions는 리스트의 리스트 (각 샘플마다 하나의 리스트)
            if decisions and len(decisions) > 0:
                sample_decisions = decisions[0] if decisions[0] is not None else []
            else:
                sample_decisions = []
            
            # decisions에서 상세 정보 추출
            num_facts = len(sample_decisions) if sample_decisions else 0
            num_accurate = sum(1 for d in sample_decisions if d.get("is_supported", False)) if sample_decisions else 0
            num_verifiable = num_facts  # FactScore에서 모든 fact는 verifiable로 간주
            
            return {
                "score": float(score),
                "init_score": float(init_score),
                "num_facts": num_facts,
                "num_accurate": num_accurate,
                "num_verifiable": num_verifiable,
                "details": sample_decisions if sample_decisions else []
            }
            
        except Exception as e:
            print(f"❌ Error calculating FactScore: {e}")
            import traceback
            traceback.print_exc()
            return {
                "score": 0.0,
                "init_score": 0.0,
                "num_facts": 0,
                "num_accurate": 0,
                "num_verifiable": 0,
                "details": []
            }
    
    def evaluate_jsonl(self, 
                      jsonl_file: str,
                      text_field: str = "watermarked_text",
                      topic_field: Optional[str] = None,
                      max_samples: Optional[int] = None,
                      output_file: Optional[str] = None) -> Dict:
        """
        JSONL 파일의 모든 샘플에 대해 FactScore를 평가합니다.
        
        Args:
            jsonl_file: 입력 JSONL 파일 경로
            text_field: 평가할 텍스트 필드 이름
            topic_field: 주제 필드 이름 (optional)
            max_samples: 최대 평가 샘플 수 (None이면 전체)
            output_file: 결과 저장 파일 경로 (optional)
            
        Returns:
            전체 통계 딕셔너리
        """
        # 경로 정규화 (절대 경로로 변환)
        jsonl_file = os.path.abspath(os.path.expanduser(jsonl_file))
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"File not found: {jsonl_file}")
        
        all_results = []
        total_samples = 0
        valid_samples = 0
        
        print(f"\n{'='*80}")
        print(f"FactScore Evaluation")
        print(f"{'='*80}")
        print(f"Input file: {jsonl_file}")
        print(f"Text field: {text_field}")
        print(f"Max samples: {max_samples if max_samples else 'all'}")
        print(f"{'='*80}\n")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and total_samples >= max_samples:
                    break
                
                total_samples += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue
                
                # 텍스트 필드 확인 (다양한 필드 이름 지원)
                text = None
                if text_field in data:
                    text = data[text_field]
                elif 'watermarked_text' in data:
                    text = data['watermarked_text']
                elif 'original_text' in data:
                    text = data['original_text']
                elif 'natural_text' in data:
                    text = data['natural_text']
                elif 'text1' in data:  # PostMark format
                    text = data['text1']
                elif 'text2' in data:  # PostMark format
                    text = data['text2']
                
                if not text:
                    print(f"⚠️  Warning: No text found in sample {line_num}, skipping...")
                    continue
                
                # Topic 확인 (FactScore는 topic/entity가 필요할 수 있음)
                topic = None
                if topic_field and topic_field in data:
                    topic = data[topic_field]
                elif 'topic' in data:
                    topic = data['topic']
                elif 'entity' in data:
                    topic = data['entity']
                elif 'name' in data:
                    topic = data['name']
                # topic이 없으면 None으로 두고 FactScore가 자동으로 추출하도록 함
                
                print(f"\n[{line_num}] Processing sample {valid_samples + 1}...")
                print(f"  Text length: {len(text)} characters")
                
                # FactScore 계산
                result = self.calculate_factscore(text, topic)
                
                # 결과 저장
                sample_result = {
                    "sample_id": line_num,
                    "topic": topic,
                    "text_length": len(text),
                    "factscore": result["score"],
                    "init_score": result.get("init_score", result["score"]),
                    "num_facts": result["num_facts"],
                    "num_accurate": result["num_accurate"],
                    "num_verifiable": result["num_verifiable"],
                    "details": result["details"]
                }
                
                all_results.append(sample_result)
                valid_samples += 1
                
                print(f"  ✅ FactScore: {result['score']:.4f} ({result['num_accurate']}/{result['num_facts']} accurate facts)")
                
                # 진행 상황 출력
                if valid_samples % 10 == 0:
                    print(f"\n  Progress: {valid_samples} samples processed...")
        
        if valid_samples == 0:
            print("❌ Error: No valid samples found!")
            return {}
        
        # 통계 계산
        factscores = [r["factscore"] for r in all_results]
        num_facts_list = [r["num_facts"] for r in all_results]
        num_accurate_list = [r["num_accurate"] for r in all_results]
        num_verifiable_list = [r["num_verifiable"] for r in all_results]
        
        stats = {
            "total_samples": valid_samples,
            "factscore_stats": {
                "mean": float(np.mean(factscores)),
                "median": float(np.median(factscores)),
                "std": float(np.std(factscores)),
                "min": float(np.min(factscores)),
                "max": float(np.max(factscores))
            },
            "num_facts_stats": {
                "mean": float(np.mean(num_facts_list)),
                "median": float(np.median(num_facts_list)),
                "min": int(np.min(num_facts_list)),
                "max": int(np.max(num_facts_list)),
                "total": int(np.sum(num_facts_list))
            },
            "num_accurate_stats": {
                "mean": float(np.mean(num_accurate_list)),
                "median": float(np.median(num_accurate_list)),
                "min": int(np.min(num_accurate_list)),
                "max": int(np.max(num_accurate_list)),
                "total": int(np.sum(num_accurate_list))
            },
            "num_verifiable_stats": {
                "mean": float(np.mean(num_verifiable_list)),
                "median": float(np.median(num_verifiable_list)),
                "min": int(np.min(num_verifiable_list)),
                "max": int(np.max(num_verifiable_list)),
                "total": int(np.sum(num_verifiable_list))
            },
            "per_sample_results": all_results
        }
        
        # 결과 출력
        print(f"\n{'='*80}")
        print(f"FactScore Evaluation Results")
        print(f"{'='*80}")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"\nFactScore Statistics:")
        print(f"  Mean:   {stats['factscore_stats']['mean']:.4f}")
        print(f"  Median: {stats['factscore_stats']['median']:.4f}")
        print(f"  Std:    {stats['factscore_stats']['std']:.4f}")
        print(f"  Min:    {stats['factscore_stats']['min']:.4f}")
        print(f"  Max:    {stats['factscore_stats']['max']:.4f}")
        print(f"\nAtomic Facts Statistics:")
        print(f"  Total Facts:     {stats['num_facts_stats']['total']}")
        print(f"  Mean per Sample: {stats['num_facts_stats']['mean']:.2f}")
        print(f"  Accurate Facts:  {stats['num_accurate_stats']['total']} ({stats['num_accurate_stats']['total']/stats['num_facts_stats']['total']*100:.2f}%)")
        print(f"  Verifiable Facts: {stats['num_verifiable_stats']['total']} ({stats['num_verifiable_stats']['total']/stats['num_facts_stats']['total']*100:.2f}%)")
        print(f"{'='*80}\n")
        
        # 결과 저장
        if output_file:
            # 상세 결과는 별도 파일로 저장
            stats_to_save = {k: v for k, v in stats.items() if k != 'per_sample_results'}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
            print(f"✅ Statistics saved to: {output_file}")
            
            # Per-sample 결과 저장
            per_sample_file = output_file.replace('.json', '_per_sample.json')
            with open(per_sample_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Per-sample results saved to: {per_sample_file}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FactScore using LLaMA-3-8B-Instruct"
    )
    parser.add_argument(
        '--jsonl_file',
        type=str,
        required=True,
        help="Path to JSONL file with texts to evaluate"
    )
    parser.add_argument(
        '--text_field',
        type=str,
        default='watermarked_text',
        help="Field name containing text to evaluate (default: watermarked_text)"
    )
    parser.add_argument(
        '--topic_field',
        type=str,
        default=None,
        help="Field name containing topic (optional)"
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="Path to save results as JSON file (optional)"
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=None,
        help="GPU device ID (default: use CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        '--openai_key',
        type=str,
        default=None,
        help="OpenAI API key (for GPT-3 based models)"
    )
    parser.add_argument(
        '--gpt3_cache_dir',
        type=str,
        default=None,
        help="GPT-3 cache directory"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="retrieval+llama",
        help="FactScore model name (default: retrieval+llama)"
    )
    parser.add_argument(
        '--llama_model_path',
        type=str,
        default=None,
        help="Path to LLaMA model (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct' or local path)"
    )
    
    args = parser.parse_args()
    
    # FactScore 평가기 초기화
    evaluator = FactScoreEvaluator(
        openai_key=args.openai_key,
        gpt3_cache_dir=args.gpt3_cache_dir,
        model_name=args.model_name,
        llama_model_path=args.llama_model_path,
        device_id=args.device_id
    )
    
    # 평가 수행
    stats = evaluator.evaluate_jsonl(
        jsonl_file=args.jsonl_file,
        text_field=args.text_field,
        topic_field=args.topic_field,
        max_samples=args.max_samples,
        output_file=args.output_file
    )
    
    if stats:
        print("\n✅ FactScore evaluation completed successfully!")
    else:
        print("\n❌ FactScore evaluation failed!")


if __name__ == "__main__":
    main()


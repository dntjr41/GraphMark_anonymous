#!/usr/bin/env python3
"""
Extract and summarize input_results from JSONL files
"""

import json
import os
from collections import defaultdict
import numpy as np

def load_jsonl(file_path):
    """Load JSONL file and extract input_results"""
    data = []
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    if 'input_results' in item:
                        data.append({
                            'line_number': line_num,
                            'input_results': item['input_results']
                        })
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: JSON 파싱 오류 at line {line_num}: {e}")
                    continue
                except Exception as e:
                    error_count += 1
                    print(f"Warning: 예상치 못한 오류 at line {line_num}: {e}")
                    continue
        
        if error_count > 0:
            print(f"총 {error_count}개의 오류가 발생했습니다.")
        
        print(f"성공적으로 {len(data)}개의 input_results를 로드했습니다.")
        return data
        
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return []

def analyze_input_results(data):
    """Analyze and summarize input_results data"""
    if not data:
        print("분석할 데이터가 없습니다.")
        return
    
    # Initialize summary containers
    summary = {
        'perplexity': {'natural': [], 'watermarked': []},
        'log_diversity': {'natural': [], 'watermarked': []},
        'gpt_evaluation': {
            'relevance': {'natural': [], 'watermarked': []},
            'coherence': {'natural': [], 'watermarked': []},
            'interestingness': {'natural': [], 'watermarked': []}
        }
    }
    
    # Count-based comparison results
    comparison_results = {
        'relevance': {'natural_better': 0, 'watermarked_better': 0, 'tie': 0},
        'coherence': {'natural_better': 0, 'watermarked_better': 0, 'tie': 0},
        'interestingness': {'natural_better': 0, 'watermarked_better': 0, 'tie': 0}
    }
    
    # Collect all values and count comparisons
    for item in data:
        input_results = item['input_results']
        
        # Perplexity
        if 'perplexity' in input_results:
            if 'natural' in input_results['perplexity']:
                summary['perplexity']['natural'].append(input_results['perplexity']['natural'])
            if 'watermarked' in input_results['perplexity']:
                summary['perplexity']['watermarked'].append(input_results['perplexity']['watermarked'])
        
        # Log Diversity
        if 'log_diversity' in input_results:
            if 'natural' in input_results['log_diversity']:
                summary['log_diversity']['natural'].append(input_results['log_diversity']['natural'])
            if 'watermarked' in input_results['log_diversity']:
                summary['log_diversity']['watermarked'].append(input_results['log_diversity']['watermarked'])
        
        # GPT Evaluation with comparison counting
        if 'gpt_evaluation' in input_results:
            gpt_eval = input_results['gpt_evaluation']
            
            for metric in ['relevance', 'coherence', 'interestingness']:
                if metric in gpt_eval:
                    if 'natural' in gpt_eval[metric] and 'watermarked' in gpt_eval[metric]:
                        natural_score = gpt_eval[metric]['natural']
                        watermarked_score = gpt_eval[metric]['watermarked']
                        
                        # Add scores to summary
                        summary['gpt_evaluation'][metric]['natural'].append(natural_score)
                        summary['gpt_evaluation'][metric]['watermarked'].append(watermarked_score)
                        
                        # Count which is better
                        if natural_score > watermarked_score:
                            comparison_results[metric]['natural_better'] += 1
                        elif watermarked_score > natural_score:
                            comparison_results[metric]['watermarked_better'] += 1
                        else:
                            comparison_results[metric]['tie'] += 1
    
    return summary, comparison_results

def calculate_statistics(summary):
    """Calculate statistical measures for each metric"""
    stats = {}
    
    for category, metrics in summary.items():
        if category == 'gpt_evaluation':
            stats[category] = {}
            for metric_name, values in metrics.items():
                stats[category][metric_name] = {}
                for text_type in ['natural', 'watermarked']:
                    if values[text_type]:
                        stats[category][metric_name][text_type] = {
                            'count': len(values[text_type]),
                            'mean': np.mean(values[text_type]),
                            'std': np.std(values[text_type]),
                            'min': np.min(values[text_type]),
                            'max': np.max(values[text_type]),
                            'median': np.median(values[text_type])
                        }
        else:
            stats[category] = {}
            for text_type in ['natural', 'watermarked']:
                if summary[category][text_type]:
                    stats[category][text_type] = {
                        'count': len(summary[category][text_type]),
                        'mean': np.mean(summary[category][text_type]),
                        'std': np.std(summary[category][text_type]),
                        'min': np.min(summary[category][text_type]),
                        'max': np.max(summary[category][text_type]),
                        'median': np.median(summary[category][text_type])
                    }
    
    return stats

def print_summary(stats, comparison_results, total_items):
    """Print formatted summary of results"""
    print("\n" + "="*80)
    print("INPUT_RESULTS ANALYSIS SUMMARY")
    print("="*80)
    
    # Perplexity
    print("\n📊 PERPLEXITY ANALYSIS:")
    print("-" * 50)
    for text_type in ['natural', 'watermarked']:
        if text_type in stats['perplexity']:
            data = stats['perplexity'][text_type]
            print(f"{text_type.upper()}:")
            print(f"  Count: {data['count']}")
            print(f"  Mean: {data['mean']:.3f} ± {data['std']:.3f}")
            print(f"  Range: [{data['min']:.3f}, {data['max']:.3f}]")
            print(f"  Median: {data['median']:.3f}")
    
    # Log Diversity
    print("\n📊 LOG DIVERSITY ANALYSIS:")
    print("-" * 50)
    for text_type in ['natural', 'watermarked']:
        if text_type in stats['log_diversity']:
            data = stats['log_diversity'][text_type]
            print(f"{text_type.upper()}:")
            print(f"  Count: {data['count']}")
            print(f"  Mean: {data['mean']:.3f} ± {data['std']:.3f}")
            print(f"  Range: [{data['min']:.3f}, {data['max']:.3f}]")
            print(f"  Median: {data['median']:.3f}")
    
    # GPT Evaluation with Count-based Comparison
    print("\n📊 GPT EVALUATION ANALYSIS:")
    print("-" * 50)
    
    # Relevance
    print("Relevance:")
    if 'relevance' in comparison_results:
        rel = comparison_results['relevance']
        print(f"  Natural Better: {rel['natural_better'] / total_items * 100:.2f}%")
        print(f"  Watermarked Better: {rel['watermarked_better'] / total_items * 100:.2f}%")
        print(f"  Tie: {rel['tie'] / total_items * 100:.2f}%")
    
    # Coherence
    print("\nCoherence:")
    if 'coherence' in comparison_results:
        coh = comparison_results['coherence']
        print(f"  Natural Better: {coh['natural_better'] / total_items * 100:.2f}%")
        print(f"  Watermarked Better: {coh['watermarked_better'] / total_items * 100:.2f}%")
        print(f"  Tie: {coh['tie'] / total_items * 100:.2f}%")
    
    # Interestingness
    print("\nInterestingness:")
    if 'interestingness' in comparison_results:
        int = comparison_results['interestingness']
        print(f"  Natural Better: {int['natural_better'] / total_items * 100:.2f}%")
        print(f"  Watermarked Better: {int['watermarked_better'] / total_items * 100:.2f}%")
        print(f"  Tie: {int['tie'] / total_items * 100:.2f}%")
    
    # Detailed GPT Evaluation Statistics
    print("\n📊 DETAILED GPT EVALUATION STATISTICS:")
    print("-" * 50)
    for metric in ['relevance', 'coherence', 'interestingness']:
        print(f"\n{metric.upper()}:")
        for text_type in ['natural', 'watermarked']:
            if metric in stats['gpt_evaluation'] and text_type in stats['gpt_evaluation'][metric]:
                data = stats['gpt_evaluation'][metric][text_type]
                print(f"  {text_type.upper()}:")
                print(f"    Count: {data['count']}")
                print(f"    Mean: {data['mean']:.3f} ± {data['std']:.3f}")
                print(f"    Range: [{data['min']:.3f}, {data['max']:.3f}]")
                print(f"    Median: {data['median']:.3f}")

def save_summary_to_file(stats, comparison_results, output_file):
    """Save summary statistics to a JSON file"""
    try:
        output_data = {
            'statistics': stats,
            'comparison_results': comparison_results,
            'summary': {
                'total_items_analyzed': len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]),
                'comparison_percentages': {
                    'relevance': {
                        'natural_better_pct': comparison_results['relevance']['natural_better'] / len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) > 0 else 0,
                        'watermarked_better_pct': comparison_results['relevance']['watermarked_better'] / len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) > 0 else 0,
                        'tie_pct': comparison_results['relevance']['tie'] / len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['relevance']['natural'] if item]) > 0 else 0
                    },
                    'coherence': {
                        'natural_better_pct': comparison_results['coherence']['natural_better'] / len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) > 0 else 0,
                        'watermarked_better_pct': comparison_results['coherence']['watermarked_better'] / len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) > 0 else 0,
                        'tie_pct': comparison_results['coherence']['tie'] / len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['coherence']['natural'] if item]) > 0 else 0
                    },
                    'interestingness': {
                        'natural_better_pct': comparison_results['interestingness']['natural_better'] / len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) > 0 else 0,
                        'watermarked_better_pct': comparison_results['interestingness']['watermarked_better'] / len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) > 0 else 0,
                        'tie_pct': comparison_results['interestingness']['tie'] / len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) * 100 if len([item for item in stats['gpt_evaluation']['interestingness']['natural'] if item]) > 0 else 0
                    }
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n📁 요약 통계가 {output_file}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 오류: {e}")

def main():
    """Main function"""
    # 파일 경로 설정
    file_path = "/home/wooseok/KG_Mark/outputs/c4/llama-3-8b-inst_GraphMark_15_with_results.jsonl"
    
    print("🔍 INPUT_RESULTS 분석 시작...")
    print(f"📂 파일 경로: {file_path}")
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    # 데이터 로드
    data = load_jsonl(file_path)
    
    if not data:
        print("❌ 데이터를 로드할 수 없습니다.")
        return
    
    # 분석
    summary, comparison_results = analyze_input_results(data)
    
    if not summary:
        print("❌ 분석할 수 있는 데이터가 없습니다.")
        return
    
    # 통계 계산
    stats = calculate_statistics(summary)
    
    # 결과 출력
    print_summary(stats, comparison_results, len(data))
    
    # 파일로 저장
    output_file = "input_results_summary.json"
    save_summary_to_file(stats, comparison_results, output_file)
    
    print("\n✅ 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate win/tie/lose statistics from _with_results.jsonl file with sampling support.

Usage:
    python generate_win_tie_stats.py --input_file <path_to_with_results.jsonl> [--sample_size N] [--seed SEED]
"""

import json
import argparse
import random
from typing import Dict, List, Tuple
import os


def load_data(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def sample_data(data: List[Dict], sample_size: int = None, seed: int = None) -> List[Dict]:
    """Sample data randomly if sample_size is specified."""
    if seed is not None:
        random.seed(seed)
    
    if sample_size is None or sample_size >= len(data):
        return data
    
    return random.sample(data, sample_size)


def calculate_win_tie_lose(original_score: float, watermarked_score: float) -> str:
    """Determine win/tie/lose based on scores."""
    if watermarked_score > original_score:
        return "win"
    elif watermarked_score == original_score:
        return "tie"
    else:
        return "lose"


def calculate_statistics(data: List[Dict]) -> Dict:
    """Calculate win/tie/lose statistics from data with input_results."""
    # Initialize counters
    coherence_result = {"win": 0, "tie": 0, "lose": 0}
    relevance_result = {"win": 0, "tie": 0, "lose": 0}
    groundedness_result = {"win": 0, "tie": 0, "lose": 0}
    log_diversity_result = {"win": 0, "tie": 0, "lose": 0}
    factscore_result = {"win": 0, "tie": 0, "lose": 0}
    
    # For factscore averages
    factscore_original_sum = 0.0
    factscore_watermarked_sum = 0.0
    factscore_count = 0
    
    # Process each item
    processed_count = 0
    for item in data:
        if "input_results" not in item:
            continue
        
        input_results = item["input_results"]
        processed_count += 1
        
        # Coherence
        if "gpt_evaluation" in input_results and "coherence" in input_results["gpt_evaluation"]:
            coherence = input_results["gpt_evaluation"]["coherence"]
            if "original" in coherence and "watermarked" in coherence:
                result = calculate_win_tie_lose(
                    coherence["original"], 
                    coherence["watermarked"]
                )
                coherence_result[result] += 1
        
        # Relevance
        if "gpt_evaluation" in input_results and "relevance" in input_results["gpt_evaluation"]:
            relevance = input_results["gpt_evaluation"]["relevance"]
            if "original" in relevance and "watermarked" in relevance:
                result = calculate_win_tie_lose(
                    relevance["original"], 
                    relevance["watermarked"]
                )
                relevance_result[result] += 1
        
        # Groundedness
        if "gpt_evaluation" in input_results and "groundedness" in input_results["gpt_evaluation"]:
            groundedness = input_results["gpt_evaluation"]["groundedness"]
            if "original" in groundedness and "watermarked" in groundedness:
                result = calculate_win_tie_lose(
                    groundedness["original"], 
                    groundedness["watermarked"]
                )
                groundedness_result[result] += 1
        
        # Log Diversity
        if "log_diversity" in input_results:
            log_diversity = input_results["log_diversity"]
            if "original" in log_diversity and "watermarked" in log_diversity:
                result = calculate_win_tie_lose(
                    log_diversity["original"], 
                    log_diversity["watermarked"]
                )
                log_diversity_result[result] += 1
        
        # FActScore
        if "factscore" in input_results:
            factscore = input_results["factscore"]
            if "original" in factscore and "watermarked" in factscore:
                orig_score = factscore["original"]
                water_score = factscore["watermarked"]
                
                result = calculate_win_tie_lose(orig_score, water_score)
                factscore_result[result] += 1
                
                # Accumulate for average
                factscore_original_sum += orig_score
                factscore_watermarked_sum += water_score
                factscore_count += 1
    
    # Calculate rates
    total_items = processed_count
    
    # Calculate averages for factscore
    avg_factscore_original = factscore_original_sum / factscore_count if factscore_count > 0 else 0.0
    avg_factscore_watermarked = factscore_watermarked_sum / factscore_count if factscore_count > 0 else 0.0
    
    # Build statistics dictionary
    stats = {
        "coherence": {
            "win": coherence_result["win"],
            "tie": coherence_result["tie"],
            "lose": coherence_result["lose"],
            "win_rate": coherence_result["win"] / total_items * 100 if total_items > 0 else 0.0,
            "tie_rate": coherence_result["tie"] / total_items * 100 if total_items > 0 else 0.0,
            "lose_rate": coherence_result["lose"] / total_items * 100 if total_items > 0 else 0.0
        },
        "relevance": {
            "win": relevance_result["win"],
            "tie": relevance_result["tie"],
            "lose": relevance_result["lose"],
            "win_rate": relevance_result["win"] / total_items * 100 if total_items > 0 else 0.0,
            "tie_rate": relevance_result["tie"] / total_items * 100 if total_items > 0 else 0.0,
            "lose_rate": relevance_result["lose"] / total_items * 100 if total_items > 0 else 0.0
        },
        "groundedness": {
            "win": groundedness_result["win"],
            "tie": groundedness_result["tie"],
            "lose": groundedness_result["lose"],
            "win_rate": groundedness_result["win"] / total_items * 100 if total_items > 0 else 0.0,
            "tie_rate": groundedness_result["tie"] / total_items * 100 if total_items > 0 else 0.0,
            "lose_rate": groundedness_result["lose"] / total_items * 100 if total_items > 0 else 0.0
        },
        "log_diversity": {
            "win": log_diversity_result["win"],
            "tie": log_diversity_result["tie"],
            "lose": log_diversity_result["lose"],
            "win_rate": log_diversity_result["win"] / total_items * 100 if total_items > 0 else 0.0,
            "tie_rate": log_diversity_result["tie"] / total_items * 100 if total_items > 0 else 0.0,
            "lose_rate": log_diversity_result["lose"] / total_items * 100 if total_items > 0 else 0.0
        },
        "factscore": {
            "win": factscore_result["win"],
            "tie": factscore_result["tie"],
            "lose": factscore_result["lose"],
            "win_rate": factscore_result["win"] / total_items * 100 if total_items > 0 else 0.0,
            "tie_rate": factscore_result["tie"] / total_items * 100 if total_items > 0 else 0.0,
            "lose_rate": factscore_result["lose"] / total_items * 100 if total_items > 0 else 0.0,
            "avg_original": avg_factscore_original,
            "avg_watermarked": avg_factscore_watermarked
        },
        "total_items": total_items
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate win/tie/lose statistics from _with_results.jsonl file with sampling support"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input _with_results.jsonl file"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of items to sample (default: use all items)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling (default: None)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSON file (default: <input_file>_win_tie_stats.json)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} items")
    
    # Sample if requested
    if args.sample_size is not None:
        print(f"Sampling {args.sample_size} items (seed={args.seed})...")
        data = sample_data(data, args.sample_size, args.seed)
        print(f"Sampled {len(data)} items")
    
    # Calculate statistics
    print("Calculating win/tie/lose statistics...")
    stats = calculate_statistics(data)
    
    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        # Replace _with_results.jsonl with _win_tie_stats.json
        if args.input_file.endswith("_with_results.jsonl"):
            output_file = args.input_file.replace("_with_results.jsonl", "_win_tie_stats.json")
        else:
            output_file = args.input_file.replace(".jsonl", "_win_tie_stats.json")
    
    # Save statistics
    print(f"Saving statistics to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nStatistics saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total items processed: {stats['total_items']}")
    print(f"  Coherence: Win={stats['coherence']['win']} ({stats['coherence']['win_rate']:.1f}%), "
          f"Tie={stats['coherence']['tie']} ({stats['coherence']['tie_rate']:.1f}%), "
          f"Lose={stats['coherence']['lose']} ({stats['coherence']['lose_rate']:.1f}%)")
    print(f"  Relevance: Win={stats['relevance']['win']} ({stats['relevance']['win_rate']:.1f}%), "
          f"Tie={stats['relevance']['tie']} ({stats['relevance']['tie_rate']:.1f}%), "
          f"Lose={stats['relevance']['lose']} ({stats['relevance']['lose_rate']:.1f}%)")
    print(f"  Groundedness: Win={stats['groundedness']['win']} ({stats['groundedness']['win_rate']:.1f}%), "
          f"Tie={stats['groundedness']['tie']} ({stats['groundedness']['tie_rate']:.1f}%), "
          f"Lose={stats['groundedness']['lose']} ({stats['groundedness']['lose_rate']:.1f}%)")
    print(f"  Log Diversity: Win={stats['log_diversity']['win']} ({stats['log_diversity']['win_rate']:.1f}%), "
          f"Tie={stats['log_diversity']['tie']} ({stats['log_diversity']['tie_rate']:.1f}%), "
          f"Lose={stats['log_diversity']['lose']} ({stats['log_diversity']['lose_rate']:.1f}%)")
    print(f"  FActScore: Win={stats['factscore']['win']} ({stats['factscore']['win_rate']:.1f}%), "
          f"Tie={stats['factscore']['tie']} ({stats['factscore']['tie_rate']:.1f}%), "
          f"Lose={stats['factscore']['lose']} ({stats['factscore']['lose_rate']:.1f}%)")
    if stats['factscore']['win'] + stats['factscore']['tie'] + stats['factscore']['lose'] > 0:
        print(f"  FActScore Average: Original={stats['factscore']['avg_original']:.6f}, "
              f"Watermarked={stats['factscore']['avg_watermarked']:.6f}")


if __name__ == "__main__":
    main()


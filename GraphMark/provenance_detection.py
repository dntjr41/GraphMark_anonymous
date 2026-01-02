import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime
from watermark_detection import WatermarkDetector


class ProvenanceDetector(WatermarkDetector):
    """
    Provenance Detection: 정확하게 실제 삽입된 Triplet만을 탐지하는 클래스
    Precision, Recall, F1-score를 계산합니다.
    """
    
    def __init__(self, device_id=None):
        """
        Initialize provenance detector
        
        Args:
            device_id: Not used (kept for compatibility)
        """
        super().__init__(device_id)
    
    def _sentence_contains_entity(self, sentence: str, entity_names: List[str]) -> bool:
        """
        Check if sentence contains any of the entity names (case-insensitive)
        더 유연한 매칭: 단어 단위 매칭과 substring 매칭 모두 사용
        """
        if not entity_names:
            return False
        
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        for entity_name in entity_names:
            if not entity_name or not isinstance(entity_name, str):
                continue
            
            entity_lower = entity_name.lower().strip()
            if not entity_lower:
                continue
            
            # Method 1: Exact substring matching (기존 방식)
            if entity_lower in sentence_lower:
                return True
            
            # Method 2: Word-level matching (더 유연한 매칭)
            # Entity name을 단어로 분리하여 모든 단어가 문장에 있는지 확인
            entity_words = entity_lower.split()
            if len(entity_words) > 1:
                # Multi-word entity: 모든 단어가 순서대로 나타나는지 확인
                if self._words_appear_in_order(sentence_lower, entity_words):
                    return True
            elif len(entity_words) == 1:
                # Single word entity: 단어가 문장에 있는지 확인 (단, 너무 짧은 단어는 제외)
                if len(entity_words[0]) >= 3 and entity_words[0] in sentence_words:
                    return True
        
        return False
    
    def _words_appear_in_order(self, text: str, words: List[str]) -> bool:
        """Check if words appear in order in the text"""
        if not words:
            return False
        
        text_lower = text.lower()
        last_pos = -1
        
        for word in words:
            pos = text_lower.find(word, last_pos + 1)
            if pos == -1:
                return False
            last_pos = pos
        
        return True
    
    def detect_triplets_in_text(self, text: str, candidate_triplets: List[List[str]], 
                                 min_detected_count: int = 1) -> Dict:
        """
        텍스트에서 실제로 탐지된 triplet 리스트를 반환
        
        Args:
            text: 탐지할 텍스트
            candidate_triplets: 탐지 대상 triplet 리스트 (selected_triplets 등)
            min_detected_count: 최소 탐지 개수 (사용하지 않지만 호환성을 위해 유지)
        
        Returns:
            Dictionary with detected triplets
        """
        if not candidate_triplets:
            return {
                "detected_triplets": [],
                "total_candidate_triplets": 0,
                "matching_sentences": []
            }
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return {
                "detected_triplets": [],
                "total_candidate_triplets": len(candidate_triplets),
                "matching_sentences": []
            }
        
        # Extract all Head and Tail entity names from candidate_triplets
        head_entity_ids = set()
        tail_entity_ids = set()
        for head_id, relation_id, tail_id in candidate_triplets:
            head_entity_ids.add(head_id)
            tail_entity_ids.add(tail_id)
        
        # Get all name candidates for each entity
        head_entity_names = {}
        tail_entity_names = {}
        
        for entity_id in head_entity_ids:
            head_entity_names[entity_id] = self._get_all_entity_names(entity_id)
        
        for entity_id in tail_entity_ids:
            tail_entity_names[entity_id] = self._get_all_entity_names(entity_id)
        
        # Find sentences containing both head and tail entities
        detected_triplets = set()
        matching_sentences = []
        
        for sent_idx, sentence in enumerate(sentences):
            for head_id, relation_id, tail_id in candidate_triplets:
                triplet_tuple = (head_id, relation_id, tail_id)
                
                # Skip if already detected
                if triplet_tuple in detected_triplets:
                    continue
                
                # Get all name candidates for head and tail
                head_names = head_entity_names.get(head_id, [])
                tail_names = tail_entity_names.get(tail_id, [])
                
                # Check if sentence contains both head and tail entities
                head_found = self._sentence_contains_entity(sentence, head_names)
                tail_found = self._sentence_contains_entity(sentence, tail_names)
                
                if head_found and tail_found:
                    detected_triplets.add(triplet_tuple)
                    matching_sentences.append({
                        "sentence_idx": sent_idx,
                        "sentence": sentence,
                        "triplet": list(triplet_tuple)
                    })
        
        return {
            "detected_triplets": [list(t) for t in detected_triplets],
            "total_candidate_triplets": len(candidate_triplets),
            "matching_sentences": matching_sentences
        }
    
    def _verify_triplet_actually_inserted(self, text: str, triplet: List[str]) -> bool:
        """
        Triplet이 실제로 텍스트에 삽입되었는지 확인 (triplet_insertion_verification.py 참고)
        
        Args:
            text: Watermarked text
            triplet: Triplet [head_id, relation_id, tail_id]
        
        Returns:
            True if both head and tail entities are found in the same sentence
        """
        if len(triplet) < 3:
            return False
        
        head_id, tail_id = triplet[0], triplet[2]
        
        # Get all entity names for head and tail
        head_entity_names = self._get_all_entity_names(head_id)
        tail_entity_names = self._get_all_entity_names(tail_id)
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Check if both head and tail are found in the same sentence
        for sentence in sentences:
            sentence_lower = sentence.lower()
            head_found = False
            tail_found = False
            
            # Check head entity
            for head_name in head_entity_names:
                if head_name and isinstance(head_name, str):
                    if head_name.lower() in sentence_lower:
                        head_found = True
                        break
            
            # Check tail entity
            for tail_name in tail_entity_names:
                if tail_name and isinstance(tail_name, str):
                    if tail_name.lower() in sentence_lower:
                        tail_found = True
                        break
            
            # If both found in same sentence, triplet is actually inserted
            if head_found and tail_found:
                return True
        
        return False
    
    def calculate_provenance_metrics(self, actual_triplets: List[List[str]], 
                                     detected_triplets: List[List[str]],
                                     text: str = None) -> Dict:
        """
        Precision, Recall, F1-score 계산
        실제로 텍스트에 삽입된 triplet만 actual_triplets로 고려
        
        Args:
            actual_triplets: 선택된 triplet 리스트 (selected_triplets)
            detected_triplets: 탐지된 triplet 리스트
            text: Watermarked text (실제 삽입 여부 확인용, None이면 기존 방식 사용)
        
        Returns:
            Dictionary with Precision, Recall, F1-score, TP, FP, FN
        """
        # If text is provided, filter actual_triplets to only those actually inserted
        if text:
            actually_inserted_triplets = [
                t for t in actual_triplets 
                if self._verify_triplet_actually_inserted(text, t)
            ]
        else:
            # Fallback: use all actual_triplets (backward compatibility)
            actually_inserted_triplets = actual_triplets
        
        # Convert to tuples for set operations
        actual_set = {tuple(t) for t in actually_inserted_triplets}
        detected_set = {tuple(t) for t in detected_triplets}
        
        # Calculate TP, FP, FN
        true_positives = actual_set & detected_set  # 실제 삽입되고 탐지됨
        false_positives = detected_set - actual_set  # 탐지되었지만 실제로는 삽입되지 않음
        false_negatives = actual_set - detected_set  # 실제 삽입되었지만 탐지되지 않음
        
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        # Calculate Precision, Recall, F1
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "actual_count": len(actual_triplets),  # Original count
            "actually_inserted_count": len(actually_inserted_triplets),  # Actually inserted count
            "detected_count": len(detected_triplets),
            "true_positive_triplets": [list(t) for t in true_positives],
            "false_positive_triplets": [list(t) for t in false_positives],
            "false_negative_triplets": [list(t) for t in false_negatives],
            "not_inserted_triplets": [list(t) for t in {tuple(t) for t in actual_triplets} - actual_set]  # Selected but not actually inserted
        }
    
    def evaluate_provenance(self, jsonl_file: str, 
                           use_paraphrased_text: bool = False,
                           max_samples: Optional[int] = None,
                           save_results: Optional[str] = None) -> Dict:
        """
        Watermarked Text 및 Attacked Text에 대해 실제 삽입된 Triplet만을 정확하게 탐지하고
        Precision, Recall, F1-score를 계산
        
        Args:
            jsonl_file: Path to JSONL file with watermarked examples
            use_paraphrased_text: If True, evaluate on paraphrased_watermarked_text with selected_paraphrased_triplets
            max_samples: Maximum number of samples to evaluate (None = all)
            save_results: Path to save results as JSONL (optional)
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = []
        total_samples = 0
        valid_samples = 0
        
        # Aggregate metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_actual = 0
        total_actually_inserted = 0
        total_detected = 0
        
        # Per-sample metrics for averaging
        precisions = []
        recalls = []
        f1_scores = []
        hits_at_1 = []  # Hits@1: 각 샘플에서 최소 1개의 실제 triplet이 탐지되었는지
        
        print(f"\n{'='*80}")
        print(f"Provenance Detection Evaluation")
        print(f"{'='*80}")
        print(f"Input file: {jsonl_file}")
        print(f"Mode: {'Paraphrased Text' if use_paraphrased_text else 'Watermarked Text'}")
        if max_samples:
            print(f"Max samples: {max_samples}")
        print(f"{'='*80}\n")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and total_samples >= max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue
                
                total_samples += 1
                
                # Get text and triplets based on mode
                if use_paraphrased_text:
                    # Try both field names for compatibility
                    text = data.get('paraphrased_watermarked', '') or data.get('paraphrased_watermarked_text', '')
                    actual_triplets = data.get('selected_paraphrased_triplets', [])
                    text_type = "paraphrased_watermarked"
                else:
                    text = data.get('watermarked_text', '')
                    actual_triplets = data.get('selected_triplets', [])
                    text_type = "watermarked"
                
                # Skip if no text or no triplets
                if not text or not actual_triplets:
                    if not text:
                        print(f"Warning: Sample {line_num} has no {text_type} text, skipping")
                    if not actual_triplets:
                        print(f"Warning: Sample {line_num} has no actual triplets, skipping")
                    continue
                
                valid_samples += 1
                
                # Detect triplets in text
                detection_result = self.detect_triplets_in_text(text, actual_triplets)
                detected_triplets = detection_result["detected_triplets"]
                
                # Calculate metrics for this sample (text를 전달하여 실제 삽입된 triplet만 고려)
                sample_metrics = self.calculate_provenance_metrics(actual_triplets, detected_triplets, text=text)
                
                # Aggregate statistics
                total_tp += sample_metrics["true_positives"]
                total_fp += sample_metrics["false_positives"]
                total_fn += sample_metrics["false_negatives"]
                total_actual += sample_metrics["actual_count"]
                total_actually_inserted += sample_metrics.get("actually_inserted_count", sample_metrics["actual_count"])
                total_detected += sample_metrics["detected_count"]
                
                precisions.append(sample_metrics["precision"])
                recalls.append(sample_metrics["recall"])
                f1_scores.append(sample_metrics["f1_score"])
                
                # Hits@1: 최소 1개의 실제 triplet이 탐지되었는지 (TP > 0)
                hit_at_1 = 1 if sample_metrics["true_positives"] > 0 else 0
                hits_at_1.append(hit_at_1)
                
                # Store result for this sample
                sample_result = {
                    "sample_id": line_num,
                    "text_type": text_type,
                    "actual_triplets": actual_triplets,
                    "detected_triplets": detected_triplets,
                    "metrics": sample_metrics,
                    "hits_at_1": hit_at_1,  # 이 샘플에서 최소 1개 triplet 탐지 여부
                    "matching_sentences": detection_result["matching_sentences"]
                }
                results.append(sample_result)
                
                # Print progress every 10 samples
                if valid_samples % 10 == 0:
                    print(f"Processed {valid_samples} samples...")
        
        # Calculate aggregate metrics
        if valid_samples == 0:
            print("Error: No valid samples found!")
            return {
                "error": "No valid samples found",
                "total_samples": total_samples,
                "valid_samples": 0
            }
        
        # Macro-averaged metrics (average of per-sample metrics)
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        hits_at_1_score = np.mean(hits_at_1) if hits_at_1 else 0.0
        
        # Micro-averaged metrics (calculated from aggregate TP/FP/FN)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Summary
        summary = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "text_type": "paraphrased_watermarked" if use_paraphrased_text else "watermarked",
            "aggregate_counts": {
                "total_selected_triplets": total_actual,
                "total_actually_inserted_triplets": total_actually_inserted,
                "total_detected_triplets": total_detected,
                "total_true_positives": total_tp,
                "total_false_positives": total_fp,
                "total_false_negatives": total_fn
            },
            "macro_averaged_metrics": {
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1_score": float(macro_f1),
                "hits_at_1": float(hits_at_1_score)
            },
            "micro_averaged_metrics": {
                "precision": float(micro_precision),
                "recall": float(micro_recall),
                "f1_score": float(micro_f1)
            }
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Provenance Detection Results")
        print(f"{'='*80}")
        print(f"Total samples: {total_samples}")
        print(f"Valid samples: {valid_samples}")
        print(f"\nAggregate Counts:")
        print(f"  Total selected triplets: {total_actual}")
        print(f"  Total actually inserted triplets: {total_actually_inserted}")
        print(f"  Total detected triplets: {total_detected}")
        print(f"  True Positives: {total_tp}")
        print(f"  False Positives: {total_fp}")
        print(f"  False Negatives: {total_fn}")
        print(f"  Note: False Negatives only count triplets that were actually inserted but not detected")
        print(f"\nMacro-Averaged Metrics (per-sample average):")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall: {macro_recall:.4f}")
        print(f"  F1-Score: {macro_f1:.4f}")
        print(f"  Hits@1: {hits_at_1_score:.4f}")
        print(f"\nMicro-Averaged Metrics (aggregate calculation):")
        print(f"  Precision: {micro_precision:.4f}")
        print(f"  Recall: {micro_recall:.4f}")
        print(f"  F1-Score: {micro_f1:.4f}")
        print(f"{'='*80}\n")
        
        # Save results if requested
        if save_results:
            output_data = {
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "input_file": jsonl_file,
                "samples": results
            }
            with open(save_results, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_data, ensure_ascii=False, indent=2))
            print(f"Results saved to: {save_results}\n")
        
        return {
            "summary": summary,
            "samples": results
        }


def main():
    parser = argparse.ArgumentParser(description="Provenance Detection: 정확하게 실제 삽입된 Triplet만을 탐지")
    parser.add_argument("--jsonl_file", type=str, required=True, 
                       help="Path to JSONL file with watermarked examples")
    parser.add_argument("--use_paraphrased_text", action="store_true", default=False,
                       help="Use paraphrased_watermarked_text with selected_paraphrased_triplets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save detailed results as JSON file")
    parser.add_argument("--device_id", type=int, default=None, 
                       help="CUDA device ID (not used, kept for compatibility)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ProvenanceDetector(device_id=args.device_id)
    
    # Evaluate provenance
    results = detector.evaluate_provenance(
        args.jsonl_file,
        use_paraphrased_text=args.use_paraphrased_text,
        max_samples=args.max_samples,
        save_results=args.save_results
    )
    
    if "error" not in results:
        print("\n✅ Provenance detection evaluation completed successfully!")


if __name__ == "__main__":
    main()


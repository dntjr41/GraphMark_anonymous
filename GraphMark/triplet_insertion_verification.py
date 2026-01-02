import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime
from watermark_detection import WatermarkDetector


class TripletInsertionVerifier(WatermarkDetector):
    """
    Triplet Insertion Verification: Watermarked Text에 실제로 삽입된 Triplet과
    Head, Tail Entity의 형태 유지 여부를 확인하는 클래스
    """
    
    def __init__(self, device_id=None):
        """
        Initialize triplet insertion verifier
        
        Args:
            device_id: Not used (kept for compatibility)
        """
        super().__init__(device_id)
    
    def verify_triplet_insertion(self, text: str, triplet: List[str]) -> Dict:
        """
        특정 Triplet이 텍스트에 실제로 삽입되었는지, Head와 Tail Entity가 형태를 유지하는지 확인
        
        Args:
            text: Watermarked text
            triplet: Triplet [head_id, relation_id, tail_id]
        
        Returns:
            Dictionary with verification results
        """
        if len(triplet) < 3:
            return {
                "triplet": triplet,
                "is_inserted": False,
                "head_found": False,
                "tail_found": False,
                "both_found": False,
                "head_entity_names": [],
                "tail_entity_names": [],
                "matched_head_names": [],
                "matched_tail_names": [],
                "matching_sentences": []
            }
        
        head_id, relation_id, tail_id = triplet[0], triplet[1], triplet[2]
        
        # Get all entity names for head and tail
        head_entity_names = self._get_all_entity_names(head_id)
        tail_entity_names = self._get_all_entity_names(tail_id)
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Find sentences containing head and/or tail entities
        head_found = False
        tail_found = False
        both_found = False
        matched_head_names = []
        matched_tail_names = []
        matching_sentences = []
        
        for sent_idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            head_matched = False
            tail_matched = False
            matched_head_name = None
            matched_tail_name = None
            
            # Check head entity
            for head_name in head_entity_names:
                if head_name and isinstance(head_name, str):
                    head_lower = head_name.lower()
                    if head_lower in sentence_lower:
                        head_found = True
                        head_matched = True
                        if head_name not in matched_head_names:
                            matched_head_names.append(head_name)
                        matched_head_name = head_name
                        break
            
            # Check tail entity
            for tail_name in tail_entity_names:
                if tail_name and isinstance(tail_name, str):
                    tail_lower = tail_name.lower()
                    if tail_lower in sentence_lower:
                        tail_found = True
                        tail_matched = True
                        if tail_name not in matched_tail_names:
                            matched_tail_names.append(tail_name)
                        matched_tail_name = tail_name
                        break
            
            # If both found in same sentence
            if head_matched and tail_matched:
                both_found = True
                matching_sentences.append({
                    "sentence_idx": sent_idx,
                    "sentence": sentence,
                    "matched_head_name": matched_head_name,
                    "matched_tail_name": matched_tail_name
                })
        
        # Triplet is considered inserted if both head and tail are found
        is_inserted = both_found
        
        return {
            "triplet": triplet,
            "head_id": head_id,
            "relation_id": relation_id,
            "tail_id": tail_id,
            "is_inserted": is_inserted,
            "head_found": head_found,
            "tail_found": tail_found,
            "both_found": both_found,
            "head_entity_names": head_entity_names,
            "tail_entity_names": tail_entity_names,
            "matched_head_names": matched_head_names,
            "matched_tail_names": matched_tail_names,
            "matching_sentences": matching_sentences,
            "entity_form_preserved": {
                "head": head_found,
                "tail": tail_found,
                "both": both_found
            }
        }
    
    def verify_all_triplets(self, jsonl_file: str, 
                           use_paraphrased_text: bool = False,
                           max_samples: Optional[int] = None,
                           save_results: Optional[str] = None) -> Dict:
        """
        JSONL 파일의 모든 샘플에 대해 Triplet 삽입 여부와 Entity 형태 유지 여부를 확인
        
        Args:
            jsonl_file: Path to JSONL file with watermarked examples
            use_paraphrased_text: If True, verify on paraphrased_watermarked_text with selected_paraphrased_triplets
            max_samples: Maximum number of samples to verify (None = all)
            save_results: Path to save detailed results as JSON file
        
        Returns:
            Dictionary with verification results
        """
        results = []
        total_samples = 0
        valid_samples = 0
        
        # Aggregate statistics
        total_triplets = 0
        inserted_triplets = 0
        head_only_found = 0
        tail_only_found = 0
        not_found = 0
        
        # Per-sample statistics
        sample_insertion_rates = []
        
        print(f"\n{'='*80}")
        print(f"Triplet Insertion Verification")
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
                    triplets = data.get('selected_paraphrased_triplets', [])
                    text_type = "paraphrased_watermarked"
                else:
                    text = data.get('watermarked_text', '')
                    triplets = data.get('selected_triplets', [])
                    text_type = "watermarked"
                
                # Skip if no text or no triplets
                if not text or not triplets:
                    if not text:
                        print(f"Warning: Sample {line_num} has no {text_type} text, skipping")
                    if not triplets:
                        print(f"Warning: Sample {line_num} has no triplets, skipping")
                    continue
                
                valid_samples += 1
                
                # Verify each triplet
                sample_results = []
                sample_inserted = 0
                sample_head_only = 0
                sample_tail_only = 0
                sample_not_found = 0
                
                for triplet in triplets:
                    verification_result = self.verify_triplet_insertion(text, triplet)
                    sample_results.append(verification_result)
                    
                    total_triplets += 1
                    
                    if verification_result["is_inserted"]:
                        inserted_triplets += 1
                        sample_inserted += 1
                    elif verification_result["head_found"] and not verification_result["tail_found"]:
                        head_only_found += 1
                        sample_head_only += 1
                    elif verification_result["tail_found"] and not verification_result["head_found"]:
                        tail_only_found += 1
                        sample_tail_only += 1
                    else:
                        not_found += 1
                        sample_not_found += 1
                
                # Calculate insertion rate for this sample
                insertion_rate = sample_inserted / len(triplets) if triplets else 0.0
                sample_insertion_rates.append(insertion_rate)
                
                # Store result for this sample
                sample_result = {
                    "sample_id": line_num,
                    "text_type": text_type,
                    "total_triplets": len(triplets),
                    "inserted_triplets": sample_inserted,
                    "head_only_found": sample_head_only,
                    "tail_only_found": sample_tail_only,
                    "not_found": sample_not_found,
                    "insertion_rate": insertion_rate,
                    "triplet_verifications": sample_results
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
        
        # Overall insertion rate
        overall_insertion_rate = inserted_triplets / total_triplets if total_triplets > 0 else 0.0
        avg_sample_insertion_rate = np.mean(sample_insertion_rates) if sample_insertion_rates else 0.0
        
        # Summary
        summary = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "text_type": "paraphrased_watermarked" if use_paraphrased_text else "watermarked",
            "aggregate_statistics": {
                "total_triplets": total_triplets,
                "inserted_triplets": inserted_triplets,
                "head_only_found": head_only_found,
                "tail_only_found": tail_only_found,
                "not_found": not_found,
                "overall_insertion_rate": float(overall_insertion_rate),
                "avg_sample_insertion_rate": float(avg_sample_insertion_rate)
            },
            "entity_form_preservation": {
                "head_entity_preserved": (inserted_triplets + head_only_found) / total_triplets if total_triplets > 0 else 0.0,
                "tail_entity_preserved": (inserted_triplets + tail_only_found) / total_triplets if total_triplets > 0 else 0.0,
                "both_entities_preserved": overall_insertion_rate
            }
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Triplet Insertion Verification Results")
        print(f"{'='*80}")
        print(f"Total samples: {total_samples}")
        print(f"Valid samples: {valid_samples}")
        print(f"\nAggregate Statistics:")
        print(f"  Total triplets: {total_triplets}")
        print(f"  Inserted triplets (both head & tail found): {inserted_triplets} ({overall_insertion_rate*100:.2f}%)")
        print(f"  Head only found: {head_only_found}")
        print(f"  Tail only found: {tail_only_found}")
        print(f"  Not found: {not_found}")
        print(f"\nInsertion Rates:")
        print(f"  Overall insertion rate: {overall_insertion_rate:.4f} ({overall_insertion_rate*100:.2f}%)")
        print(f"  Average per-sample insertion rate: {avg_sample_insertion_rate:.4f} ({avg_sample_insertion_rate*100:.2f}%)")
        print(f"\nEntity Form Preservation:")
        print(f"  Head entity preserved: {summary['entity_form_preservation']['head_entity_preserved']*100:.2f}%")
        print(f"  Tail entity preserved: {summary['entity_form_preservation']['tail_entity_preserved']*100:.2f}%")
        print(f"  Both entities preserved: {summary['entity_form_preservation']['both_entities_preserved']*100:.2f}%")
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
    parser = argparse.ArgumentParser(description="Triplet Insertion Verification: Watermarked Text에 실제로 삽입된 Triplet과 Entity 형태 유지 여부 확인")
    parser.add_argument("--jsonl_file", type=str, required=True, 
                       help="Path to JSONL file with watermarked examples")
    parser.add_argument("--use_paraphrased_text", action="store_true", default=False,
                       help="Use paraphrased_watermarked_text with selected_paraphrased_triplets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to verify (default: all)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save detailed results as JSON file")
    parser.add_argument("--device_id", type=int, default=None, 
                       help="CUDA device ID (not used, kept for compatibility)")
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = TripletInsertionVerifier(device_id=args.device_id)
    
    # Verify triplet insertions
    results = verifier.verify_all_triplets(
        args.jsonl_file,
        use_paraphrased_text=args.use_paraphrased_text,
        max_samples=args.max_samples,
        save_results=args.save_results
    )
    
    if "error" not in results:
        print("\n✅ Triplet insertion verification completed successfully!")


if __name__ == "__main__":
    main()


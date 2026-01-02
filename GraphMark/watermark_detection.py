import os
import json
import numpy as np
import spacy
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime

class WatermarkDetector:
    def __init__(self, device_id=None):
        """
        Initialize watermark detector with Entity-Pair Detection Strategy
        
        Args:
            device_id: Not used (kept for compatibility)
        """
        # Load spacy for sentence tokenization
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load entity data
        self.entity_dict = {}
        self._load_kg_data()
    
    def _load_kg_data(self):
        """Load entity dictionary from KG files"""
        kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
        
        # Load entities
        entity_file = f"{kg_root_path}/entities.txt"
        if os.path.exists(entity_file):
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    entity_id = parts[0]
                    entity_names = parts[1:] if len(parts) > 1 else [entity_id]
                    self.entity_dict[entity_id] = {"entity": entity_names}
            print(f"Loaded {len(self.entity_dict)} entities")
    

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def _select_best_name(self, item_id: str, data_dict: Dict, item_type: str = "entity") -> str:
        """Select best name for entity or relation ID"""
        try:
            if item_id not in data_dict:
                return str(item_id)
            
            item_data = data_dict[item_id]
            if isinstance(item_data, dict):
                key = item_type if item_type in item_data else "name"
                names = item_data.get(key, [str(item_id)])
            else:
                names = item_data if isinstance(item_data, list) else [str(item_id)]
            
            if not names:
                return str(item_id)
            
            # Filter English names
            english_names = [name for name in names if isinstance(name, str) and self._is_english_text(name)]
            candidate_names = english_names if english_names else [name for name in names if isinstance(name, str)]
            
            if not candidate_names:
                return str(item_id)
            
            # Return the first candidate
            return candidate_names[0]
            
        except Exception as e:
            print(f"Error selecting name for {item_id}: {e}")
            return str(item_id)
    
    def _is_english_text(self, text: str) -> bool:
        """Check if text is English"""
        if not text or not isinstance(text, str):
            return False
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return False
        english_ratio = english_chars / total_chars
        return english_ratio >= 0.8
    
    


    
    def _get_all_entity_names(self, entity_id: str) -> List[str]:
        """Get all entity name candidates (all aliases) for an entity ID"""
        names = []
        if entity_id in self.entity_dict:
            entity_data = self.entity_dict[entity_id]
            if isinstance(entity_data, dict):
                names.extend(entity_data.get("entity", []))
            elif isinstance(entity_data, list):
                names.extend(entity_data)
        
        # If no names found, use entity_id as fallback
        if not names:
            names = [str(entity_id)]
        
        # Filter to only English strings and return
        english_names = [name for name in names if isinstance(name, str) and self._is_english_text(name)]
        return english_names if english_names else [name for name in names if isinstance(name, str)]
    
    def _sentence_contains_entity(self, sentence: str, entity_names: List[str]) -> bool:
        """
        Check if sentence contains any of the entity names (case-insensitive)
        Uses flexible matching: both word-level matching and substring matching
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
            
            # Method 1: Exact substring matching
            if entity_lower in sentence_lower:
                return True
            
            # Method 2: Word-level matching
            # Split entity name into words and check if all words appear in sentence
            entity_words = entity_lower.split()
            if len(entity_words) > 1:
                # Multi-word entity: check if all words appear in order
                if self._words_appear_in_order(sentence_lower, entity_words):
                    return True
            elif len(entity_words) == 1:
                # Single word entity: check if word is in sentence (exclude very short words)
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
    
    def detect_watermark(self, text: str, selected_triplets: List[List[str]], 
                         threshold: float = 0.6, min_detected_count: int = 1,
                         print_case_study: bool = False) -> Dict:
        """
        Entity-Pair Detection Strategy: Check if both head and tail entities appear in the same sentence
        using exact string matching (case-insensitive)
        
        Args:
            text: The text to check
            selected_triplets: Actually used triplets for watermarking (only these are checked)
            threshold: Not used (kept for compatibility)
            min_detected_count: Minimum number of sentences containing both head and tail entities
            print_case_study: If True, print detailed test cases for matching sentences
        
        Returns:
            Dictionary with detection results
        """
        if not selected_triplets:
            return {
                "is_watermarked": False,
                "detected_selected_triplets": [],
                "total_selected_triplets": 0,
                "total_sentences": 0,
                "matching_sentences": 0
            }
        
        # Step 1: Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return {
                "is_watermarked": False,
                "detected_selected_triplets": [],
                "total_selected_triplets": len(selected_triplets),
                "total_sentences": 0,
                "matching_sentences": 0
            }
        
        # Step 2: Extract all Head and Tail entity names from selected_triplets
        # Collect all unique entity IDs and their names
        head_entity_ids = set()
        tail_entity_ids = set()
        for head_id, relation_id, tail_id in selected_triplets:
            head_entity_ids.add(head_id)
            tail_entity_ids.add(tail_id)
        
        # Get all name candidates for each entity (all aliases)
        head_entity_names = {}  # entity_id -> List[str]
        tail_entity_names = {}  # entity_id -> List[str]
        
        for entity_id in head_entity_ids:
            head_entity_names[entity_id] = self._get_all_entity_names(entity_id)
        
        for entity_id in tail_entity_ids:
            tail_entity_names[entity_id] = self._get_all_entity_names(entity_id)
        
        # Step 3: Find sentences containing both head and tail entities
        matching_sentences = []  # List of (sentence_idx, triplet) tuples
        detected_triplets = set()
        matching_details = []  # Store detailed matching information for printing
        
        for sent_idx, sentence in enumerate(sentences):
            for head_id, relation_id, tail_id in selected_triplets:
                triplet_tuple = (head_id, relation_id, tail_id)
                
                # Get all name candidates for head and tail
                head_names = head_entity_names.get(head_id, [])
                tail_names = tail_entity_names.get(tail_id, [])
                
                # Check if sentence contains both head and tail entities (case-insensitive)
                head_found = self._sentence_contains_entity(sentence, head_names)
                tail_found = self._sentence_contains_entity(sentence, tail_names)
                
                if head_found and tail_found:
                    matching_sentences.append((sent_idx, triplet_tuple))
                    detected_triplets.add(triplet_tuple)
                    
                    # Find which entity names actually matched
                    matched_head_name = None
                    matched_tail_name = None
                    sentence_lower = sentence.lower()
                    
                    for name in head_names:
                        if name and isinstance(name, str) and name.lower() in sentence_lower:
                            matched_head_name = name
                            break
                    
                    for name in tail_names:
                        if name and isinstance(name, str) and name.lower() in sentence_lower:
                            matched_tail_name = name
                            break
                    
                    matching_details.append({
                        "sentence_idx": sent_idx,
                        "sentence": sentence,
                        "triplet": (head_id, relation_id, tail_id),
                        "matched_head_name": matched_head_name,
                        "matched_tail_name": matched_tail_name
                    })
        
        # Print test cases for matching sentences (if enabled)
        if print_case_study and matching_details:
            print(f"\n{'='*80}")
            print(f"Test Cases: Head & Tail Entity-Pair Detection")
            print(f"{'='*80}")
            print(f"Total Matching Sentences: {len(matching_details)}")
            print(f"{'='*80}\n")
            
            for idx, detail in enumerate(matching_details, 1):
                print(f"Test Case #{idx}:")
                print(f"  Sentence Index: {detail['sentence_idx']}")
                print(f"  Sentence: {detail['sentence']}")
                print(f"  Triplet: {detail['triplet']}")
                print(f"  Matched Head Entity: {detail['matched_head_name']} (ID: {detail['triplet'][0]})")
                print(f"  Matched Tail Entity: {detail['matched_tail_name']} (ID: {detail['triplet'][2]})")
                print()
        
        # Step 4: Determine if watermarked based on number of matching sentences
        is_watermarked = len(matching_sentences) >= min_detected_count
        
        return {
            "is_watermarked": is_watermarked,
            "detected_selected_triplets": list(detected_triplets),
            "total_selected_triplets": len(selected_triplets),
            "total_sentences": len(sentences),
            "matching_sentences": len(matching_sentences),
            "detection_rate": len(detected_triplets) / len(selected_triplets) if selected_triplets else 0.0
        }
    
    def evaluate_dataset(self, jsonl_file: str, threshold: float = 0.6, 
                         min_detected_count: int = 1, watermarked_only: bool = False,
                         use_paraphrased_text: bool = False, 
                         max_watermarked_samples: int = 200,
                         max_original_samples: int = 200,
                         save_results: Optional[str] = None,
                         debug_false_positive_examples: int = 0,
                         debug_hits_per_example: int = 3,
                         print_case_study: bool = False):
        """
        Evaluate detection performance on a dataset
        
        Args:
            jsonl_file: Path to JSONL file with watermarked examples
            threshold: Similarity threshold for detection
            min_detected_count: Minimum number of detected selected triplets to consider as watermarked
            watermarked_only: Only evaluate watermarked examples (skip originals)
            use_paraphrased_text: If True, use paraphrased_watermarked with paraphrased_subgraph_triples and selected_paraphrased_triplets
            max_watermarked_samples: Maximum number of watermarked examples to sample (default: 200)
            max_original_samples: Maximum number of original text examples to sample (default: 200)
            save_results: Path to save results as JSONL (optional)
            print_case_study: If True, print detailed test cases for matching sentences
        
        Returns:
            Dictionary with evaluation metrics
        """
        import random
        
        results = []
        total_examples = 0
        total_watermarked_examples = 0  # Count of examples with watermark (for TPR calculation)
        total_original_examples = 0    # Count of examples with original_text (for FPR calculation)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # Counters for sampling
        watermarked_count = 0
        original_count = 0
        debug_fp_count = 0
        
        print(f"\n{'='*80}")
        print(f"Evaluating detection on {jsonl_file}")
        if use_paraphrased_text:
            print(f"Mode: Paraphrased Text Detection")
            print(f"  - Using field: 'paraphrased_watermarked' for detection")
            print(f"  - Using triplets: 'selected_paraphrased_triplets' for detection")
        else:
            print(f"Mode: Watermarked Text Detection")
            print(f"  - Using field: 'watermarked_text' for detection")
            print(f"  - Using triplets: 'selected_triplets' for detection")
        print(f"Threshold: {threshold}")
        print(f"Min Detected Count: {min_detected_count}")
        print(f"Sampling: Max {max_watermarked_samples} watermarked examples, Max {max_original_samples} original examples")
        print(f"\nNote: TPR is calculated from {'paraphrased' if use_paraphrased_text else 'watermarked'} text, FPR is calculated from original text")
        print(f"{'='*80}\n")
        
        # First pass: collect all examples with unique identifiers
        # Use a stable identifier to avoid duplicates
        all_examples_dict = {}  # key: unique_id, value: example_data
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Choose text and triplet fields based on mode
                    if use_paraphrased_text:
                        # Use paraphrased text and paraphrased triplets
                        text_field = "paraphrased_watermarked"
                        subgraph_field = "paraphrased_subgraph_triples"
                        selected_field = "selected_paraphrased_triplets"
                    else:
                        # Use original watermarked text and triplets
                        text_field = "watermarked_text"
                        subgraph_field = "subgraph_triples"
                        selected_field = "selected_triplets"
                    
                    # Skip if watermarked_only and no watermark info
                    if watermarked_only and selected_field not in data:
                        continue
                    
                    # Extract data
                    # text_field is "paraphrased_watermarked" if use_paraphrased_text=True, else "watermarked_text"
                    text_to_detect = data.get(text_field, "")
                    original_text = data.get("original_text", "")
                    
                    if not text_to_detect:
                        continue
                    
                    # Check if this is actually watermarked
                    # A document is watermarked if it has text AND selected triplets
                    has_watermark = (
                        text_field in data and 
                        selected_field in data and 
                        len(data.get(selected_field, [])) > 0
                    )
                    
                    # Use appropriate triplets for detection
                    subgraph_triples = data.get(subgraph_field, [])
                    selected_triplets = data.get(selected_field, [])
                    
                    # Create a unique identifier for this example
                    # Use original_text and selected_triplets as key (same example = same original_text + same triplets)
                    unique_id = hash((original_text if original_text else text_to_detect, tuple(map(tuple, selected_triplets))))
                    
                    # Store example for later processing
                    example_data = {
                        "data": data,
                        "text_to_detect": text_to_detect,  # Changed from "watermarked_text" to be more accurate
                        "original_text": original_text,
                        "has_watermark": has_watermark,
                        "subgraph_triples": subgraph_triples,
                        "selected_triplets": selected_triplets,
                        "text_field": text_field,
                        "subgraph_field": subgraph_field,
                        "selected_field": selected_field
                    }
                    
                    # Store example (will overwrite if duplicate, ensuring uniqueness)
                    all_examples_dict[unique_id] = example_data
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_idx}: {e}")
                    continue
        
        # Separate into watermarked-only, original-only, and both
        watermarked_only_examples = []  # has_watermark but no original_text
        original_only_examples = []     # has original_text but no watermark (shouldn't happen, but just in case)
        both_examples = []               # has both watermarked and original
        
        for unique_id, ex_data in all_examples_dict.items():
            has_watermark = ex_data["has_watermark"]
            has_original = ex_data["original_text"] and ex_data["original_text"].strip()
            
            if has_watermark and has_original:
                both_examples.append(ex_data)
            elif has_watermark:
                watermarked_only_examples.append(ex_data)
            elif has_original:
                original_only_examples.append(ex_data)
        
        total_watermarked_available = len(watermarked_only_examples) + len(both_examples)
        total_original_available = len(original_only_examples) + len(both_examples)
        
        print(f"Found {len(watermarked_only_examples)} watermarked-only examples")
        print(f"Found {len(original_only_examples)} original-only examples")
        print(f"Found {len(both_examples)} examples with both watermarked and original")
        
        # Sample non-overlapping examples
        # Strategy: 
        # 1. Sample from both_examples first (these count for both watermarked and original)
        # 2. Then fill remaining slots from watermarked_only and original_only
        
        sampled_both = []
        sampled_watermarked_only = []
        sampled_original_only = []
        
        # Sample from both_examples (these count for both categories)
        if len(both_examples) > 0:
            # Sample up to min(max_watermarked_samples, max_original_samples) from both
            max_from_both = min(max_watermarked_samples, max_original_samples, len(both_examples))
            if max_from_both > 0:
                sampled_both = random.sample(both_examples, max_from_both)
                print(f"Sampled {len(sampled_both)} examples with both watermarked and original")
        
        # Calculate remaining slots needed
        remaining_watermarked_slots = max(0, max_watermarked_samples - len(sampled_both))
        remaining_original_slots = max(0, max_original_samples - len(sampled_both))
        
        # Sample watermarked-only examples
        if remaining_watermarked_slots > 0 and len(watermarked_only_examples) > 0:
            sample_count = min(remaining_watermarked_slots, len(watermarked_only_examples))
            sampled_watermarked_only = random.sample(watermarked_only_examples, sample_count)
            print(f"Sampled {len(sampled_watermarked_only)} watermarked-only examples")
        
        # Sample original-only examples
        if remaining_original_slots > 0 and len(original_only_examples) > 0:
            sample_count = min(remaining_original_slots, len(original_only_examples))
            sampled_original_only = random.sample(original_only_examples, sample_count)
            print(f"Sampled {len(sampled_original_only)} original-only examples")
        
        # Create processing list (no duplicates guaranteed)
        all_examples_to_process = []
        
        # Add both examples (tagged as "both")
        for ex in sampled_both:
            all_examples_to_process.append(("both", ex, ex))
        
        # Add watermarked-only examples
        for ex in sampled_watermarked_only:
            all_examples_to_process.append(("watermarked", ex, None))
        
        # Add original-only examples
        for ex in sampled_original_only:
            all_examples_to_process.append(("original", None, ex))
        
        # Shuffle to mix watermarked and original processing
        random.shuffle(all_examples_to_process)
        
        print(f"\nFinal sample: {len(sampled_both)} both, {len(sampled_watermarked_only)} watermarked-only, {len(sampled_original_only)} original-only")
        print(f"Processing {len(all_examples_to_process)} unique examples (no overlaps)...\n")
        
        # Process examples
        for tag, watermarked_ex, original_ex in all_examples_to_process:
            # Determine which example data to use
            if tag == "both":
                ex_data = watermarked_ex
                original_text = original_ex["original_text"]
            elif tag == "watermarked":
                ex_data = watermarked_ex
                original_text = ex_data.get("original_text", "")
            else:  # original only
                ex_data = original_ex
                original_text = original_ex["original_text"]
            
            data = ex_data["data"]
            # Get text to detect (could be watermarked_text or paraphrased_watermarked depending on mode)
            text_to_detect = ex_data.get("text_to_detect", ex_data.get("watermarked_text", ""))  # Support both old and new field names
            has_watermark = ex_data["has_watermark"]
            subgraph_triples = ex_data["subgraph_triples"]
            selected_triplets = ex_data["selected_triplets"]
            text_field = ex_data["text_field"]
            subgraph_field = ex_data["subgraph_field"]
            selected_field = ex_data["selected_field"]
            
            # Verify we're using the correct text field
            if use_paraphrased_text:
                expected_text = data.get("paraphrased_watermarked", "")
                if not expected_text:
                    print(f"⚠️  WARNING: paraphrased_watermarked field is empty or missing!")
                elif text_to_detect != expected_text:
                    print(f"⚠️  WARNING: Text mismatch! Expected paraphrased_watermarked but got different text")
                    print(f"   Expected length: {len(expected_text)}, Got length: {len(text_to_detect)}")
                    print(f"   Expected starts with: {expected_text[:50]}...")
                    print(f"   Got starts with: {text_to_detect[:50]}...")
                    # Use the correct text
                    text_to_detect = expected_text
                # Debug: Print first example to verify
                if total_examples == 0:
                    print(f"✓ Using paraphrased text for detection (length: {len(text_to_detect)})")
                    print(f"  Text field: {text_field}, Selected field: {selected_field}")
                    print(f"  Selected triplets count: {len(selected_triplets)}")
            
            # Detect watermark on text (for TP/FN)
            # Note: text_to_detect is paraphrased_watermarked if use_paraphrased_text=True, otherwise watermarked_text
            detection_result_watermarked = None
            if tag in ("watermarked", "both") and has_watermark:
                detection_result_watermarked = self.detect_watermark(
                    text_to_detect,  # Use the appropriate text (paraphrased or original watermarked)
                    selected_triplets,  # Use selected_paraphrased_triplets if use_paraphrased_text=True
                    threshold,
                    min_detected_count,
                    print_case_study=print_case_study
                )
                
                # Count TP and FN from watermarked text (for TPR calculation)
                total_watermarked_examples += 1
                if detection_result_watermarked["is_watermarked"]:
                    true_positives += 1
                else:
                    false_negatives += 1
            
            # Detect watermark on original text (for FP/TN - FPR calculation)
            # Original text has NO watermark, so we check if selected_triplets appear in original text
            # We use the SAME threshold and min_detected_count as watermarked text for fair comparison
            detection_result_original = None
            if tag in ("original", "both") and original_text and original_text.strip() and has_watermark:
                total_original_examples += 1
                # Use same threshold and min_detected_count as watermarked text (fair comparison)
                detection_result_original = self.detect_watermark(
                    original_text,
                    selected_triplets,  # Only check selected_triplets
                    threshold,  # Same threshold as watermarked text
                    min_detected_count,  # Same min_detected_count as watermarked text
                    print_case_study=print_case_study
                )
                
                # Count FP and TN from original text (for FPR calculation)
                # FP: selected_triplets detected in original (should not happen)
                # TN: selected_triplets NOT detected in original (correct)
                if detection_result_original["is_watermarked"]:
                    false_positives += 1
                    if debug_false_positive_examples > 0 and debug_fp_count < debug_false_positive_examples:
                        debug_fp_count += 1
                        print("-" * 80)
                        print(f"[FP DEBUG] Example #{debug_fp_count}: Original text flagged as watermarked")
                        print(f"  Example Source : {ex_data.get('data', {}).get('source', 'unknown')}")
                        snippet = original_text[:200].replace('\n', ' ')
                        ellipsis = '...' if len(original_text) > 200 else ''
                        print(f"  Text Snippet   : {snippet}{ellipsis}")
                        detected_triplets = detection_result_original.get("detected_selected_triplets", [])
                        if not detected_triplets:
                            print("  (No detected triplets)")
                        else:
                            print(f"  Detected {len(detected_triplets)} triplets:")
                            for idx, triplet in enumerate(detected_triplets[:debug_hits_per_example], 1):
                                head_id, relation_id, tail_id = triplet
                                head_name = self._select_best_name(head_id, self.entity_dict, "entity")
                                tail_name = self._select_best_name(tail_id, self.entity_dict, "entity")
                                print(f"    [{idx}] {head_name} ... {tail_name}")
                                print(f"        triplet  : {triplet}")
                else:
                    true_negatives += 1
            
            results.append({
                "watermarked": detection_result_watermarked,
                "original": detection_result_original
            })
            total_examples += 1
            
            # Print progress
            if total_examples % 50 == 0:
                print(f"Processed {total_examples} examples... (TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives})")
        
        # Calculate metrics
        tp = true_positives
        fp = false_positives
        fn = false_negatives
        tn = true_negatives
        
        # TPR is calculated from watermarked examples
        total_positive = total_watermarked_examples  # Examples with watermark
        # FPR is calculated from original (unwatermarked) examples
        total_negative = total_original_examples     # Examples without watermark (original text or no watermark)
        
        tpr = tp / total_positive if total_positive > 0 else 0.0
        fpr = fp / total_negative if total_negative > 0 else 0.0
        fnr = fn / total_positive if total_positive > 0 else 0.0
        tnr = tn / total_negative if total_negative > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Prepare full metrics with parameters
        metrics = {
            "experiment_type": "evaluate_dataset",
            "timestamp": datetime.now().isoformat(),
            "input_file": jsonl_file,
            "parameters": {
                "threshold": threshold,
                "min_detected_count": min_detected_count,
                "watermarked_only": watermarked_only,
                "use_paraphrased_text": use_paraphrased_text
            },
            "results": {
                "total_examples": total_examples,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_positive": total_positive,
                "total_negative": total_negative,
                "tpr": float(tpr),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "tnr": float(tnr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy)
            }
        }
        
        # Save results if path provided
        if save_results:
            self._save_results_to_jsonl(save_results, metrics)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"Evaluation Results (Threshold: {threshold}, Min Detected Count: {min_detected_count})")
        print(f"{'='*80}")
        print(f"Total Examples Processed: {total_examples}")
        print(f"Watermarked Examples (for TPR): {total_watermarked_examples}")
        print(f"Unwatermarked Examples (for FPR): {total_original_examples}")
        if total_original_examples == 0:
            print(f"WARNING: No original_text found in dataset. FPR cannot be accurately calculated.")
        print(f"True Positives: {tp} / False Negatives: {fn} (Watermarked Examples: {total_positive})")
        print(f"False Positives: {fp} / True Negatives: {tn} (Unwatermarked Examples: {total_negative})")
        print(f"\nMetrics:")
        print(f"  TPR (Recall):  {tpr:.4f}")
        print(f"  FPR:           {fpr:.4f}")
        print(f"  FNR:           {fnr:.4f}")
        print(f"  TNR:           {tnr:.4f}")
        print(f"  Precision:     {precision:.4f}")
        print(f"  F1-Score:      {f1:.4f}")
        print(f"  Accuracy:      {accuracy:.4f}")
        if save_results:
            print(f"\nResults saved to: {save_results}")
        print(f"{'='*80}\n")
        
        return metrics
    
    def _save_results_to_jsonl(self, filepath: str, result_dict: Dict):
        """Save experiment results to JSONL file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # Append to file (JSONL format)
            with open(filepath, 'a', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to save results to {filepath}: {e}")
    
    def find_threshold_for_fpr(self, jsonl_file: str, target_fpr: float = 0.01,
                               min_detected_count: int = 1, 
                               search_min_count: bool = False,
                               use_paraphrased_text: bool = False,
                               save_results: Optional[str] = None) -> Tuple[float, float, int]:
        """
        Find threshold and min_count that achieves target FPR (TPR@1%FPR)
        
        Args:
            jsonl_file: Path to JSONL file
            target_fpr: Target FPR (e.g., 0.01 for 1%)
            min_detected_count: Fixed min_detected_count if search_min_count=False
            search_min_count: If True, search over both threshold and min_count
            use_paraphrased_text: If True, use paraphrased_watermarked with paraphrased triplets
            save_results: Path to save results as JSONL (optional)
        
        Returns:
            (best_threshold, best_tpr, best_min_count)
        """
        print(f"\nFinding optimal parameters for {target_fpr*100}% FPR...")
        print(f"Searching min_count: {search_min_count}\n")
        
        if search_min_count:
            # Grid search over both threshold and min_count
            thresholds = np.arange(0.1, 0.3, 0.05)
            min_counts = [1, 2, 3, 4]
            results = []
            
            for thresh in thresholds:
                for min_count in min_counts:
                    metrics = self.evaluate_dataset(
                        jsonl_file, 
                        threshold=thresh, 
                        min_detected_count=min_count,
                        use_paraphrased_text=use_paraphrased_text,
                        max_watermarked_samples=200,  # Use default for threshold search
                        max_original_samples=200,  # Use default for threshold search
                        save_results=None  # Don't save intermediate results
                    )
                    results.append((thresh, min_count, metrics["results"]["fpr"], metrics["results"]["tpr"]))
                    print(f"Threshold: {thresh:.2f}, Min Count: {min_count} -> FPR: {metrics['results']['fpr']:.4f}, TPR: {metrics['results']['tpr']:.4f}")
        else:
            # Only search over threshold
            thresholds = np.arange(0.3, 1.0, 0.05)
            results = []
            
            for thresh in thresholds:
                metrics = self.evaluate_dataset(
                    jsonl_file, 
                    threshold=thresh, 
                    min_detected_count=min_detected_count,
                    use_paraphrased_text=use_paraphrased_text,
                    max_watermarked_samples=200,  # Use default for threshold search
                    max_original_samples=200,  # Use default for threshold search
                    save_results=None  # Don't save intermediate results
                )
                results.append((thresh, min_detected_count, metrics["results"]["fpr"], metrics["results"]["tpr"]))
                print(f"Threshold: {thresh:.2f}, Min Count: {min_detected_count} -> FPR: {metrics['results']['fpr']:.4f}, TPR: {metrics['results']['tpr']:.4f}")
        
        # Find parameters closest to target FPR
        best_threshold = thresholds[0] if not search_min_count else 0.5
        best_min_count = min_detected_count
        best_tpr = 0.0
        closest_fpr_diff = float('inf')
        
        for thresh, min_cnt, fpr, tpr in results:
            fpr_diff = abs(fpr - target_fpr)
            if fpr_diff < closest_fpr_diff:
                closest_fpr_diff = fpr_diff
                best_threshold = thresh
                best_min_count = min_cnt
                best_tpr = tpr
        
        # Prepare results
        actual_fpr = closest_fpr_diff + target_fpr
        
        # Save all search results if path provided
        if save_results:
            search_results_list = []
            for thresh, min_cnt, fpr, tpr in results:
                search_result = {
                    "experiment_type": "threshold_search",
                    "timestamp": datetime.now().isoformat(),
                    "input_file": jsonl_file,
                    "search_config": {
                        "target_fpr": target_fpr,
                        "search_min_count": search_min_count,
                        "threshold": float(thresh),
                        "min_detected_count": min_cnt,
                        "use_paraphrased_text": use_paraphrased_text
                    },
                    "results": {
                        "fpr": float(fpr),
                        "tpr": float(tpr),
                        "is_best": (thresh == best_threshold and min_cnt == best_min_count)
                    }
                }
                search_results_list.append(search_result)
            
            # Save all search results
            for result in search_results_list:
                self._save_results_to_jsonl(save_results, result)
            
            # Save best result summary
            best_result = {
                "experiment_type": "best_parameters",
                "timestamp": datetime.now().isoformat(),
                "input_file": jsonl_file,
                "search_config": {
                    "target_fpr": target_fpr,
                    "search_min_count": search_min_count
                },
                "best_parameters": {
                    "threshold": float(best_threshold),
                    "min_detected_count": best_min_count,
                    "actual_fpr": float(actual_fpr),
                    "tpr_at_target_fpr": float(best_tpr),
                    "fpr_diff": float(closest_fpr_diff)
                }
            }
            self._save_results_to_jsonl(save_results, best_result)
        
        print(f"\n{'='*80}")
        print(f"Best parameters:")
        print(f"  Threshold: {best_threshold:.2f}")
        print(f"  Min Detected Count: {best_min_count}")
        print(f"  Actual FPR: {actual_fpr:.4f}")
        print(f"  TPR@{target_fpr*100:.0f}%FPR: {best_tpr:.4f}")
        if save_results:
            print(f"\nResults saved to: {save_results}")
        print(f"{'='*80}\n")
        
        return best_threshold, best_tpr, best_min_count


def main():
    parser = argparse.ArgumentParser(description="Watermark Detection")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--min_detected_count", type=int, default=1, 
                       help="Minimum number of detected selected triplets to consider as watermarked")
    parser.add_argument("--target_fpr", type=float, default=0.01, help="Target FPR for threshold finding")
    parser.add_argument("--find_threshold", action="store_true", help="Find threshold for target FPR")
    parser.add_argument("--search_min_count", action="store_true", 
                       help="Search over both threshold and min_count (only works with --find_threshold)")
    parser.add_argument("--device_id", type=int, default=None, help="CUDA device ID (not used, kept for compatibility)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save experiment results as JSONL file")
    parser.add_argument("--use_paraphrased_text", action="store_true", default=False,
                       help="Use paraphrased_watermarked text with selected_paraphrased_triplets")
    parser.add_argument("--max_watermarked_samples", type=int, default=14,
                       help="Maximum number of watermarked examples to sample for evaluation (default: 200)")
    parser.add_argument("--max_original_samples", type=int, default=14,
                       help="Maximum number of original text examples to sample for evaluation (default: 200)")
    parser.add_argument("--debug_false_positive_examples", type=int, default=0,
                       help="Number of false-positive original examples to display")
    parser.add_argument("--debug_hits_per_example", type=int, default=3,
                       help="Number of detected triplets to show per debugged false-positive example")
    parser.add_argument("--print_case_study", action="store_true", default=False,
                       help="Print detailed test cases for matching sentences (Head & Tail entity-pair detection)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = WatermarkDetector(
        device_id=args.device_id
    )
    
    if args.find_threshold:
        # Find threshold for target FPR
        best_threshold, tpr, best_min_count = detector.find_threshold_for_fpr(
            args.jsonl_file, 
            args.target_fpr,
            args.min_detected_count,
            args.search_min_count,
            use_paraphrased_text=args.use_paraphrased_text,
            save_results=args.save_results
        )
        print(f"\nFinal Result: TPR@{args.target_fpr*100:.0f}%FPR = {tpr:.4f}")
        print(f"Optimal Threshold: {best_threshold:.2f}")
        print(f"Optimal Min Detected Count: {best_min_count}")
    else:
        # Evaluate with given threshold and min_count
        metrics = detector.evaluate_dataset(
            args.jsonl_file, 
            threshold=args.threshold,
            min_detected_count=args.min_detected_count,
            use_paraphrased_text=args.use_paraphrased_text,
            max_watermarked_samples=args.max_watermarked_samples,
            max_original_samples=args.max_original_samples,
            save_results=args.save_results,
            debug_false_positive_examples=args.debug_false_positive_examples,
            debug_hits_per_example=args.debug_hits_per_example,
            print_case_study=args.print_case_study
        )
        print(f"\nFinal Result: TPR={metrics['results']['tpr']:.4f}, FPR={metrics['results']['fpr']:.4f}")
        print(f"Using Threshold: {args.threshold:.2f}, Min Detected Count: {args.min_detected_count}")


if __name__ == "__main__":
    main()

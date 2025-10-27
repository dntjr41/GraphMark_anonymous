import os
import json
import sys
import torch
import numpy as np
from models import subgraph_construction, LLM
from transformers import RobertaModel, RobertaTokenizer
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# Default KG paths
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"
GLOBAL_STATS_PATH = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_global_triplet_statistics_30.json"
OPENGEN_FILE_PATH = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_GraphMark_paraphrased_30.jsonl"
RATIO = 0.15

# Triplet embedding style
TRIPLET_STYLE = "CompactProjection"

# Hyperparameter combinations for detection
MAHAL_THRESHOLDS = [30, 35, 40, 45, 50]
TRIPLET_COUNT_THRESHOLDS = [2, 3, 4, 5, 6]

def load_global_statistics_to_cuda(stats_path=GLOBAL_STATS_PATH, device_id=None):
    """Load global triplet statistics to CUDA"""
    print(f"Loading global statistics from: {stats_path}")
    
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    try:
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
        
        mu_data = stats_data['global_mahalanobis_parameters']['mu']
        sigma_data = stats_data['global_mahalanobis_parameters']['sigma']
        
        mu_array = np.array(mu_data, dtype=np.float32)
        sigma_array = np.array(sigma_data, dtype=np.float32)
        
        # Determine expected dimensions based on triplet style
        if TRIPLET_STYLE == "concatenation":
            expected_dim = 2304
        else:
            expected_dim = 768
        
        print(f"Triplet style: {TRIPLET_STYLE}, Expected dimension: {expected_dim}")
        
        # Adjust dimensions if needed
        if mu_array.shape != (expected_dim,):
            if len(mu_array) > expected_dim:
                mu_array = mu_array[:expected_dim]
            else:
                mu_array = np.pad(mu_array, (0, expected_dim - len(mu_array)), 'constant')
        
        if sigma_array.shape != (expected_dim, expected_dim):
            if len(sigma_array.shape) == 1:
                total_elements = len(sigma_array)
                side_length = int(np.sqrt(total_elements))
                if side_length * side_length == total_elements:
                    sigma_array = sigma_array.reshape(side_length, side_length)
                else:
                    sigma_array = np.eye(expected_dim, dtype=np.float32)
            
            if sigma_array.shape != (expected_dim, expected_dim):
                target_sigma = np.zeros((expected_dim, expected_dim), dtype=np.float32)
                target_sigma[:sigma_array.shape[0], :sigma_array.shape[1]] = sigma_array
                sigma_array = target_sigma
        
        mu_tensor = torch.from_numpy(mu_array).to(device)
        sigma_tensor = torch.from_numpy(sigma_array).to(device)
        
        return mu_tensor, sigma_tensor
        
    except Exception as e:
        print(f"Error loading global statistics: {e}")
        return None, None

def load_opengen_data_enhanced(file_path=OPENGEN_FILE_PATH, start_line=0, end_line=None):
    """Load enhanced opengen data with all text types"""
    print(f"Loading enhanced opengen data from: {file_path}")
    
    data = []
    try:
        # Fix encoding issue by explicitly using UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if end_line is not None and i >= end_line:
                    break
                
                try:
                    line_data = json.loads(line.strip())
                    required_fields = ['original_text', 'watermarked_text', 'paraphrased_watermarked']
                    if all(field in line_data for field in required_fields):
                        data.append({
                            'line_number': i,
                            'original_text': line_data['original_text'],
                            'watermarked_text': line_data['watermarked_text'],
                            'paraphrased_watermarked': line_data['paraphrased_watermarked'],
                            'subgraph_triples': line_data.get('subgraph_triples', []),
                            'paraphrased_subgraph_triples': line_data.get('paraphrased_subgraph_triples', [])
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i}: {e}")
                    continue
        
        print(f"Successfully loaded {len(data)} lines with enhanced text types")
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try alternative encoding if UTF-8 fails
        try:
            print("Trying with latin-1 encoding...")
            with open(file_path, 'r', encoding='latin-1') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if end_line is not None and i >= end_line:
                        break
                    
                    try:
                        line_data = json.loads(line.strip())
                        required_fields = ['original_text', 'watermarked_text', 'paraphrased_watermarked']
                        if all(field in line_data for field in required_fields):
                            data.append({
                                'line_number': i,
                                'original_text': line_data['original_text'],
                                'watermarked_text': line_data['watermarked_text'],
                                'paraphrased_watermarked': line_data['paraphrased_watermarked'],
                                'subgraph_triples': line_data.get('subgraph_triples', []),
                                'paraphrased_subgraph_triples': line_data.get('paraphrased_subgraph_triples', [])
                            })
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue
            
            print(f"Successfully loaded {len(data)} lines with latin-1 encoding")
            return data
            
        except Exception as e2:
            print(f"Both UTF-8 and latin-1 encoding failed: {e2}")
            return []

def create_triplet_embedding_compact_projection(h_emb, r_emb, t_emb, device):
    """Create triplet embedding using CompactProjection"""
    concatenated_embedding = np.concatenate([h_emb, r_emb, t_emb])
    chunk_size = 768
    triplet_embedding = np.zeros(768)
    
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = concatenated_embedding[start_idx:end_idx]
        triplet_embedding += chunk / 3.0
    
    return torch.tensor(triplet_embedding, dtype=torch.float32).to(device)

def create_triplet_embedding(head_emb, relation_emb, tail_emb, device, style="CompactProjection"):
    """Create triplet embedding using specified style"""
    if style == "CompactProjection":
        return create_triplet_embedding_compact_projection(head_emb, relation_emb, tail_emb, device)
    else:
        print(f"Warning: Unknown triplet style '{style}', using CompactProjection as default")
        return create_triplet_embedding_compact_projection(head_emb, relation_emb, tail_emb, device)

def calculate_mahalanobis_distance(embedding, mu_tensor, sigma_tensor):
    """Calculate Mahalanobis Distance"""
    try:
        embedding = embedding.to(mu_tensor.device)
        diff = embedding - mu_tensor
        
        try:
            sigma_inv_diff = torch.linalg.solve(sigma_tensor, diff)
        except:
            sigma_reg = sigma_tensor + torch.eye(sigma_tensor.shape[0], device=sigma_tensor.device) * 1e-6
            sigma_inv_diff = torch.linalg.solve(sigma_reg, diff)
        
        mahal_dist_squared = torch.dot(diff, sigma_inv_diff)
        mahal_dist = torch.sqrt(mahal_dist_squared)
        
        return mahal_dist.item()
        
    except Exception as e:
        print(f"Error calculating Mahalanobis Distance: {e}")
        return float('inf')

def calculate_tpr_at_fpr(fpr, tpr, target_fpr=0.01):
    """Calculate True Positive Rate at a specific False Positive Rate"""
    try:
        if all(f == 0 for f in fpr):  # Handle case where all FPRs are 0
            return {
                'target_fpr': target_fpr,
                'achieved_fpr': 0.0,
                'tpr_at_target_fpr': max(tpr) if tpr else 0.0,
                'closest_idx': np.argmax(tpr) if tpr else 0
            }
        
        # Find the closest FPR value to target
        fpr_diff = np.abs(np.array(fpr) - target_fpr)
        closest_idx = np.argmin(fpr_diff)
        closest_fpr = fpr[closest_idx]
        tpr_at_target = tpr[closest_idx]
        
        # If we don't have exact target FPR, interpolate
        if closest_fpr != target_fpr:
            # Find points around target FPR for interpolation
            lower_idx = None
            upper_idx = None
            
            for i in range(len(fpr)):
                if fpr[i] <= target_fpr:
                    lower_idx = i
                else:
                    upper_idx = i
                    break
            
            if lower_idx is not None and upper_idx is not None:
                # Linear interpolation
                fpr_lower, fpr_upper = fpr[lower_idx], fpr[upper_idx]
                tpr_lower, tpr_upper = tpr[lower_idx], tpr[upper_idx]
                
                # Interpolate TPR
                tpr_at_target = tpr_lower + (tpr_upper - tpr_lower) * (target_fpr - fpr_lower) / (fpr_upper - fpr_lower)
                closest_fpr = target_fpr
        
        return {
            'target_fpr': target_fpr,
            'achieved_fpr': closest_fpr,
            'tpr_at_target_fpr': tpr_at_target,
            'closest_idx': closest_idx
        }
        
    except Exception as e:
        print(f"Error calculating TPR at {target_fpr*100}% FPR: {e}")
        return {'target_fpr': target_fpr, 'achieved_fpr': 0.0, 'tpr_at_target_fpr': 0.0, 'closest_idx': 0}

def detect_watermark_in_document(sentence_results, mu_tensor, sigma_tensor, kepler_model, mahal_threshold=30.0, triplet_count_threshold=3):
    """Detect if a document contains watermarks based on Mahalanobis Distance"""
    low_mahal_triplets = set()  # Use set to track unique triplets
    low_mahal_details = []
    
    for sentence_idx, sentence_result in enumerate(sentence_results):
        top_1_match = sentence_result['top_1_match']
        
        if top_1_match:
            triplet_embedding = top_1_match['triplet_embedding']
            triplet_text = top_1_match['triplet_text']
            mahal_distance = calculate_mahalanobis_distance(triplet_embedding, mu_tensor, sigma_tensor)
            
            if mahal_distance <= mahal_threshold:
                # Add triplet text to set to ensure uniqueness
                low_mahal_triplets.add(triplet_text)
                low_mahal_details.append({
                    'sentence_idx': sentence_idx,
                    'sentence_text': sentence_result['sentence_text'][:100] + "..." if len(sentence_result['sentence_text']) > 100 else sentence_result['sentence_text'],
                    'triplet_text': triplet_text,
                    'similarity_score': top_1_match['similarity_score'],
                    'mahalanobis_distance': mahal_distance
                })
    
    # Count unique triplets instead of total occurrences
    unique_low_mahal_count = len(low_mahal_triplets)
    is_watermarked = unique_low_mahal_count >= triplet_count_threshold
    
    if triplet_count_threshold > 0:
        confidence_score = min(1.0, unique_low_mahal_count / triplet_count_threshold)
    else:
        confidence_score = 1.0 if unique_low_mahal_count > 0 else 0.0
    
    total_sentences = len(sentence_results)
    watermark_ratio = unique_low_mahal_count / total_sentences if total_sentences > 0 else 0.0

    return {
        'is_watermarked': is_watermarked,
        'confidence_score': confidence_score,
        'low_mahal_count': unique_low_mahal_count,  # Now represents unique count
        'unique_low_mahal_count': unique_low_mahal_count,  # Explicit unique count
        'total_low_mahal_occurrences': len(low_mahal_details),  # Total occurrences across sentences
        'triplet_count_threshold': triplet_count_threshold,
        'total_sentences': total_sentences,
        'watermark_ratio': watermark_ratio,
        'low_mahal_details': low_mahal_details,
        'mahal_threshold': mahal_threshold,
        'unique_low_mahal_triplets': list(low_mahal_triplets)  # List of unique triplet texts
    }

def sentence_triplet_matching(watermarked_text, subgraph_triplets, sentence_model, kepler_model, mu_tensor, sigma_tensor):
    """Perform sentence-triplet matching using RoBERTa and KEPLER embeddings"""
    import re
    sentences = re.split(r'[.!?]+', watermarked_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentence_results = []
    
    for sentence_idx, sentence in enumerate(sentences):
        if len(sentence) < 10:
            continue
            
        # Get sentence embeddings using RoBERTa
        inputs = sentence_model.tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(mu_tensor.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentence_model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
        summed_embeddings = torch.sum(masked_embeddings, dim=1)
        token_counts = torch.sum(attention_mask, dim=1, keepdim=True)
        sentence_embedding = summed_embeddings / token_counts
        sentence_embedding = sentence_embedding.squeeze(0)
        
        # Ensure correct dimension
        target_dim = 768 if TRIPLET_STYLE != "concatenation" else 2304
        if sentence_embedding.shape[0] != target_dim:
            if sentence_embedding.shape[0] > target_dim:
                sentence_embedding = sentence_embedding[:target_dim]
            else:
                sentence_embedding = F.pad(sentence_embedding, (0, target_dim - sentence_embedding.shape[0]))
        
        sentence_embedding = sentence_embedding.to(mu_tensor.device)
        
        # Get triplet embeddings using KEPLER
        triplet_embeddings = []
        triplet_texts = []
        
        for i, triplet in enumerate(subgraph_triplets):
            if len(triplet) >= 3:
                h, r, t = triplet[0], triplet[1], triplet[2]
                triplet_text = f"{h} {r} {t}"
                triplet_texts.append(triplet_text)
                
                try:
                    h_emb = kepler_model.get_entity_embedding_by_id(h)
                    
                    try:
                        if r.startswith('P'):
                            relation_idx = int(r[1:])
                            if hasattr(kepler_model, 'relation_embeddings') and relation_idx < len(kepler_model.relation_embeddings):
                                r_emb = kepler_model.relation_embeddings[relation_idx]
                            else:
                                r_emb = None
                        else:
                            r_emb = None
                    except:
                        r_emb = None
                    
                    t_emb = kepler_model.get_entity_embedding_by_id(t)
                    
                    if h_emb is not None and r_emb is not None and t_emb is not None:
                        triplet_emb = create_triplet_embedding(h_emb, r_emb, t_emb, mu_tensor.device, TRIPLET_STYLE)
                        triplet_embeddings.append(triplet_emb)
                    else:
                        zero_dim = 768 if TRIPLET_STYLE != "concatenation" else 2304
                        triplet_emb = torch.zeros(zero_dim, dtype=torch.float32).to(mu_tensor.device)
                        triplet_embeddings.append(triplet_emb)
                except Exception as e:
                    zero_dim = 768 if TRIPLET_STYLE != "concatenation" else 2304
                    triplet_emb = torch.zeros(zero_dim, dtype=torch.float32).to(mu_tensor.device)
                    triplet_embeddings.append(triplet_emb)
        
        # Calculate cosine similarities and get Top-1
        similarities = []
        
        for i, triplet_emb in enumerate(triplet_embeddings):
            cos_sim = F.cosine_similarity(sentence_embedding.unsqueeze(0), triplet_emb.unsqueeze(0), dim=1)
            similarities.append({
                'triplet_index': i,
                'triplet_text': triplet_texts[i] if i < len(triplet_texts) else f"triplet_{i}",
                'similarity_score': cos_sim.item(),
                'triplet_embedding': triplet_emb
            })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_1_match = similarities[0] if similarities else None
        
        sentence_results.append({
            'sentence_idx': sentence_idx,
            'sentence_text': sentence,
            'sentence_embedding': sentence_embedding,
            'top_1_match': top_1_match,
            'all_similarities': similarities
        })

    return {
        'sentence_results': sentence_results,
        'total_sentences': len(sentence_results),
        'successful_matches': len([r for r in sentence_results if r['top_1_match']])
    }

def calculate_tpr_at_fpr(fpr, tpr, target_fpr=0.01):
    """Calculate True Positive Rate at a specific False Positive Rate"""
    try:
        if all(f == 0 for f in fpr):  # Handle case where all FPRs are 0
            return {
                'target_fpr': target_fpr,
                'achieved_fpr': 0.0,
                'tpr_at_target_fpr': max(tpr) if tpr else 0.0,
                'closest_idx': np.argmax(tpr) if tpr else 0
            }
        
        # Find the closest FPR value to target
        fpr_diff = np.abs(np.array(fpr) - target_fpr)
        closest_idx = np.argmin(fpr_diff)
        closest_fpr = fpr[closest_idx]
        tpr_at_target = tpr[closest_idx]
        
        # If we don't have exact target FPR, interpolate
        if closest_fpr != target_fpr:
            # Find points around target FPR for interpolation
            lower_idx = None
            upper_idx = None
            
            for i in range(len(fpr)):
                if fpr[i] <= target_fpr:
                    lower_idx = i
                else:
                    upper_idx = i
                    break
            
            if lower_idx is not None and upper_idx is not None:
                # Linear interpolation
                fpr_lower, fpr_upper = fpr[lower_idx], fpr[upper_idx]
                tpr_lower, tpr_upper = tpr[lower_idx], tpr[upper_idx]
                
                # Interpolate TPR
                tpr_at_target = tpr_lower + (tpr_upper - tpr_lower) * (target_fpr - fpr_lower) / (fpr_upper - fpr_lower)
                closest_fpr = target_fpr
        
        return {
            'target_fpr': target_fpr,
            'achieved_fpr': closest_fpr,
            'tpr_at_target_fpr': tpr_at_target,
            'closest_idx': closest_idx
        }
        
    except Exception as e:
        print(f"Error calculating TPR at {target_fpr*100}% FPR: {e}")
        return {'target_fpr': target_fpr, 'achieved_fpr': 0.0, 'tpr_at_target_fpr': 0.0, 'closest_idx': 0}

def create_experiment_datasets(opengen_data, num_samples=100):
    """Create datasets for two experiments: A) Watermarked detection, B) Paraphrased detection"""
    print(f"Creating experiment datasets with {num_samples} samples each")
    
    # First 100 documents: Original Text only (Negative samples)
    original_data = opengen_data[:num_samples]
    
    # Second 100 documents: Watermarked + Paraphrased (Positive samples)
    watermarked_paraphrased_data = opengen_data[num_samples:2*num_samples]
    
    # Experiment A: Original (Negative) vs Watermarked (Positive)
    experiment_a_samples = []
    
    # Add Original texts from first 100 documents as negative samples
    for doc_item in original_data:
        if doc_item.get('subgraph_triples'):
            experiment_a_samples.append({
                'doc_index': doc_item['line_number'],
                'text_type': 'original_text',
                'text': doc_item['original_text'],
                'subgraph_triples': doc_item['subgraph_triples'],
                'label': 0,  # Negative: no watermark
                'source': 'original_text'
            })
    
    # Add Watermarked texts from second 100 documents as positive samples
    for doc_item in watermarked_paraphrased_data:
        if doc_item.get('subgraph_triples'):
            experiment_a_samples.append({
                'doc_index': doc_item['line_number'],
                'text_type': 'watermarked_text',
                'text': doc_item['watermarked_text'],
                'subgraph_triples': doc_item['subgraph_triples'],
                'label': 1,  # Positive: watermark exists
                'source': 'watermarked_text'
            })
    
    # Experiment B: Original (Negative) vs Paraphrased (Positive)
    experiment_b_samples = []
    
    # Add Original texts from first 100 documents as negative samples
    for doc_item in original_data:
        if doc_item.get('subgraph_triples'):
            experiment_b_samples.append({
                'doc_index': doc_item['line_number'],
                'text_type': 'original_text',
                'text': doc_item['original_text'],
                'subgraph_triples': doc_item['subgraph_triples'],
                'label': 0,  # Negative: no watermark
                'source': 'original_text'
            })
    
    # Add Paraphrased watermarked texts from second 100 documents as positive samples
    for doc_item in watermarked_paraphrased_data:
        if doc_item.get('paraphrased_subgraph_triples'):
            experiment_b_samples.append({
                'doc_index': doc_item['line_number'],
                'text_type': 'paraphrased_watermarked',
                'text': doc_item['paraphrased_watermarked'],
                'subgraph_triples': doc_item['paraphrased_subgraph_triples'],
                'label': 1,  # Positive: watermark exists
                'source': 'paraphrased_watermarked'
            })
    
    print(f"Experiment A (Original vs Watermarked): {len(experiment_a_samples)} samples")
    print(f"  - Original (Negative): {len([s for s in experiment_a_samples if s['label'] == 0])} samples")
    print(f"  - Watermarked (Positive): {len([s for s in experiment_a_samples if s['label'] == 1])} samples")
    
    print(f"Experiment B (Original vs Paraphrased): {len(experiment_b_samples)} samples")
    print(f"  - Original (Negative): {len([s for s in experiment_b_samples if s['label'] == 0])} samples")
    print(f"  - Paraphrased (Positive): {len([s for s in experiment_b_samples if s['label'] == 1])} samples")
    
    return experiment_a_samples, experiment_b_samples

def run_watermark_detection_experiment(experiment_samples, sentence_model, kepler_model, mu_tensor, sigma_tensor, experiment_name):
    """Run watermark detection experiment on given samples"""
    print(f"\nRunning {experiment_name}...")
    
    all_results = []
    
    for sample_idx, sample in enumerate(experiment_samples):
        print(f"Processing sample {sample_idx + 1}/{len(experiment_samples)} (Line {sample['doc_index']}, Type: {sample['text_type']}, Label: {sample['label']})")
        
        # Perform triplet matching using appropriate subgraph
        if sample['subgraph_triples']:
            matching_result = sentence_triplet_matching(
                sample['text'], sample['subgraph_triples'], 
                sentence_model, kepler_model, mu_tensor, sigma_tensor
            )
            
            # Detect watermark
            watermark_result = detect_watermark_in_document(
                matching_result['sentence_results'], mu_tensor, sigma_tensor
            )
            
            result = {
                'sample_index': sample_idx,
                'doc_index': sample['doc_index'],
                'text_type': sample['text_type'],
                'source': sample['source'],
                'ground_truth_label': sample['label'],
                'predicted_label': 1 if watermark_result['is_watermarked'] else 0,
                'confidence_score': watermark_result['confidence_score'],
                'watermark_detection': watermark_result,
                **matching_result
            }
        else:
            # No subgraph available
            result = {
                'sample_index': sample_idx,
                'doc_index': sample['doc_index'],
                'text_type': sample['text_type'],
                'source': sample['source'],
                'ground_truth_label': sample['label'],
                'predicted_label': 0,  # No watermark detected without subgraph
                'confidence_score': 0.0,
                'watermark_detection': {
                    'is_watermarked': False,
                    'confidence_score': 0.0,
                    'low_mahal_count': 0,
                    'total_sentences': 0,
                    'low_mahal_details': []
                },
                'sentence_results': [],
                'total_sentences': 0,
                'successful_matches': 0
            }
        
        all_results.append(result)
        
        # Progress output
        print(f"  ✅ Sample {sample_idx + 1} completed:")
        print(f"     - Type: {sample['text_type']}, Source: {sample['source']}")
        print(f"     - Ground Truth: {sample['label']}, Predicted: {result['predicted_label']}")
        print(f"     - Confidence: {result['confidence_score']:.3f}")
        if 'watermark_detection' in result and 'unique_low_mahal_count' in result['watermark_detection']:
            wd = result['watermark_detection']
            print(f"     - Unique low-Mahal triplets: {wd['unique_low_mahal_count']}, Total occurrences: {wd.get('total_low_mahal_occurrences', 0)}")
    
    return all_results

def evaluate_experiment_performance(experiment_results, experiment_name):
    """Evaluate performance of watermark detection experiment"""
    print(f"\nEvaluating {experiment_name}: {len(experiment_results)} samples")
    
    # Extract ground truth and predictions
    ground_truth = [result['ground_truth_label'] for result in experiment_results]
    predicted_labels = [result['predicted_label'] for result in experiment_results]
    confidence_scores = [result['confidence_score'] for result in experiment_results]
    
    # Calculate basic metrics
    try:
        precision = precision_score(ground_truth, predicted_labels, zero_division=0)
        recall = recall_score(ground_truth, predicted_labels, zero_division=0)
        f1 = f1_score(ground_truth, predicted_labels, zero_division=0)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    # Calculate confusion matrix
    try:
        cm = confusion_matrix(ground_truth, predicted_labels)
        tn, fp, fn, tp = cm.ravel()
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        tn = fp = fn = tp = 0
    
    # Calculate ROC curve and AUC
    try:
        fpr, tpr, roc_thresholds = roc_curve(ground_truth, confidence_scores)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"Error calculating ROC curve: {e}")
        fpr = tpr = roc_thresholds = []
        roc_auc = 0.0
    
    # Calculate TPR at 1% FPR
    tpr_at_1pct = calculate_tpr_at_fpr(fpr, tpr, target_fpr=0.01)
    
    # Calculate accuracy
    accuracy = (tp + tn) / len(experiment_results) if len(experiment_results) > 0 else 0.0
    
    # Analyze by source type
    source_analysis = {}
    for result in experiment_results:
        source = result['source']
        if source not in source_analysis:
            source_analysis[source] = {'total': 0, 'correct': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        
        source_analysis[source]['total'] += 1
        
        if result['ground_truth_label'] == result['predicted_label']:
            source_analysis[source]['correct'] += 1
        
        if result['ground_truth_label'] == 1 and result['predicted_label'] == 1:
            source_analysis[source]['tp'] += 1
        elif result['ground_truth_label'] == 0 and result['predicted_label'] == 0:
            source_analysis[source]['tn'] += 1
        elif result['ground_truth_label'] == 0 and result['predicted_label'] == 1:
            source_analysis[source]['fp'] += 1
        elif result['ground_truth_label'] == 1 and result['predicted_label'] == 0:
            source_analysis[source]['fn'] += 1
    
    # Calculate source-specific metrics
    for source, stats in source_analysis.items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
            if stats['tp'] + stats['fp'] > 0:
                stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'])
            else:
                stats['precision'] = 0.0
            if stats['tp'] + stats['fn'] > 0:
                stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'])
            else:
                stats['recall'] = 0.0
    
    return {
        'experiment_name': experiment_name,
        'overall_performance': {
            'total_samples': len(experiment_results),
            'positive_samples': sum(ground_truth),
            'negative_samples': len(ground_truth) - sum(ground_truth),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'tpr_at_1pct_fpr': tpr_at_1pct['tpr_at_target_fpr']
        },
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
        'ground_truth_labels': ground_truth,
        'predicted_labels': predicted_labels,
        'confidence_scores': confidence_scores,
        'tpr_at_1pct_fpr': tpr_at_1pct,
        'source_analysis': source_analysis
    }

def print_experiment_results(experiment_performance):
    """Print results for watermark detection experiment"""
    print(f"\n{'='*80}")
    print(f"{experiment_performance['experiment_name'].upper()} RESULTS")
    print(f"{'='*80}")
    
    # Overall performance
    perf = experiment_performance['overall_performance']
    print(f"\nOverall Performance:")
    print("-" * 50)
    print(f"Total samples: {perf['total_samples']}")
    print(f"Positive samples: {perf['positive_samples']}")
    print(f"Negative samples: {perf['negative_samples']}")
    print(f"Accuracy: {perf['accuracy']:.4f}")
    print(f"Precision: {perf['precision']:.4f}")
    print(f"Recall: {perf['recall']:.4f}")
    print(f"F1-Score: {perf['f1_score']:.4f}")
    print(f"ROC AUC: {perf['roc_auc']:.4f}")
    print(f"TPR at 1% FPR: {perf['tpr_at_1pct_fpr']:.4f}")
    
    # Confusion matrix
    cm = experiment_performance['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Watermarked  Non-watermarked")
    print(f"Actual Watermarked    {cm['tp']:4d}        {cm['fn']:4d}")
    print(f"Actual Non-watermarked {cm['fp']:4d}        {cm['tn']:4d}")
    
    # Source-specific analysis
    print(f"\nSource-Specific Analysis:")
    print("-" * 50)
    for source, stats in experiment_performance['source_analysis'].items():
        print(f"\n{source}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")
        print(f"  TP: {stats['tp']}, TN: {stats['tn']}, FP: {stats['fp']}, FN: {stats['fn']}")

def run_hyperparameter_grid_search(experiment_samples, sentence_model, kepler_model, mu_tensor, sigma_tensor, experiment_name):
    """Run watermark detection with different hyperparameter combinations"""
    print(f"\nRunning hyperparameter grid search for {experiment_name}")
    
    all_results = {}
    
    for mahal_threshold in MAHAL_THRESHOLDS:
        for triplet_count_threshold in TRIPLET_COUNT_THRESHOLDS:
            print(f"\nTesting: mahal_threshold={mahal_threshold}, triplet_count_threshold={triplet_count_threshold}")
            
            # Run detection with current hyperparameters
            results = []
            for sample_idx, sample in enumerate(experiment_samples):
                if sample['subgraph_triples']:
                    matching_result = sentence_triplet_matching(
                        sample['text'], sample['subgraph_triples'], 
                        sentence_model, kepler_model, mu_tensor, sigma_tensor
                    )
                    
                    # Detect watermark with current hyperparameters
                    watermark_result = detect_watermark_in_document(
                        matching_result['sentence_results'], mu_tensor, sigma_tensor, kepler_model,
                        mahal_threshold=mahal_threshold, triplet_count_threshold=triplet_count_threshold
                    )
                    
                    result = {
                        'sample_index': sample_idx,
                        'doc_index': sample['doc_index'],
                        'text_type': sample['text_type'],
                        'source': sample['source'],
                        'is_watermarked': watermark_result['is_watermarked'],
                        'confidence_score': watermark_result['confidence_score'],
                        'low_mahal_count': watermark_result['low_mahal_count'],
                        'total_sentences': watermark_result['total_sentences']
                    }
                else:
                    result = {
                        'sample_index': sample_idx,
                        'doc_index': sample['doc_index'],
                        'text_type': sample['text_type'],
                        'source': sample['source'],
                        'is_watermarked': False,
                        'confidence_score': 0.0,
                        'low_mahal_count': 0,
                        'total_sentences': 0
                    }
                
                results.append(result)
            
            # Calculate performance metrics
            total_samples = len(results)
            watermarked_samples = sum(1 for r in results if r['is_watermarked'])
            avg_confidence = np.mean([r['confidence_score'] for r in results])
            avg_low_mahal_count = np.mean([r['low_mahal_count'] for r in results])
            
            all_results[f"mahal_{mahal_threshold}_triplet_{triplet_count_threshold}"] = {
                'mahal_threshold': mahal_threshold,
                'triplet_count_threshold': triplet_count_threshold,
                'total_samples': total_samples,
                'watermarked_samples': watermarked_samples,
                'watermark_ratio': watermarked_samples / total_samples if total_samples > 0 else 0.0,
                'avg_confidence': avg_confidence,
                'avg_low_mahal_count': avg_low_mahal_count,
                'results': results
            }
            
            print(f"  Results: {watermarked_samples}/{total_samples} watermarked ({watermarked_samples/total_samples*100:.1f}%)")
    
    return all_results

def print_comparison_results(exp_a_performance, exp_b_performance):
    """Print comparison between two experiments"""
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    exp_a = exp_a_performance['overall_performance']
    exp_b = exp_b_performance['overall_performance']
    
    print(f"\nPerformance Comparison:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Exp A (Watermarked)':<20} {'Exp B (Paraphrased)':<20} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Accuracy':<20} {exp_a['accuracy']:<20.4f} {exp_b['accuracy']:<20.4f} {exp_a['accuracy'] - exp_b['accuracy']:<+15.4f}")
    print(f"{'Precision':<20} {exp_a['precision']:<20.4f} {exp_b['precision']:<20.4f} {exp_a['precision'] - exp_b['precision']:<+15.4f}")
    print(f"{'Recall':<20} {exp_a['recall']:<20.4f} {exp_b['recall']:<20.4f} {exp_a['recall'] - exp_b['recall']:<+15.4f}")
    print(f"{'F1-Score':<20} {exp_a['f1_score']:<20.4f} {exp_b['f1_score']:<20.4f} {exp_a['f1_score'] - exp_b['f1_score']:<+15.4f}")
    print(f"{'ROC AUC':<20} {exp_a['roc_auc']:<20.4f} {exp_b['roc_auc']:<20.4f} {exp_a['roc_auc'] - exp_b['roc_auc']:<+15.4f}")
    print(f"{'TPR@1%FPR':<20} {exp_a['tpr_at_1pct_fpr']:<20.4f} {exp_b['tpr_at_1pct_fpr']:<20.4f} {exp_a['tpr_at_1pct_fpr'] - exp_b['tpr_at_1pct_fpr']:<+15.4f}")
    
    print(f"\nKey Findings:")
    print("-" * 50)
    
    # Determine which experiment performs better
    if exp_a['f1_score'] > exp_b['f1_score']:
        print(f"✓ Experiment A (Watermarked Detection) performs better in F1-Score")
    else:
        print(f"✓ Experiment B (Paraphrased Detection) performs better in F1-Score")
    
    if exp_a['tpr_at_1pct_fpr'] > exp_b['tpr_at_1pct_fpr']:
        print(f"✓ Experiment A (Watermarked Detection) performs better in TPR@1%FPR")
    else:
        print(f"✓ Experiment B (Paraphrased Detection) performs better in TPR@1%FPR")
    
    # Analyze differences
    f1_diff = exp_a['f1_score'] - exp_b['f1_score']
    tpr_diff = exp_a['tpr_at_1pct_fpr'] - exp_b['tpr_at_1pct_fpr']
    
    if abs(f1_diff) < 0.05:
        print(f"✓ Both experiments show similar overall performance (F1 difference: {f1_diff:.4f})")
    elif f1_diff > 0:
        print(f"⚠ Experiment A shows {f1_diff:.4f} better F1-Score than Experiment B")
    else:
        print(f"⚠ Experiment B shows {abs(f1_diff):.4f} better F1-Score than Experiment A")
    
    if abs(tpr_diff) < 0.05:
        print(f"✓ Both experiments show similar TPR@1%FPR performance (difference: {tpr_diff:.4f})")
    elif tpr_diff > 0:
        print(f"⚠ Experiment A shows {tpr_diff:.4f} better TPR@1%FPR than Experiment B")
    else:
        print(f"⚠ Experiment B shows {abs(tpr_diff):.4f} better TPR@1%FPR than Experiment A")

if __name__ == "__main__":
    # Load global statistics to CUDA
    print("=" * 60)
    print("LOADING GLOBAL TRIPLET STATISTICS TO CUDA")
    print("=" * 60)
    
    mu_tensor, sigma_tensor = load_global_statistics_to_cuda()
    
    if mu_tensor is None or sigma_tensor is None:
        print("Failed to load global statistics. Exiting.")
        sys.exit(1)
    
    print("=" * 60)
    print("GLOBAL STATISTICS LOADED SUCCESSFULLY")
    print("=" * 60)
    print(f"Mu tensor: {mu_tensor.shape} on {mu_tensor.device}")
    print(f"Sigma tensor: {sigma_tensor.shape} on {sigma_tensor.device}")
    print("=" * 60)
    
    # Load opengen data
    print("\n" + "=" * 60)
    print("LOADING ENHANCED OPENGEN DATA")
    print("=" * 60)
    
    opengen_data = load_opengen_data_enhanced(start_line=0, end_line=200)  # Load 200 documents
    
    if not opengen_data:
        print("No opengen data loaded. Exiting.")
        sys.exit(1)
    
    print(f"Loaded {len(opengen_data)} lines of enhanced opengen data")
    
    # Display triplet embedding style information
    print(f"\nTriplet Embedding Style: {TRIPLET_STYLE}")
    if TRIPLET_STYLE == "CompactProjection":
        print("Using CompactProjection: W * [e_h; e_r; e_t] → 768 dimensions")
    
    # Initialize models
    print("\n" + "=" * 60)
    print("INITIALIZING MODELS")
    print("=" * 60)
    
    try:
        # Initialize RoBERTa model
        print("Loading RoBERTa model...")
        sentence_model = RobertaModel.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        sentence_model = sentence_model.to(mu_tensor.device)
        sentence_model.eval()
        sentence_model.tokenizer = tokenizer
        
        print("RoBERTa model loaded successfully")
        
        # Initialize KEPLER model
        print("Loading KEPLER model...")
        # First create LLM instance, then pass it to subgraph_construction
        llm_instance = LLM("llama-3-8b-inst", device_id=None)
        kepler_model = subgraph_construction(llm_instance, ratio=0.1, kg_entity_path=KG_ENTITY_PATH, 
                                           kg_relation_path=KG_RELATION_PATH, kg_triple_path=KG_TRIPLE_PATH)
        
        print("Loading KG data...")
        kepler_model.load_kg(KG_ENTITY_PATH, KG_RELATION_PATH, KG_TRIPLE_PATH)
        print("KG data loaded successfully")
        
        # Create experiment datasets
        print(f"\n" + "=" * 60)
        print("CREATING EXPERIMENT DATASETS")
        print("=" * 60)
        
        experiment_a_samples, experiment_b_samples = create_experiment_datasets(opengen_data, num_samples=100)
        
        if not experiment_a_samples or not experiment_b_samples:
            print("Failed to create experiment datasets. Exiting.")
            sys.exit(1)
        
        # Run hyperparameter grid search for Experiment A: Original vs Watermarked
        print(f"\n" + "=" * 60)
        print("EXPERIMENT A: ORIGINAL vs WATERMARKED TEXT DETECTION")
        print("=" * 60)
        
        exp_a_grid_results = run_hyperparameter_grid_search(
            experiment_a_samples, sentence_model, kepler_model, mu_tensor, sigma_tensor,
            "Original vs Watermarked Text Detection"
        )
        
        # Run hyperparameter grid search for Experiment B: Original vs Paraphrased
        print(f"\n" + "=" * 60)
        print("EXPERIMENT B: ORIGINAL vs PARAPHRASED TEXT DETECTION")
        print("=" * 60)
        
        exp_b_grid_results = run_hyperparameter_grid_search(
            experiment_b_samples, sentence_model, kepler_model, mu_tensor, sigma_tensor,
            "Original vs Paraphrased Text Detection"
        )
        
        # Print summary of results
        print(f"\n" + "=" * 60)
        print("HYPERPARAMETER GRID SEARCH SUMMARY")
        print("=" * 60)
        
        print(f"\nExperiment A (Original vs Watermarked):")
        print("-" * 50)
        for config_name, config_results in exp_a_grid_results.items():
            print(f"{config_name}: {config_results['watermarked_samples']}/{config_results['total_samples']} "
                  f"watermarked ({config_results['watermark_ratio']*100:.1f}%)")
        
        print(f"\nExperiment B (Original vs Paraphrased):")
        print("-" * 50)
        for config_name, config_results in exp_b_grid_results.items():
            print(f"{config_name}: {config_results['watermarked_samples']}/{config_results['total_samples']} "
                  f"watermarked ({config_results['watermark_ratio']*100:.1f}%)")
        
        # Save simplified results
        output_file = f"hyperparameter_grid_search_{TRIPLET_STYLE}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'experiment_a': {
                        'name': 'Original vs Watermarked Text Detection',
                        'grid_results': exp_a_grid_results
                    },
                    'experiment_b': {
                        'name': 'Original vs Paraphrased Text Detection',
                        'grid_results': exp_b_grid_results
                    },
                    'experiment_info': {
                        'num_samples_per_experiment': 100,
                        'triplet_style': TRIPLET_STYLE,
                        'mahal_thresholds': MAHAL_THRESHOLDS,
                        'triplet_count_thresholds': TRIPLET_COUNT_THRESHOLDS,
                        'experiment_type': 'hyperparameter_grid_search'
                    }
                }, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
        
    except Exception as e:
        print(f"Error processing documents: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
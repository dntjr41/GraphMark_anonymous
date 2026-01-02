import os
import json
import argparse
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from models import subgraph_construction, LLM
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Default KG paths
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"

class StatisticalTripletDetector:
    """Statistical approach using KG triplet distribution"""
    
    def __init__(self, kg_triplets=None, device='cuda:0'):
        self.device = device
        self.kg_triplets = kg_triplets or []
        self.distribution_stats = {}
        self.sentence_model = None
        self.kepler_model = None
        
    def load_models(self):
        """Load RoBERTa and KEPLER models"""
        print("Loading RoBERTa model...")
        self.sentence_model = RobertaModel.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.sentence_model = self.sentence_model.to(self.device)
        self.sentence_model.eval()
        self.sentence_model.tokenizer = tokenizer
        print("✓ RoBERTa loaded")
        
        print("Loading KEPLER model...")
        device_str = str(self.device)
        device_id = int(device_str.split(':')[1]) if ':cuda:' in device_str else 0
        llm_instance = LLM("llama-3-8b", device_id=device_id)
        self.kepler_model = subgraph_construction(
            llm_instance, ratio=0.1,
            kg_entity_path=KG_ENTITY_PATH,
            kg_relation_path=KG_RELATION_PATH,
            kg_triple_path=KG_TRIPLE_PATH
        )
        self.kepler_model.load_kg(KG_ENTITY_PATH, KG_RELATION_PATH, KG_TRIPLE_PATH)
        print("✓ KEPLER loaded")
    
    def create_triplet_embedding(self, head_emb, relation_emb, tail_emb):
        """Create triplet embedding using CompactProjection"""
        concatenated_embedding = np.concatenate([head_emb, relation_emb, tail_emb])
        chunk_size = 768
        triplet_embedding = np.zeros(768)
        
        for i in range(3):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx <= len(concatenated_embedding):
                chunk = concatenated_embedding[start_idx:end_idx]
                triplet_embedding += chunk / 3.0
        
        return triplet_embedding
    
    def get_triplet_embedding_from_ids(self, h, r, t):
        """Get triplet embedding from entity/relation IDs"""
        try:
            # Get entity embeddings
            h_emb = self.kepler_model.get_entity_embedding_by_id(h)
            
            # Get relation embedding
            if r.startswith('P'):
                relation_idx = int(r[1:])
                if hasattr(self.kepler_model, 'relation_embeddings') and relation_idx < len(self.kepler_model.relation_embeddings):
                    r_emb = self.kepler_model.relation_embeddings[relation_idx]
                else:
                    r_emb = np.zeros(768)  # Fallback
            else:
                r_emb = np.zeros(768)
            
            t_emb = self.kepler_model.get_entity_embedding_by_id(t)
            
            if h_emb is not None and r_emb is not None and t_emb is not None:
                return self.create_triplet_embedding(h_emb, r_emb, t_emb)
            else:
                return np.zeros(768)
        except Exception as e:
            print(f"Error getting triplet embedding: {e}")
            return np.zeros(768)
    
    def save_distribution(self, filepath='kg_triplet_distribution.pkl'):
        """Save distribution statistics to file"""
        if self.distribution_stats:
            with open(filepath, 'wb') as f:
                pickle.dump(self.distribution_stats, f)
            print(f"✓ Distribution saved to {filepath}")
    
    def load_distribution(self, filepath='kg_triplet_distribution.pkl'):
        """Load distribution statistics from file"""
        try:
            with open(filepath, 'rb') as f:
                self.distribution_stats = pickle.load(f)
            print(f"✓ Distribution loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"✗ Distribution file not found: {filepath}")
            return False
    
    def compute_statistical_distribution(self, triplets_list=None, sample_size=10000, save_path='kg_triplet_distribution.pkl'):
        """Compute statistical distribution of triplet embeddings in KG"""
        
        if triplets_list is None:
            # Load triplets from file
            print(f"Loading triplets from {KG_TRIPLE_PATH}...")
            triplets_list = []
            with open(KG_TRIPLE_PATH, 'r') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        triplets_list.append((parts[0], parts[1], parts[2]))
            
            # Sample if too large
            if len(triplets_list) > sample_size:
                triplets_list = np.random.choice(triplets_list, sample_size, replace=False)
        
        print(f"Computing distribution for {len(triplets_list)} triplets...")
        
        # Get all triplet embeddings
        triplet_embeddings = []
        for i, (h, r, t) in enumerate(triplets_list):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(triplets_list)}")
            
            emb = self.get_triplet_embedding_from_ids(h, r, t)
            triplet_embeddings.append(emb)
        
        triplet_embeddings = np.array(triplet_embeddings)
        
        # Compute statistics
        mean = np.mean(triplet_embeddings, axis=0)
        cov = np.cov(triplet_embeddings.T)
        
        # Add regularization for numerical stability
        epsilon = 1e-6
        cov += np.eye(cov.shape[0]) * epsilon
        
        # Store statistics
        self.distribution_stats = {
            'mean': mean,
            'cov': cov,
            'cov_inv': np.linalg.inv(cov),
            'n_samples': len(triplets_list)
        }
        
        # Save distribution
        self.save_distribution(save_path)
        
        print(f"✓ Distribution computed: mean={mean.shape}, cov={cov.shape}")
        return self.distribution_stats
    
    def mahalanobis_distance(self, embedding):
        """Compute Mahalanobis distance from the distribution"""
        if not self.distribution_stats:
            return 0.0
        
        mean = self.distribution_stats['mean']
        cov_inv = self.distribution_stats['cov_inv']
        
        try:
            diff = embedding - mean
            distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
            return distance
        except Exception as e:
            print(f"Error computing Mahalanobis distance: {e}")
            return 0.0
    
    def probability_density(self, embedding):
        """Compute probability density under the distribution"""
        if not self.distribution_stats:
            return 0.0
        
        mean = self.distribution_stats['mean']
        cov = self.distribution_stats['cov']
        
        try:
            diff = embedding - mean
            n = len(mean)
            
            # Multivariate Gaussian PDF
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                return 0.0
            
            const = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)
            exponent = -0.5 * np.dot(np.dot(diff, self.distribution_stats['cov_inv']), diff)
            
            density = const * np.exp(exponent)
            return density
        except Exception as e:
            print(f"Error computing PDF: {e}")
            return 0.0
    
    def percentile_score(self, embedding, reference_distribution):
        """Compute percentile score within reference distribution"""
        # For now, use Mahalanobis distance as a proxy
        distances = []
        for ref_emb in reference_distribution:
            diff = embedding - ref_emb
            dist = np.linalg.norm(diff)
            distances.append(dist)
        
        test_distance = np.linalg.norm(embedding - self.distribution_stats['mean'])
        percentile = (test_distance > np.array(distances)).sum() / len(distances)
        return percentile
    
    def detect_triplet(self, embedding):
        """Detect if a triplet embedding is watermarked"""
        # Method 1: Mahalanobis distance
        mahal_dist = self.mahalanobis_distance(embedding)
        
        # Method 2: Probability density
        pdf = self.probability_density(embedding)
        
        # Method 3: Z-score in each dimension
        mean = self.distribution_stats['mean']
        std = np.sqrt(np.diag(self.distribution_stats['cov']))
        
        z_scores = np.abs((embedding - mean) / (std + 1e-6))
        max_z = np.max(z_scores)
        mean_z = np.mean(z_scores)
        
        # Combined score: higher = more likely to be watermarked
        score = mahal_dist / (np.mean(self.distribution_stats['cov'].diagonal()) ** 0.5)
        
        return {
            'embedding': embedding,
            'mahalanobis_distance': mahal_dist,
            'probability_density': pdf,
            'max_z_score': max_z,
            'mean_z_score': mean_z,
            'watermark_score': score
        }
    
    def extract_triplets_from_text(self, text, max_triplets=10):
        """Extract potential triplets from text - simplified, fast version"""
        # Return empty list - triplets will be provided from data
        return []
    
    def detect_document(self, triplets_list, min_watermarked_triplets=3, distance_threshold=None):
        """Detect watermark in a document using provided triplets"""
        
        if not triplets_list:
            return 0, {}, []
        
        # Get embeddings and detection scores
        triplet_scores = []
        for triplet in triplets_list:
            if len(triplet) >= 3:
                h, r, t = triplet[0], triplet[1], triplet[2]
                emb = self.get_triplet_embedding_from_ids(h, r, t)
                detection_result = self.detect_triplet(emb)
                triplet_scores.append({
                    'triplet': triplet,
                    'detection': detection_result
                })
        
        if not triplet_scores:
            return 0, {}, []
        
        # Aggregate scores
        distances = [s['detection']['mahalanobis_distance'] for s in triplet_scores]
        scores = [s['detection']['watermark_score'] for s in triplet_scores]
        pdf_values = [s['detection']['probability_density'] for s in triplet_scores]
        z_scores = [s['detection']['max_z_score'] for s in triplet_scores]
        
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        median_distance = np.median(distances)
        avg_score = np.mean(scores)
        avg_pdf = np.mean(pdf_values)
        max_z = np.max(z_scores)
        
        # Set threshold if not provided
        if distance_threshold is None:
            # Use median + 2*std as threshold
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            distance_threshold = mean_dist + 2 * std_dist
        
        # Count watermarked triplets (using multiple criteria)
        watermarked_count = sum(1 for s in triplet_scores 
                                if s['detection']['mahalanobis_distance'] > distance_threshold)
        
        # Document-level decision with multiple signals
        is_watermarked = (
            watermarked_count >= min_watermarked_triplets or
            avg_distance > distance_threshold or
            max_distance > distance_threshold * 1.5 or
            max_z > 3.0  # Significant deviation in any dimension
        )
        
        details = {
            'num_triplets': len(triplet_scores),
            'watermarked_count': watermarked_count,
            'avg_distance': float(avg_distance),
            'median_distance': float(median_distance),
            'max_distance': float(max_distance),
            'avg_score': float(avg_score),
            'avg_pdf': float(avg_pdf),
            'max_z': float(max_z),
            'distance_threshold': float(distance_threshold),
            'distances': [float(d) for d in distances],
            'scores': [float(s) for s in scores]
        }
        
        return 1 if is_watermarked else 0, details, triplet_scores


def main():
    parser = argparse.ArgumentParser(description='Statistical KG Watermark Detection')
    parser.add_argument('--data_path', type=str, 
                       default='/home/wooseok/KG_Mark/outputs/c4/llama-3-8b_GraphMark_20_with_original_subgraph.jsonl')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=10000, 
                       help='Number of KG triplets to sample for distribution')
    args = parser.parse_args()
    
    print("="*60)
    print("STATISTICAL KG WATERMARK DETECTION")
    print("="*60)
    
    # Setup
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize detector
    detector = StatisticalTripletDetector(device=device)
    detector.load_models()
    
    # Load or compute KG triplet distribution
    print("\n" + "="*60)
    print("Loading/Computing KG Triplet Distribution...")
    print("="*60)
    
    distribution_file = 'kg_triplet_distribution.pkl'
    if not detector.load_distribution(distribution_file):
        # Compute new distribution if not found
        distribution_stats = detector.compute_statistical_distribution(sample_size=args.sample_size, save_path=distribution_file)
    else:
        distribution_stats = detector.distribution_stats
    
    print(f"\nDistribution Statistics:")
    print(f"  Samples: {distribution_stats['n_samples']}")
    print(f"  Mean shape: {distribution_stats['mean'].shape}")
    print(f"  Covariance shape: {distribution_stats['cov'].shape}")
    print(f"  Mean Mahalanobis distance: {np.mean(np.sqrt(np.diag(distribution_stats['cov']))):.4f}")
    
    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data...")
    print("="*60)
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} documents")
    
    # Evaluate
    predictions = []
    labels = []
    all_details = []
    
    print("\n" + "="*60)
    print("Detecting Watermarks...")
    print("="*60)
    
    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"Processing {i}/{len(data)}")
        
        # Get triplets from data
        watermarked_triplets = item.get('selected_triplets', [])
        subgraph_triplets = item.get('subgraph_triples', {})
        
        # For original text: use random triplets from subgraph (not selected ones)
        if isinstance(subgraph_triplets, dict):
            original_triplets = list(subgraph_triplets.values())[:len(watermarked_triplets)]
        elif isinstance(subgraph_triplets, list):
            original_triplets = subgraph_triplets[:len(watermarked_triplets)]
        else:
            original_triplets = []
        
        # For watermarked text: use selected_triplets (used for watermarking)
        pred_orig, details_orig, _ = detector.detect_document(original_triplets)
        predictions.append(pred_orig)
        labels.append(0)
        all_details.append(details_orig)
        
        # Watermarked text (label=1)
        pred_wm, details_wm, _ = detector.detect_document(watermarked_triplets)
        predictions.append(pred_wm)
        labels.append(1)
        all_details.append(details_wm)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Accuracy
    accuracy = np.mean(predictions == labels)
    
    # TPR, FPR
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    print(f"\nDetection Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  TPR (Recall): {tpr:.4f}")
    print(f"  FPR: {fpr:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'f1_score': float(f1),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
        'distribution_stats': {
            'n_samples': distribution_stats['n_samples'],
            'mean_norm': float(np.linalg.norm(distribution_stats['mean'])),
            'cov_trace': float(np.trace(distribution_stats['cov']))
        }
    }
    
    output_file = 'outputs/statistical_kg_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n✓ Detection complete!")


if __name__ == "__main__":
    main()


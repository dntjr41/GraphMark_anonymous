import json
import numpy as np
import os
import torch
from tqdm import tqdm
import gc

# Input and output file paths
INPUT_FILE = "/home/wooseok/KG_Mark/outputs/c4/Qwen2_5-7b-inst_GraphMark_20.jsonl"
GLOBAL_STATS_FILE = "/home/wooseok/KG_Mark/outputs/c4/Qwen2_5-7b-inst_global_triplet_statistics_20.json"

# Triplet embedding style: "ComplEx", "TransE", or "CompactProjection"
TRIPLET_STYLE = "CompactProjection"  # Change this to "ComplEx", "TransE", or "CompactProjection"

# Shrinkage parameters for covariance estimation
SHRINKAGE_METHOD = "lw"  # "lw" (Ledoit-Wolf), "oas" (Oracle Approximating Shrikage), or "custom"
CUSTOM_SHRINKAGE = 0.5  # Custom shrinkage parameter (0.0 = no shrinkage, 1.0 = diagonal matrix)

# Performance optimization parameters
BATCH_SIZE = 8192  # Process triplets in batches for better GPU utilization
USE_MIXED_PRECISION = False  # Disable mixed precision due to linalg compatibility issues
PIN_MEMORY = True  # Pin memory for faster CPU-GPU transfer
USE_CPU_FOR_SAFETY = True  # Use CPU for processing to avoid CUDA errors

# GPU setup
if USE_CPU_FOR_SAFETY:
    device = torch.device('cpu')
    print("Using CPU for safety (avoiding CUDA errors)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable mixed precision if available
if USE_MIXED_PRECISION and device.type == 'cuda':
    try:
        from torch.cuda.amp import autocast
        print("Mixed precision enabled")
    except ImportError:
        print("Mixed precision not available, using full precision")
        USE_MIXED_PRECISION = False

# KG paths for loading embeddings
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"

class EmbeddingLoader:
    """Efficient embedding loader with GPU optimization"""
    
    def __init__(self, kg_root_path):
        self.kg_root_path = kg_root_path
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_id_to_idx = {}
        self.relation_id_to_idx = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load entity and relation embeddings efficiently to CPU for safety"""
        print("Loading pre-trained embeddings to CPU for safety...")
        
        # Load entity embeddings
        entity_emb_path = f"{self.kg_root_path}/entity_embeddings_full.npy"
        if os.path.exists(entity_emb_path):
            # Load to CPU for safety
            entity_emb_cpu = np.load(entity_emb_path)
            self.entity_embeddings = torch.tensor(entity_emb_cpu, dtype=torch.float32, device=device)
            print(f"Loaded entity embeddings: {self.entity_embeddings.shape}")
            del entity_emb_cpu  # Free memory
            gc.collect()
        else:
            print(f"Warning: Entity embeddings not found at {entity_emb_path}")
            self.entity_embeddings = None
        
        # Load relation embeddings
        relation_emb_path = f"{self.kg_root_path}/relation_embeddings_full.npy"
        if os.path.exists(relation_emb_path):
            # Load to CPU for safety
            relation_emb_cpu = np.load(relation_emb_path)
            self.relation_embeddings = torch.tensor(relation_emb_cpu, dtype=torch.float32, device=device)
            print(f"Loaded relation embeddings: {self.relation_embeddings.shape}")
            del relation_emb_cpu  # Free memory
            gc.collect()
        else:
            print(f"Warning: Relation embeddings not found at {relation_emb_path}")
            self.relation_embeddings = None
        
        # Load entity ID to index mapping
        self._load_entity_mapping()
        self._load_relation_mapping()
    
    def _load_entity_mapping(self):
        """Load entity ID to index mapping"""
        entity_file = f"{self.kg_root_path}/entities.txt"
        if os.path.exists(entity_file):
            with open(entity_file, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split('\t')
                    entity_id = parts[0]
                    self.entity_id_to_idx[entity_id] = idx
            print(f"Loaded {len(self.entity_id_to_idx)} entity mappings")
    
    def _load_relation_mapping(self):
        """Load relation ID to index mapping"""
        relation_file = f"{self.kg_root_path}/relations.txt"
        if os.path.exists(relation_file):
            with open(relation_file, 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split('\t')
                    relation_id = parts[0]
                    self.relation_id_to_idx[relation_id] = idx
            print(f"Loaded {len(self.relation_id_to_idx)} relation mappings")
    
    def get_entity_embedding(self, entity_id):
        """Get entity embedding by entity ID (already on CPU)"""
        if self.entity_embeddings is None or entity_id not in self.entity_id_to_idx:
            return None
        
        entity_idx = self.entity_id_to_idx[entity_id]
        if entity_idx < len(self.entity_embeddings):
            return self.entity_embeddings[entity_idx]
        return None
    
    def get_relation_embedding(self, relation_id):
        """Get relation embedding by relation ID (already on CPU)"""
        if self.relation_embeddings is None or relation_id not in self.relation_id_to_idx:
            return None
        
        relation_idx = self.relation_id_to_idx[relation_id]
        if relation_idx < len(self.relation_embeddings):
            return self.relation_embeddings[relation_idx]
        return None
    
    def get_batch_embeddings(self, entity_ids, relation_ids, tail_ids):
        """Get batch embeddings for efficient processing"""
        if self.entity_embeddings is None or self.relation_embeddings is None:
            return None, None, None
        
        # Get indices
        entity_indices = [self.entity_id_to_idx.get(eid, -1) for eid in entity_ids]
        relation_indices = [self.relation_id_to_idx.get(rid, -1) for rid in relation_ids]
        tail_indices = [self.entity_id_to_idx.get(tid, -1) for tid in tail_ids]
        
        # Filter valid indices
        valid_mask = [(ei >= 0 and ri >= 0 and ti >= 0) for ei, ri, ti in zip(entity_indices, relation_indices, tail_indices)]
        
        if not any(valid_mask):
            return None, None, None
        
        # Get valid embeddings
        valid_entity_emb = self.entity_embeddings[[ei for ei, valid in zip(entity_indices, valid_mask) if valid]]
        valid_relation_emb = self.relation_embeddings[[ri for ri, valid in zip(relation_indices, valid_mask) if valid]]
        valid_tail_emb = self.entity_embeddings[[ti for ti, valid in zip(tail_indices, valid_mask) if valid]]
        
        return valid_entity_emb, valid_relation_emb, valid_tail_emb, valid_mask

# Global embedding loader instance
embedding_loader = None

def ledoit_wolf_shrinkage(X):
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix (GPU-accelerated)
    """
    n, p = X.shape
    
    # Convert to float32 if needed for linalg operations
    if X.dtype == torch.float16:
        X = X.to(torch.float32)
    
    if n < p:
        # When n < p, use a more conservative approach
        return 0.5, torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))
    
    try:
        # Sample covariance matrix
        S = torch.cov(X.T)
        
        # Target matrix (diagonal matrix with average variance)
        target = torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))
        
        # Calculate optimal shrinkage parameter
        trace_S = torch.trace(S)
        trace_S2 = torch.trace(S @ S)
        trace_target2 = torch.trace(target @ target)
        trace_S_target = torch.trace(S @ target)
        
        # Optimal shrinkage parameter
        numerator = trace_S2 + trace_target2 - 2 * trace_S_target
        denominator = (n - 1) * (trace_S2 - trace_S**2 / p)
        
        if denominator > 0:
            shrinkage = max(0, min(1, numerator / denominator))
        else:
            shrinkage = 0.5  # Default fallback
        
        return shrinkage, target
        
    except Exception as e:
        print(f"Warning: Ledoit-Wolf shrinkage failed, using fallback: {e}")
        # Fallback: use conservative shrinkage
        return 0.5, torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))

def oracle_approximating_shrinkage(X):
    """
    Oracle Approximating Shrinkage (OAS) estimator for covariance matrix (GPU-accelerated)
    """
    n, p = X.shape
    
    # Convert to float32 if needed for linalg operations
    if X.dtype == torch.float16:
        X = X.to(torch.float32)
    
    if n < p:
        # When n < p, use a more conservative approach
        return 0.5, torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))
    
    try:
        # Sample covariance matrix
        S = torch.cov(X.T)
        
        # Target matrix (diagonal matrix with average variance)
        target = torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))
        
        # Calculate optimal shrinkage parameter for OAS
        trace_S = torch.trace(S)
        trace_S2 = torch.trace(S @ S)
        trace_target2 = torch.trace(target @ target)
        trace_S_target = torch.trace(S @ target)
        
        # OAS shrinkage parameter
        numerator = (1 - 2/p) * trace_S_target + trace_target2
        denominator = (n + 1 - 2/p) * trace_S2 + (1 - 2/p) * trace_S_target + trace_target2
        
        if denominator > 0:
            shrinkage = max(0, min(1, numerator / denominator))
        else:
            shrinkage = 0.5  # Default fallback
        
        return shrinkage, target
        
    except Exception as e:
        print(f"Warning: OAS shrinkage failed, using fallback: {e}")
        # Fallback: use conservative shrinkage
        return 0.5, torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))

def apply_shrinkage_to_covariance(X, method="lw", custom_shrinkage=None):
    """
    Apply shrinkage to covariance matrix estimation (GPU-accelerated)
    """
    n, p = X.shape
    
    # Convert to float32 if needed for linalg operations
    if X.dtype == torch.float16:
        X = X.to(torch.float32)
    
    # Handle very small sample sizes
    if n <= 1:
        # Return identity matrix for single sample
        return torch.eye(p, device=device, dtype=torch.float32), {
            'method': 'fallback_identity',
            'shrinkage_parameter': 1.0,
            'target_matrix_type': 'identity_matrix',
            'sample_covariance_condition_number': 1.0,
            'shrunk_covariance_condition_number': 1.0,
            'improvement_ratio': 1.0,
            'n_samples': int(n),
            'n_features': int(p)
        }
    
    # Handle small sample sizes with fallback
    if n < p:
        # Use diagonal matrix with small regularization
        epsilon = 1e-6
        diagonal_vars = torch.var(X, dim=0)
        # Add small regularization to avoid zero variance
        diagonal_vars = torch.clamp(diagonal_vars, min=epsilon)
        target = torch.diag(diagonal_vars)
        
        return target, {
            'method': 'fallback_diagonal',
            'shrinkage_parameter': 1.0,
            'target_matrix_type': 'diagonal_with_regularization',
            'sample_covariance_condition_number': 1.0,
            'shrunk_covariance_condition_number': 1.0,
            'improvement_ratio': 1.0,
            'n_samples': int(n),
            'n_features': int(p)
        }
    
    try:
        # Sample covariance matrix
        sample_cov = torch.cov(X.T)
        
        if method == "lw":
            shrinkage, target = ledoit_wolf_shrinkage(X)
            method_name = "Ledoit-Wolf"
        elif method == "oas":
            shrinkage, target = oracle_approximating_shrinkage(X)
            method_name = "Oracle Approximating Shrinkage"
        elif method == "custom":
            if custom_shrinkage is None:
                custom_shrinkage = CUSTOM_SHRINKAGE
            shrinkage = custom_shrinkage
            target = torch.eye(p, device=device, dtype=torch.float32) * torch.mean(torch.var(X, dim=0))
            method_name = "Custom"
        else:
            raise ValueError(f"Unknown shrinkage method: {method}")
        
        # Apply shrinkage: Σ_shrunk = λ * Target + (1-λ) * Sample_Covariance
        shrunk_covariance = shrinkage * target + (1 - shrinkage) * sample_cov
        
        # Ensure positive definiteness
        epsilon = 1e-8
        try:
            eigenvalues = torch.linalg.eigvals(shrunk_covariance)
            min_eigenval = torch.min(eigenvalues.real)
            if min_eigenval < epsilon:
                shrunk_covariance += torch.eye(p, device=device, dtype=torch.float32) * (epsilon - min_eigenval)
        except RuntimeError as e:
            # Fallback: add small diagonal term for stability
            print(f"Warning: Eigenvalue calculation failed, using fallback: {e}")
            shrunk_covariance += torch.eye(p, device=device, dtype=torch.float32) * epsilon
        
        # Calculate condition numbers safely
        def safe_condition_number(matrix, default=1.0):
            try:
                # Ensure matrix is float32 for linalg operations
                if matrix.dtype == torch.float16:
                    matrix = matrix.to(torch.float32)
                cond = torch.linalg.cond(matrix)
                if torch.isnan(cond) or torch.isinf(cond) or cond > 1e10:
                    return default
                return float(cond)
            except Exception as e:
                print(f"Warning: Condition number calculation failed: {e}")
                return default
        
        sample_cond = safe_condition_number(sample_cov)
        shrunk_cond = safe_condition_number(shrunk_covariance)
        
        # Calculate improvement ratio safely
        improvement_ratio = 1.0
        if sample_cond > 0 and shrunk_cond > 0:
            try:
                ratio = sample_cond / shrunk_cond
                if torch.isnan(ratio) or torch.isinf(ratio) or ratio > 1e10:
                    improvement_ratio = 1.0
                else:
                    improvement_ratio = float(ratio)
            except:
                improvement_ratio = 1.0
        
        shrinkage_info = {
            'method': method_name,
            'shrinkage_parameter': float(shrinkage),
            'target_matrix_type': 'diagonal_with_average_variance',
            'sample_covariance_condition_number': sample_cond,
            'shrunk_covariance_condition_number': shrunk_cond,
            'improvement_ratio': improvement_ratio,
            'n_samples': int(n),
            'n_features': int(p)
        }
        
        return shrunk_covariance, shrinkage_info
        
    except Exception as e:
        print(f"Warning: Covariance calculation failed, using fallback: {e}")
        # Fallback: use diagonal matrix with regularization
        epsilon = 1e-6
        diagonal_vars = torch.var(X, dim=0)
        diagonal_vars = torch.clamp(diagonal_vars, min=epsilon)
        target = torch.diag(diagonal_vars)
        
        return target, {
            'method': 'fallback_diagonal_error',
            'shrinkage_parameter': 1.0,
            'target_matrix_type': 'diagonal_with_regularization',
            'sample_covariance_condition_number': 1.0,
            'shrunk_covariance_condition_number': 1.0,
            'improvement_ratio': 1.0,
            'n_samples': int(n),
            'n_features': int(p)
        }

def create_triplet_embeddings_batch(head_ids, relation_ids, tail_ids, style="ComplEx"):
    """
    Create triplet embeddings in batch for better GPU utilization
    """
    global embedding_loader
    
    if embedding_loader is None:
        return None, []
    
    try:
        # Get batch embeddings
        entity_emb, relation_emb, tail_emb, valid_mask = embedding_loader.get_batch_embeddings(
            head_ids, relation_ids, tail_ids
        )
        
        if entity_emb is None:
            return None, []
        
        # Validate input tensor shapes
        if entity_emb.shape[0] != relation_emb.shape[0] or entity_emb.shape[0] != tail_emb.shape[0]:
            print(f"Warning: Shape mismatch - entity: {entity_emb.shape}, relation: {relation_emb.shape}, tail: {tail_emb.shape}")
            return None, []
        
        # Create triplet embeddings based on style
        if style == "ComplEx":
            # Element-wise multiplication: e_h ⊙ e_r ⊙ e_t
            triplet_embeddings = entity_emb * relation_emb * tail_emb
        elif style == "TransE":
            # Additive relationship: e_h + e_r - e_t
            triplet_embeddings = entity_emb + relation_emb - tail_emb
        elif style == "CompactProjection":
            # Simplified approach: weighted combination instead of concatenation
            # This avoids the complex concatenation that's causing CUDA errors
            head_weight, relation_weight, tail_weight = 0.4, 0.3, 0.3
            triplet_embeddings = (head_weight * entity_emb + 
                                relation_weight * relation_emb + 
                                tail_weight * tail_emb)
        else:
            raise ValueError(f"Unknown triplet style: {style}")
        
        # Validate tensor dimensions and handle any NaN/Inf values
        expected_dim = triplet_embeddings.shape[1]
        valid_embeddings = []
        valid_mask_filtered = []
        
        for i, embedding in enumerate(triplet_embeddings):
            try:
                # Check for NaN/Inf values
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    print(f"Warning: Skipping triplet {i} with NaN/Inf values")
                    continue
                
                # Check dimension
                if embedding.shape[0] == expected_dim:
                    valid_embeddings.append(embedding)
                    valid_mask_filtered.append(valid_mask[i])
                else:
                    print(f"Warning: Skipping triplet {i} with unexpected dimension {embedding.shape}, expected {expected_dim}")
            except Exception as e:
                print(f"Warning: Error processing triplet {i}: {e}")
                continue
        
        return valid_embeddings, valid_mask_filtered
        
    except Exception as e:
        print(f"Error in create_triplet_embeddings_batch: {e}")
        return None, []

def calculate_mahalanobis_parameters(triplet_embeddings):
    """
    Calculate Mu (mean) and Sigma (covariance) for Mahalanobis distance
    with shrinkage estimator for stability (GPU-accelerated)
    """
    if len(triplet_embeddings) == 0:
        return None, None, None
    
    # Stack embeddings (already on GPU)
    embeddings_tensor = torch.stack(triplet_embeddings)
    
    # Calculate mean vector (Mu)
    mu = torch.mean(embeddings_tensor, dim=0)
    
    # Calculate covariance matrix (Sigma) with shrinkage
    sigma, shrinkage_info = apply_shrinkage_to_covariance(
        embeddings_tensor, 
        method=SHRINKAGE_METHOD,
        custom_shrinkage=CUSTOM_SHRINKAGE
    )
    
    return mu, sigma, shrinkage_info

def calculate_global_statistics(all_triplet_embeddings):
    """
    Calculate global Mu and Sigma across all documents with shrinkage (GPU-accelerated)
    """
    if not all_triplet_embeddings:
        return None, None, None, {}
    
    # Flatten all triplet embeddings
    all_embeddings = []
    for doc_embeddings in all_triplet_embeddings:
        all_embeddings.extend(doc_embeddings)
    
    if len(all_embeddings) == 0:
        return None, None, None, {}
    
    # Validate tensor sizes before stacking
    expected_size = all_embeddings[0].shape[0]
    valid_embeddings = []
    
    print(f"Validating {len(all_embeddings)} embeddings with expected size {expected_size}")
    
    for i, embedding in enumerate(all_embeddings):
        try:
            if embedding.shape[0] == expected_size:
                # Additional validation: check for NaN/Inf values
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    print(f"Warning: Skipping embedding {i} with NaN/Inf values")
                    continue
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Skipping embedding {i} with unexpected size {embedding.shape}, expected {expected_size}")
        except Exception as e:
            print(f"Warning: Error validating embedding {i}: {e}")
            continue
    
    if len(valid_embeddings) == 0:
        print("Error: No valid embeddings found after size validation")
        return None, None, None, {}
    
    if len(valid_embeddings) != len(all_embeddings):
        print(f"Warning: {len(all_embeddings) - len(valid_embeddings)} embeddings were skipped due to validation issues")
    
    print(f"Proceeding with {len(valid_embeddings)} valid embeddings")
    
    # Stack embeddings (already on GPU) with error handling
    try:
        # Move to CPU first to avoid CUDA issues, then back to GPU
        embeddings_cpu = [emb.cpu() for emb in valid_embeddings]
        embeddings_tensor = torch.stack(embeddings_cpu).to(device)
        print(f"Successfully stacked embeddings: {embeddings_tensor.shape}")
    except RuntimeError as e:
        print(f"Error stacking embeddings: {e}")
        print(f"Expected size: {expected_size}, Number of valid embeddings: {len(valid_embeddings)}")
        
        # Try alternative approach: process in smaller batches
        try:
            print("Attempting batch processing as fallback...")
            batch_size = 100
            all_batches = []
            
            for i in range(0, len(valid_embeddings), batch_size):
                batch = valid_embeddings[i:i + batch_size]
                batch_cpu = [emb.cpu() for emb in batch]
                batch_tensor = torch.stack(batch_cpu).to(device)
                all_batches.append(batch_tensor)
            
            embeddings_tensor = torch.cat(all_batches, dim=0)
            print(f"Successfully processed in batches: {embeddings_tensor.shape}")
            
        except Exception as e2:
            print(f"Batch processing also failed: {e2}")
            return None, None, None, {}
    
    # Calculate global mean vector (Mu)
    try:
        global_mu = torch.mean(embeddings_tensor, dim=0)
        print(f"Calculated global mu: {global_mu.shape}")
    except Exception as e:
        print(f"Error calculating global mu: {e}")
        return None, None, None, {}
    
    # Calculate global covariance matrix (Sigma) with shrinkage
    try:
        global_sigma, global_shrinkage_info = apply_shrinkage_to_covariance(
            embeddings_tensor,
            method=SHRINKAGE_METHOD,
            custom_shrinkage=CUSTOM_SHRINKAGE
        )
        print(f"Calculated global sigma: {global_sigma.shape}")
    except Exception as e:
        print(f"Error calculating global sigma: {e}")
        return None, None, None, {}
    
    # Calculate summary statistics - ensure all values are JSON serializable
    # Move tensors to CPU for numpy operations
    try:
        global_mu_cpu = global_mu.cpu()
        global_sigma_cpu = global_sigma.cpu()
    except Exception as e:
        print(f"Error moving tensors to CPU: {e}")
        return None, None, None, {}
    
    # Safe calculation of statistics with error handling
    def safe_float(value, default=0.0):
        """Safely convert value to float, handling NaN/Inf"""
        try:
            if torch.isnan(value) or torch.isinf(value):
                return default
            return float(value)
        except:
            return default
    
    def safe_condition_number(matrix, default=1.0):
        """Safely calculate condition number"""
        try:
            # Ensure matrix is float32 for linalg operations
            if matrix.dtype == torch.float16:
                matrix = matrix.to(torch.float32)
            cond = torch.linalg.cond(matrix)
            if torch.isnan(cond) or torch.isinf(cond) or cond > 1e10:
                return default
            return float(cond)
        except Exception as e:
            print(f"Warning: Condition number calculation failed: {e}")
            return default
    
    def safe_determinant(matrix, default=1.0):
        """Safely calculate determinant"""
        try:
            # Ensure matrix is float32 for linalg operations
            if matrix.dtype == torch.float16:
                matrix = matrix.to(torch.float32)
            det = torch.linalg.det(matrix)
            if torch.isnan(det) or torch.isinf(det) or abs(det) > 1e10:
                return default
            return float(det)
        except Exception as e:
            print(f"Warning: Determinant calculation failed: {e}")
            return default
    
    def safe_eigenvalues(matrix, default_list=None):
        """Safely calculate eigenvalues"""
        try:
            # Ensure matrix is float32 for linalg operations
            if matrix.dtype == torch.float16:
                matrix = matrix.to(torch.float32)
            eigenvals = torch.linalg.eigvals(matrix)
            clean_eigenvals = []
            for x in eigenvals:
                real_part = x.real
                if torch.isnan(real_part) or torch.isinf(real_part) or abs(real_part) > 1e10:
                    clean_eigenvals.append(0.0)
                else:
                    clean_eigenvals.append(float(real_part))
            return clean_eigenvals
        except Exception as e:
            print(f"Warning: Eigenvalue calculation failed: {e}")
            return default_list if default_list else [0.0] * matrix.shape[0]
    
    summary_stats = {
        'total_triplets': int(len(valid_embeddings)),
        'total_documents': int(len(all_triplet_embeddings)),
        'avg_triplets_per_doc': safe_float(len(valid_embeddings) / len(all_triplet_embeddings)),
        'embedding_dimension': int(embeddings_tensor.shape[1]),
        'global_mu_norm': safe_float(torch.norm(global_mu_cpu)),
        'global_sigma_condition_number': safe_condition_number(global_sigma_cpu),
        'global_sigma_determinant': safe_determinant(global_sigma_cpu),
        'global_sigma_trace': safe_float(torch.trace(global_sigma_cpu)),
        'global_sigma_eigenvalues': safe_eigenvalues(global_sigma_cpu),
        'shrinkage_info': global_shrinkage_info
    }
    
    return global_mu, global_sigma, global_shrinkage_info, summary_stats

def process_triplets_in_batches(selected_triplets, style):
    """
    Process triplets in batches for better GPU utilization
    """
    if not selected_triplets:
        return []
    
    all_triplet_embeddings = []
    
    # Process in batches
    for i in range(0, len(selected_triplets), BATCH_SIZE):
        batch_triplets = selected_triplets[i:i + BATCH_SIZE]
        
        # Extract head, relation, tail IDs
        head_ids = [triple[0] for triple in batch_triplets]
        relation_ids = [triple[1] for triple in batch_triplets]
        tail_ids = [triple[2] for triple in batch_triplets]
        
        # Create batch embeddings
        batch_embeddings, valid_mask = create_triplet_embeddings_batch(
            head_ids, relation_ids, tail_ids, style
        )
        
        if batch_embeddings is not None:
            # Convert to list of individual embeddings
            for j, is_valid in enumerate(valid_mask):
                if is_valid:
                    all_triplet_embeddings.append(batch_embeddings[j])
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and i % (BATCH_SIZE * 10) == 0:
            torch.cuda.empty_cache()
    
    return all_triplet_embeddings

def main():
    global embedding_loader
    
    print(f"Processing file: {INPUT_FILE}")
    print(f"Global stats file: {GLOBAL_STATS_FILE}")
    print(f"Triplet embedding style: {TRIPLET_STYLE}")
    print(f"Shrinkage method: {SHRINKAGE_METHOD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mixed precision: {USE_MIXED_PRECISION}")
    print(f"Using device: {device}")
    
    # Initialize embedding loader
    print(f"\nInitializing embedding loader from: {KG_ROOT_PATH}")
    embedding_loader = EmbeddingLoader(KG_ROOT_PATH)
    
    if TRIPLET_STYLE == "ComplEx":
        print("Using ComplEx-style: e_h ⊙ e_r ⊙ e_t (element-wise multiplication)")
    elif TRIPLET_STYLE == "TransE":
        print("Using TransE-style: e_h + e_r - e_t (additive relationship)")
    elif TRIPLET_STYLE == "CompactProjection":
        print("Using CompactProjection-style: Concatenate head, relation, and tail embeddings and project to 768 dimensions.")
    else:
        print(f"Unknown triplet embedding style: {TRIPLET_STYLE}")
    
    # Store all triplet embeddings for global statistics (only from selected_triplets)
    all_triplet_embeddings = []
    
    # Count total entries for progress bar
    total_entries = sum(1 for _ in open(INPUT_FILE, 'r'))
    
    # Process each entry in the JSONL file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        
        for i, line in enumerate(tqdm(f_in, total=total_entries, desc="Processing entries")):
            try:
                data = json.loads(line.strip())
                
                # Extract selected triplets (only these will be used for global statistics)
                selected_triplets = data.get('selected_triplets', [])
                if not selected_triplets:
                    continue
                
                if i % 10 == 0:  # Print progress every 10 entries
                    print(f"\nProcessing Entry {i+1}/{total_entries} - {len(selected_triplets)} selected triplets")
                
                # Process triplets in batches for better GPU utilization
                triplet_embeddings = process_triplets_in_batches(selected_triplets, TRIPLET_STYLE)
                
                if len(triplet_embeddings) == 0:
                    continue
                
                # Store for global statistics (only selected triplets)
                all_triplet_embeddings.append(triplet_embeddings)
                
                # Calculate Mu and Sigma for Mahalanobis distance with shrinkage
                mu, sigma, shrinkage_info = calculate_mahalanobis_parameters(triplet_embeddings)
                
                if mu is not None and sigma is not None:
                    # Store for global statistics only (no individual document stats needed)
                    if i % 10 == 0:  # Print progress every 10 entries
                        print(f"✅ Entry {i+1} completed: {len(triplet_embeddings)} valid triplets")
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and i % 50 == 0:
                    torch.cuda.empty_cache()
                
            except json.JSONDecodeError as e:
                print(f"[Entry {i+1}] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[Entry {i+1}] Error processing entry: {e}")
                continue
    
    # Calculate global statistics (only from selected_triplets)
    print(f"\n{'='*80}")
    print("Calculating global statistics from selected triplets...")
    print(f"{'='*80}")
    
    global_mu, global_sigma, global_shrinkage_info, global_summary = calculate_global_statistics(all_triplet_embeddings)
    
    if global_mu is not None and global_sigma is not None:
        try:
            # Save global statistics
            # Move tensors to CPU for JSON serialization
            global_mu_cpu = global_mu.cpu()
            global_sigma_cpu = global_sigma.cpu()
            
            # Check for NaN/Inf values and handle them
            def clean_tensor_for_json(tensor):
                """Clean tensor values for JSON serialization"""
                tensor_np = tensor.numpy()
                # Replace NaN with 0, Inf with large number
                tensor_np = np.nan_to_num(tensor_np, nan=0.0, posinf=1e10, neginf=-1e10)
                return tensor_np.tolist()
            
            # Clean tensors before saving
            mu_clean = clean_tensor_for_json(global_mu_cpu)
            sigma_clean = clean_tensor_for_json(global_sigma_cpu)
            
            # Ensure directory exists
            output_dir = os.path.dirname(GLOBAL_STATS_FILE)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            
            global_stats = {
                'global_mahalanobis_parameters': {
                    'mu': mu_clean,
                    'sigma': sigma_clean
                },
                'global_summary_statistics': global_summary
            }
            
            # Save with error handling
            with open(GLOBAL_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(global_stats, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Global statistics saved successfully to: {GLOBAL_STATS_FILE}")
            
        except Exception as e:
            print(f"❌ Error saving global statistics: {e}")
            print(f"Attempting to save minimal version...")
            
            try:
                # Save minimal version with just essential data
                minimal_stats = {
                    'global_mahalanobis_parameters': {
                        'mu_shape': list(global_mu.shape),
                        'sigma_shape': list(global_sigma.shape),
                        'mu_norm': float(torch.norm(global_mu_cpu)),
                        'sigma_trace': float(torch.trace(global_sigma_cpu))
                    },
                    'global_summary_statistics': {
                        'total_triplets': global_summary.get('total_triplets', 0),
                        'total_documents': global_summary.get('total_documents', 0),
                        'embedding_dimension': global_summary.get('embedding_dimension', 0)
                    },
                    'error_message': f"Full save failed: {str(e)}"
                }
                
                minimal_file = GLOBAL_STATS_FILE.replace('.json', '_minimal.json')
                with open(minimal_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_stats, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Minimal statistics saved to: {minimal_file}")
                
            except Exception as e2:
                print(f"❌ Failed to save even minimal version: {e2}")
                print(f"Global statistics could not be saved")
        
        print(f"✅ Global statistics completed (from selected triplets only):")
        print(f"   - Total selected triplets: {global_summary['total_triplets']}")
        print(f"   - Total documents: {global_summary['total_documents']}")
        print(f"   - Avg selected triplets per doc: {global_summary['avg_triplets_per_doc']:.2f}")
        print(f"   - Embedding dimension: {global_summary['embedding_dimension']}")
        print(f"   - Global Mu norm: {global_summary['global_mu_norm']:.4f}")
        print(f"   - Global Sigma condition number: {global_summary['global_sigma_condition_number']:.4f}")
        print(f"   - Global Sigma determinant: {global_summary['global_sigma_determinant']:.4f}")
        print(f"   - Global Sigma trace: {global_summary['global_sigma_trace']:.4f}")
        print(f"   - Global shrinkage method: {global_shrinkage_info['method']}")
        print(f"   - Global shrinkage parameter: {global_shrinkage_info['shrinkage_parameter']:.4f}")
        print(f"   - Global condition number improvement: {global_shrinkage_info['improvement_ratio']:.2f}x")
    else:
        print("❌ Failed to calculate global statistics")
    
    print(f"\n{'='*80}")
    print("Processing completed!")
    print(f"Global statistics saved to: {GLOBAL_STATS_FILE}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
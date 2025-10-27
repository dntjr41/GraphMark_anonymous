import json
import numpy as np
import os
import torch

# Input and output file paths
INPUT_FILE = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_GraphMark_15.jsonl"
GLOBAL_STATS_FILE = "/home/wooseok/KG_Mark/outputs/opengen/Qwen_global_triplet_statistics_15.json"

# Triplet embedding style: "ComplEx", "TransE", or "CompactProjection"
TRIPLET_STYLE = "CompactProjection"  # Change this to "ComplEx", "TransE", or "CompactProjection"

# Shrinkage parameters for covariance estimation
SHRINKAGE_METHOD = "lw"  # "lw" (Ledoit-Wolf), "oas" (Oracle Approximating Shrikage), or "custom"
CUSTOM_SHRINKAGE = 0.5  # Custom shrinkage parameter (0.0 = no shrinkage, 1.0 = diagonal matrix)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# KG paths for loading embeddings
KG_ROOT_PATH = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
KG_ENTITY_PATH = f"{KG_ROOT_PATH}/entities.txt"
KG_RELATION_PATH = f"{KG_ROOT_PATH}/relations.txt"
KG_TRIPLE_PATH = f"{KG_ROOT_PATH}/triplets.txt"

class EmbeddingLoader:
    """Efficient embedding loader from subgraph_construction.py"""
    
    def __init__(self, kg_root_path):
        self.kg_root_path = kg_root_path
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_id_to_idx = {}
        self.relation_id_to_idx = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load entity and relation embeddings efficiently"""
        print("Loading pre-trained embeddings...")
        
        # Load entity embeddings
        entity_emb_path = f"{self.kg_root_path}/entity_embeddings_full.npy"
        if os.path.exists(entity_emb_path):
            self.entity_embeddings = np.load(entity_emb_path)
            print(f"Loaded entity embeddings: {self.entity_embeddings.shape}")
        else:
            print(f"Warning: Entity embeddings not found at {entity_emb_path}")
            self.entity_embeddings = None
        
        # Load relation embeddings
        relation_emb_path = f"{self.kg_root_path}/relation_embeddings_full.npy"
        if os.path.exists(relation_emb_path):
            self.relation_embeddings = np.load(relation_emb_path)
            print(f"Loaded relation embeddings: {self.relation_embeddings.shape}")
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
        """Get entity embedding by entity ID"""
        if self.entity_embeddings is None or entity_id not in self.entity_id_to_idx:
            return None
        
        entity_idx = self.entity_id_to_idx[entity_id]
        if entity_idx < len(self.entity_embeddings):
            return self.entity_embeddings[entity_idx]
        return None
    
    def get_relation_embedding(self, relation_id):
        """Get relation embedding by relation ID"""
        if self.relation_embeddings is None or relation_id not in self.relation_id_to_idx:
            return None
        
        relation_idx = self.relation_id_to_idx[relation_id]
        if relation_idx < len(self.relation_embeddings):
            return self.relation_embeddings[relation_idx]
        return None

# Global embedding loader instance
embedding_loader = None

def ledoit_wolf_shrinkage(X):
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix (GPU-accelerated)
    
    Reference: Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for 
    large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.
    
    Args:
        X: Data matrix (n_samples, n_features) as torch tensor
    
    Returns:
        shrinkage: Optimal shrinkage parameter
        target: Target matrix (diagonal matrix with average variance)
    """
    n, p = X.shape
    
    if n < p:
        # When n < p, use a more conservative approach
        return 0.5, torch.eye(p, device=device) * torch.mean(torch.var(X, dim=0))
    
    # Sample covariance matrix
    S = torch.cov(X.T)
    
    # Target matrix (diagonal matrix with average variance)
    target = torch.eye(p, device=device) * torch.mean(torch.var(X, dim=0))
    
    # Calculate optimal shrinkage parameter
    # This is a simplified version of the Ledoit-Wolf estimator
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

def oracle_approximating_shrinkage(X):
    """
    Oracle Approximating Shrinkage (OAS) estimator for covariance matrix (GPU-accelerated)
    
    Reference: Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). 
    Shrinkage algorithms for MMSE covariance estimation. IEEE Transactions on Signal Processing, 58(10), 5016-5029.
    
    Args:
        X: Data matrix (n_samples, n_features) as torch tensor
    
    Returns:
        shrinkage: Optimal shrinkage parameter
        target: Target matrix (diagonal matrix with average variance)
    """
    n, p = X.shape
    
    if n < p:
        # When n < p, use a more conservative approach
        return 0.5, torch.eye(p, device=device) * torch.mean(torch.var(X, dim=0))
    
    # Sample covariance matrix
    S = torch.cov(X.T)
    
    # Target matrix (diagonal matrix with average variance)
    target = torch.eye(p, device=device) * torch.mean(torch.var(X, dim=0))
    
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

def apply_shrinkage_to_covariance(X, method="lw", custom_shrinkage=None):
    """
    Apply shrinkage to covariance matrix estimation (GPU-accelerated)
    
    Args:
        X: Data matrix (n_samples, n_features) as torch tensor
        method: Shrinkage method ("lw", "oas", or "custom")
        custom_shrinkage: Custom shrinkage parameter (0.0 to 1.0)
    
    Returns:
        shrunk_covariance: Shrunk covariance matrix
        shrinkage_info: Dictionary with shrinkage details
    """
    n, p = X.shape
    
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
        target = torch.eye(p, device=device) * torch.mean(torch.var(X, dim=0))
        method_name = "Custom"
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")
    
    # Apply shrinkage: Σ_shrunk = λ * Target + (1-λ) * Sample_Covariance
    shrunk_covariance = shrinkage * target + (1 - shrinkage) * sample_cov
    
    # Ensure positive definiteness
    epsilon = 1e-8
    eigenvalues = torch.linalg.eigvals(shrunk_covariance)
    min_eigenval = torch.min(eigenvalues.real)
    if min_eigenval < epsilon:
        shrunk_covariance += torch.eye(p, device=device) * (epsilon - min_eigenval)
    
    # Calculate condition numbers safely
    def safe_condition_number(matrix, default=1.0):
        try:
            cond = torch.linalg.cond(matrix)
            if torch.isnan(cond) or torch.isinf(cond) or cond > 1e10:
                return default
            return float(cond)
        except:
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

def create_triplet_embedding(head_id, relation_id, tail_id):
    """
    Create triplet embedding using ComplEx-style: e_h ⊙ e_r ⊙ e_t ∈ R^d
    where ⊙ represents element-wise multiplication (Hadamard product)
    """
    global embedding_loader
    
    # Get actual embeddings from the loader
    head_embedding = embedding_loader.get_entity_embedding(head_id)
    tail_embedding = embedding_loader.get_entity_embedding(tail_id)
    relation_embedding = embedding_loader.get_relation_embedding(relation_id)
    
    # Check if all embeddings are available
    if head_embedding is None or tail_embedding is None or relation_embedding is None:
        return None
    
    # Convert to torch tensors on GPU
    head_tensor = torch.tensor(head_embedding, dtype=torch.float32, device=device)
    tail_tensor = torch.tensor(tail_embedding, dtype=torch.float32, device=device)
    relation_tensor = torch.tensor(relation_embedding, dtype=torch.float32, device=device)
    
    # ComplEx-style: element-wise multiplication of head, relation, and tail embeddings
    # e_h ⊙ e_r ⊙ e_t
    triplet_embedding = head_tensor * relation_tensor * tail_tensor
    
    return triplet_embedding

def create_transE_triplet_embedding(head_id, relation_id, tail_id):
    """
    Create triplet embedding using TransE-style: e_h + e_r ≈ e_t
    where the triplet embedding represents the relationship: e_h + e_r - e_t
    """
    global embedding_loader
    
    # Get actual embeddings from the loader
    head_embedding = embedding_loader.get_entity_embedding(head_id)
    tail_embedding = embedding_loader.get_entity_embedding(tail_id)
    relation_embedding = embedding_loader.get_relation_embedding(relation_id)
    
    # Check if all embeddings are available
    if head_embedding is None or tail_embedding is None or relation_embedding is None:
        return None
    
    # Convert to torch tensors on GPU
    head_tensor = torch.tensor(head_embedding, dtype=torch.float32, device=device)
    tail_tensor = torch.tensor(tail_embedding, dtype=torch.float32, device=device)
    relation_tensor = torch.tensor(relation_embedding, dtype=torch.float32, device=device)
    
    # TransE-style: e_h + e_r - e_t
    triplet_embedding = head_tensor + relation_tensor - tail_tensor
    
    return triplet_embedding

def create_compact_projection_triplet_embedding(head_id, relation_id, tail_id):
    """
    Create triplet embedding using CompactProjection-style: 
    1. Concatenate head, relation, and tail embeddings: [e_h; e_r; e_t] ∈ R^3d
    2. Project to target dimension using linear transformation: W ∈ R^(d×3d)
    3. Result: W * [e_h; e_r; e_t] ∈ R^d
    
    For now, we'll use a simple truncation/padding approach as a placeholder
    """
    global embedding_loader
    
    # Get actual embeddings from the loader
    head_embedding = embedding_loader.get_entity_embedding(head_id)
    tail_embedding = embedding_loader.get_entity_embedding(tail_id)
    relation_embedding = embedding_loader.get_relation_embedding(relation_id)
    
    # Check if all embeddings are available
    if head_embedding is None or tail_embedding is None or relation_embedding is None:
        return None
    
    # Convert to torch tensors on GPU
    head_tensor = torch.tensor(head_embedding, dtype=torch.float32, device=device)
    tail_tensor = torch.tensor(tail_embedding, dtype=torch.float32, device=device)
    relation_tensor = torch.tensor(relation_embedding, dtype=torch.float32, device=device)
    
    # Step 1: Concatenate embeddings [e_h; e_r; e_t]
    # Each embedding is 768-dimensional, so concatenated is 2304-dimensional
    concatenated_embedding = torch.cat([head_tensor, relation_tensor, tail_tensor])
    
    # Step 2: Project to 768 dimensions
    # In a real implementation, this would be: W * concatenated_embedding
    # where W is a learned projection matrix W ∈ R^(768×2304)
    
    # For now, we'll use a simple approach:
    # Option 1: Take first 768 dimensions (truncation)
    # Option 2: Use mean pooling over chunks
    # Option 3: Use a simple linear combination
    
    # Option 1: Simple truncation (preserves head information)
    # triplet_embedding = concatenated_embedding[:768]
    
    # Option 2: Mean pooling over 3 chunks (preserves all information)
    chunk_size = 768
    triplet_embedding = torch.zeros(768, device=device)
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = concatenated_embedding[start_idx:end_idx]
        triplet_embedding += chunk / 3.0
    
    # Option 3: Weighted combination (can be tuned)
    # head_weight, relation_weight, tail_weight = 0.4, 0.3, 0.3
    # triplet_embedding = (head_weight * head_tensor + 
    #                     relation_weight * relation_tensor + 
    #                     tail_weight * tail_tensor)
    
    return triplet_embedding

def calculate_mahalanobis_parameters(triplet_embeddings):
    """
    Calculate Mu (mean) and Sigma (covariance) for Mahalanobis distance
    with shrinkage estimator for stability (GPU-accelerated)
    """
    if len(triplet_embeddings) == 0:
        return None, None, None
    
    # Convert to torch tensor and stack
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
    
    # Convert to torch tensor and stack
    embeddings_tensor = torch.stack(all_embeddings)
    
    # Calculate global mean vector (Mu)
    global_mu = torch.mean(embeddings_tensor, dim=0)
    
    # Calculate global covariance matrix (Sigma) with shrinkage
    global_sigma, global_shrinkage_info = apply_shrinkage_to_covariance(
        embeddings_tensor,
        method=SHRINKAGE_METHOD,
        custom_shrinkage=CUSTOM_SHRINKAGE
    )
    
    # Calculate summary statistics - ensure all values are JSON serializable
    # Move tensors to CPU for numpy operations
    global_mu_cpu = global_mu.cpu()
    global_sigma_cpu = global_sigma.cpu()
    
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
            cond = torch.linalg.cond(matrix)
            if torch.isnan(cond) or torch.isinf(cond) or cond > 1e10:
                return default
            return float(cond)
        except:
            return default
    
    def safe_determinant(matrix, default=1.0):
        """Safely calculate determinant"""
        try:
            det = torch.linalg.det(matrix)
            if torch.isnan(det) or torch.isinf(det) or abs(det) > 1e10:
                return default
            return float(det)
        except:
            return default
    
    def safe_eigenvalues(matrix, default_list=None):
        """Safely calculate eigenvalues"""
        try:
            eigenvals = torch.linalg.eigvals(matrix)
            clean_eigenvals = []
            for x in eigenvals:
                real_part = x.real
                if torch.isnan(real_part) or torch.isinf(real_part) or abs(real_part) > 1e10:
                    clean_eigenvals.append(0.0)
                else:
                    clean_eigenvals.append(float(real_part))
            return clean_eigenvals
        except:
            return default_list if default_list else [0.0] * matrix.shape[0]
    
    summary_stats = {
        'total_triplets': int(len(all_embeddings)),
        'total_documents': int(len(all_triplet_embeddings)),
        'avg_triplets_per_doc': safe_float(len(all_embeddings) / len(all_triplet_embeddings)),
        'embedding_dimension': int(embeddings_tensor.shape[1]),
        'global_mu_norm': safe_float(torch.norm(global_mu_cpu)),
        'global_sigma_condition_number': safe_condition_number(global_sigma_cpu),
        'global_sigma_determinant': safe_determinant(global_sigma_cpu),
        'global_sigma_trace': safe_float(torch.trace(global_sigma_cpu)),
        'global_sigma_eigenvalues': safe_eigenvalues(global_sigma_cpu),
        'shrinkage_info': global_shrinkage_info
    }
    
    return global_mu, global_sigma, global_shrinkage_info, summary_stats

def main():
    global embedding_loader
    
    print(f"Processing file: {INPUT_FILE}")
    print(f"Global stats file: {GLOBAL_STATS_FILE}")
    print(f"Triplet embedding style: {TRIPLET_STYLE}")
    print(f"Shrinkage method: {SHRINKAGE_METHOD}")
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
    document_stats = []
    
    # Process each entry in the JSONL file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line.strip())
                
                # Extract selected triplets (only these will be used for global statistics)
                selected_triplets = data.get('selected_triplets', [])
                if not selected_triplets:
                    print(f"[Entry {i+1}] No selected triplets found, skipping...")
                    continue
                
                print(f"\n{'='*80}")
                print(f"Processing Entry {i+1}")
                print(f"Number of selected triplets: {len(selected_triplets)}")
                print(f"{'='*80}")
                
                # Create triplet embeddings only for selected triplets
                triplet_embeddings = []
                valid_triples = []
                
                for triple in selected_triplets:
                    if len(triple) == 3:
                        head_id, relation_id, tail_id = triple
                        
                        # Create triplet embedding based on selected style
                        if TRIPLET_STYLE == "ComplEx":
                            triplet_embedding = create_triplet_embedding(head_id, relation_id, tail_id)
                        elif TRIPLET_STYLE == "TransE":
                            triplet_embedding = create_transE_triplet_embedding(head_id, relation_id, tail_id)
                        elif TRIPLET_STYLE == "CompactProjection":
                            triplet_embedding = create_compact_projection_triplet_embedding(head_id, relation_id, tail_id)
                        else:
                            print(f"  Skipping triple {triple} - unknown triplet style")
                            continue
                        
                        if triplet_embedding is not None:
                            triplet_embeddings.append(triplet_embedding)
                            valid_triples.append(triple)
                        else:
                            print(f"  Skipping triple {triple} - failed to create embedding")
                
                print(f"Valid triplet embeddings: {len(triplet_embeddings)}/{len(selected_triplets)}")
                
                if len(triplet_embeddings) == 0:
                    print(f"[Entry {i+1}] No valid triplet embeddings, skipping...")
                    continue
                
                # Store for global statistics (only selected triplets)
                all_triplet_embeddings.append(triplet_embeddings)
                
                # Calculate Mu and Sigma for Mahalanobis distance with shrinkage
                mu, sigma, shrinkage_info = calculate_mahalanobis_parameters(triplet_embeddings)
                
                if mu is not None and sigma is not None:
                    # Store document statistics
                    # Move tensors to CPU for numpy operations
                    mu_cpu = mu.cpu()
                    sigma_cpu = sigma.cpu()
                    
                    # Safe calculation of document statistics
                    def safe_float(value, default=0.0):
                        try:
                            if torch.isnan(value) or torch.isinf(value):
                                return default
                            return float(value)
                        except:
                            return default
                    
                    def safe_condition_number(matrix, default=1.0):
                        try:
                            cond = torch.linalg.cond(matrix)
                            if torch.isnan(cond) or torch.isinf(cond) or cond > 1e10:
                                return default
                            return float(cond)
                        except:
                            return default
                    
                    def safe_determinant(matrix, default=1.0):
                        try:
                            det = torch.linalg.det(matrix)
                            if torch.isnan(det) or torch.isinf(det) or abs(det) > 1e10:
                                return default
                            return float(det)
                        except:
                            return default
                    
                    document_stats.append({
                        'entry_id': int(i + 1),
                        'num_triplets': int(len(triplet_embeddings)),
                        'embedding_dimension': int(triplet_embeddings[0].shape[0]) if triplet_embeddings else 0,
                        'mu_norm': safe_float(torch.norm(mu_cpu)),
                        'sigma_condition_number': safe_condition_number(sigma_cpu),
                        'sigma_determinant': safe_determinant(sigma_cpu),
                        'sigma_trace': safe_float(torch.trace(sigma_cpu)),
                        'shrinkage_method': shrinkage_info['method'],
                        'shrinkage_parameter': shrinkage_info['shrinkage_parameter'],
                        'condition_number_improvement': safe_float(shrinkage_info['improvement_ratio'])
                    })
                    
                    print(f"✅ Entry {i+1} completed:")
                    print(f"   - Valid selected triplets: {len(triplet_embeddings)}")
                    print(f"   - Triplet embedding dimension: {triplet_embeddings[0].shape[0]}")
                    print(f"   - Mu shape: {mu.shape}")
                    print(f"   - Sigma shape: {sigma.shape}")
                    print(f"   - Mu norm: {torch.norm(mu_cpu):.4f}")
                    print(f"   - Sigma condition number: {torch.linalg.cond(sigma_cpu):.4f}")
                    print(f"   - Shrinkage method: {shrinkage_info['method']}")
                    print(f"   - Shrinkage parameter: {shrinkage_info['shrinkage_parameter']:.4f}")
                    print(f"   - Condition number improvement: {shrinkage_info['improvement_ratio']:.2f}x")
                else:
                    print(f"[Entry {i+1}] Failed to calculate Mahalanobis parameters")
                
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
                'global_summary_statistics': global_summary,
                'document_statistics': document_stats
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
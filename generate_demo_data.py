import os
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def generate_hdformer_demo_data(num_samples=1000, seq_len=32, output_dir="sample_data"):
    """
    Generates 1,000 synthetic samples following HDFormer data format requirements.
    
    Data Format per sequence:
    {
        'pid': str,                           # Patient/Sample ID
        'tokenized_category_feats': ndarray,  # (n_cat, seq_len)
        'tokenized_float_feats': ndarray,     # (n_float, seq_len)
        'category_feats': ndarray,            # (n_cat, seq_len)
        'float_feats': ndarray,               # (n_float, seq_len)
        'valid_mask': ndarray,                # (seq_len,)
        'time_index': ndarray,                # (seq_len,)
        'dataset_fold10': int                 # Fold assignment 1-10
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    data = []
    
    # Matching the config: 17 categorical features, 41 continuous features
    n_cat = 17
    n_float = 41
    n_category_values = 500
    n_float_values = 256
    
    for i in tqdm(range(num_samples)):
        pid = f"HD{i:05d}"
        
        # Categorical features
        category_feats = np.random.randint(0, n_category_values, size=(n_cat, seq_len)).astype(np.float32)
        tokenized_category_feats = category_feats.copy().astype(np.int32)
        
        # Continuous features
        float_feats = np.random.randn(n_float, seq_len).astype(np.float32)
        
        # Discretize continuous features into tokens (e.g., 256 bins)
        scaled = np.clip((float_feats + 3) / 6 * (n_float_values - 1), 0, n_float_values - 1)
        tokenized_float_feats = scaled.astype(np.int32)
        
        # Valid mask indicating actual timestamps vs padding
        num_valid = np.random.randint(int(seq_len * 0.5), seq_len)
        valid_mask = np.zeros(seq_len, dtype=bool)
        valid_mask[:num_valid] = True
        
        # Missing values mask (-1 for missing)
        cat_mask = np.random.rand(n_cat, seq_len) < 0.1
        category_feats[cat_mask] = -1
        tokenized_category_feats[cat_mask] = -1
        
        float_mask = np.random.rand(n_float, seq_len) < 0.1
        float_feats[float_mask] = -1
        tokenized_float_feats[float_mask] = -1
        
        # Time index
        time_index = np.arange(seq_len, dtype=np.int32)
        
        # 10-fold cross validation split
        dataset_fold10 = np.random.randint(1, 11)
        
        sample_dict = {
            'pid': pid,
            'tokenized_category_feats': tokenized_category_feats,
            'tokenized_float_feats': tokenized_float_feats,
            'category_feats': category_feats,
            'float_feats': float_feats,
            'valid_mask': valid_mask,
            'time_index': time_index,
            'dataset_fold10': dataset_fold10
        }
        data.append(sample_dict)
        
    # Save as DataFrame Pickle
    df = pd.DataFrame(data)
    out_path = os.path.join(output_dir, 'sample.pkl')
    df.to_pickle(out_path)
    print(f"Saved {num_samples} records to {out_path}")
    
    # Generate feature statistics (feat_info.json)
    float_cols_dict = {}
    for i in range(n_float):
        float_cols_dict[f"FLOAT_FEAT_{i}"] = {"mean": 0.0, "std": 1.0}
        
    feat_info = {
        "category_cols": [f"CAT_FEAT_{i}" for i in range(n_cat)],
        "float_cols": float_cols_dict
    }
    
    with open(os.path.join(output_dir, 'feat_info.json'), 'w') as f:
        json.dump(feat_info, f, indent=4)
        
    # Generate float feature list (float_feats.json)
    float_feats_list = [f"FLOAT_FEAT_{i}" for i in range(n_float)]
    with open(os.path.join(output_dir, 'float_feats.json'), 'w') as f:
        json.dump(float_feats_list, f, indent=4)
        
    # Generate PANS-style metadata.pkl
    cat_cols = [f"CAT_FEAT_{i}" for i in range(n_cat)]
    vocab_dict = {col: list(range(n_category_values)) for col in cat_cols}
    metadata = {
        'vocab': vocab_dict,
        'cat_cols': cat_cols,
        'float_cols': float_feats_list
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Successfully generated feat_info.json, float_feats.json and metadata.pkl.")

if __name__ == "__main__":
    generate_hdformer_demo_data()

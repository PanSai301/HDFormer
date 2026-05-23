# HDformer

A unified transformer-based foundation model that supports both self-supervised pretraining and supervised finetuning on sequential data with advanced feature-level masking.

## 🔥 Features

- **Unified Architecture**: Single HDformer model for both pretraining and finetuning
- **Sequential Modeling**: Handles temporal data with timestamp-aware attention
- **Feature-Level Masking**: Advanced pretraining with timestamp-wise feature masking (50% of valid features per timestamp)
- **Multi-Task Learning**: Supports both classification and regression tasks
- **Flexible Data Loading**: Efficient data loading for large-scale training

## 📋 Requirements

### Hardware
- GPU: NVIDIA GPU with CUDA support
- Memory: 16GB+ GPU memory recommended
- Storage: SSD recommended for large datasets

### Software
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+

## 🚀 Quick Start

### Installation

Clone the repository:
```bash
git clone <repository-url>
cd HDformer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run with Demo Data

The repository includes demo data (1000 synthetic samples) ready for immediate testing.

```bash
# Generate demo data if it doesn't exist
python generate_demo_data.py

# Pretraining with demo data
python pretrain.py

# Finetuning with demo data  
python finetune.py
```

## 📊 Data Format Requirements

To train HDformer on your custom data, you need to process your data into the following DataFrame columns:

```python
{
    'pid': str,                           # Patient/Sample ID (unique identifier)
    'tokenized_category_feats': ndarray,  # (n_cat, seq_len) - Discretized categorical features
    'tokenized_float_feats': ndarray,     # (n_float, seq_len) - Discretized continuous features  
    'category_feats': ndarray,            # (n_cat, seq_len) - Original categorical values
    'float_feats': ndarray,               # (n_float, seq_len) - Original continuous values
    'valid_mask': ndarray,                # (seq_len,) - Valid timestamp mask
    'time_index': ndarray,                # (seq_len,) - Time indices
    'dataset_fold10': int                 # Dataset fold for train/val/test split (1-10)
}
```

Feature statistics must be provided in `feat_info.json` and a list of float features in `float_feats.json`.
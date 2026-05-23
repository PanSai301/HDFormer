import json
import argparse
from models.hdformer import HDFormer

def main():
    parser = argparse.ArgumentParser(description="Pretrain HDformer")
    parser.add_argument("--config", type=str, default="configs/pretrain.json", help="Path to config file")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)

    print("Initializing HDformer model for Pretraining...")
    model = HDFormer(config_dict)
    
    print(f"Model instantiated successfully.")
    print("Starting Pretraining loop with feature-level masking (simulated)...")
    
    # In a real scenario, you would initialize DataLoader, PyTorch Lightning Trainer, etc.
    print("Pretraining complete.")

if __name__ == "__main__":
    main()
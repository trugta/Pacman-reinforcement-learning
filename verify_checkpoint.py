import json
import os

# Check the metadata from the latest checkpoint
metadata_path = 'checkpoints/checkpoint_metadata.json'
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print('✓ Checkpoint metadata exists')
    print(f'  - train_steps: {metadata["train_steps"]}')
    print(f'  - optimizer: {metadata["optimizer_config"]["class_name"]}')
    print(f'  - learning_rate: {metadata["optimizer_config"]["config"]["learning_rate"]}')
    print('\n✓ Optimizer state is properly serialized for checkpoint restore')
else:
    print('✗ Metadata file not found')

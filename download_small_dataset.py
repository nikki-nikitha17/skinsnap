# download_small_dataset.py
import tensorflow as tf
import tensorflow_datasets as tfds
import os

print("Downloading a smaller skin dataset for testing...")

# Try to load a smaller medical dataset
try:
    # This dataset is smaller and more reliable
    dataset, info = tfds.load(
        'patch_camelyon',
        split=['train', 'validation', 'test'],
        with_info=True,
        as_supervised=True,
        shuffle_files=True
    )
    
    train_dataset, val_dataset, test_dataset = dataset
    
    print("✅ Dataset loaded successfully!")
    print(f"Dataset info: {info}")
    print(f"Number of training samples: {len(list(train_dataset))}")
    
    # Save dataset info for later use
    with open('dataset_info.txt', 'w') as f:
        f.write(str(info))
        
    print("Dataset ready for training!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative dataset...")
    
    # Fallback to CIFAR-10 for demonstration
    dataset, info = tfds.load(
        'cifar10',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True
    )
    
    train_dataset, val_dataset, test_dataset = dataset
    print("✅ Using CIFAR-10 dataset for demonstration")
    print(f"Number of classes: {info.features['label'].num_classes}")
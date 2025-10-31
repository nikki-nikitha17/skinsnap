# dataset_manager.py
import os
import zipfile
import requests
# Import Kaggle API only when needed to avoid authentication on import
from tqdm import tqdm
import pandas as pd
import shutil

class SkinDatasetManager:
    def __init__(self):
        self.dataset_path = 'skin_datasets'
        self.available_datasets = {
            'ham10000': {
                'name': 'HAM10000',
                'kaggle_url': 'kmader/skin-cancer-mnist-ham10000',
                'description': '10,000 dermatoscopic images of common pigmented skin lesions'
            },
            'dermnet': {
                'name': 'DermNet',
                'kaggle_url': 'shubhamgoel27/dermnet',
                'description': '23 types of skin diseases with images'
            },
            'isic2019': {
                'name': 'ISIC 2019',
                'kaggle_url': 'noulam/tomato',
                'description': 'International Skin Imaging Collaboration 2019'
            }
        }
        
    def download_dataset(self, dataset_name):
        """Download dataset from Kaggle"""
        if dataset_name not in self.available_datasets:
            print(f"❌ Dataset {dataset_name} not available")
            return False
            
        try:
            print(f"📥 Downloading {dataset_name} dataset...")
            print("This may take several minutes depending on your internet connection...")
            # Lazy import and authenticate
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            os.makedirs(self.dataset_path, exist_ok=True)
            api.dataset_download_files(
                self.available_datasets[dataset_name]['kaggle_url'],
                path=self.dataset_path,
                unzip=True
            )
            print(f"✅ {dataset_name} downloaded successfully!")
            
            # Organize the dataset if it's HAM10000
            if dataset_name == 'ham10000':
                self._organize_ham10000()
                
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nTo enable Kaggle downloads:")
            print("1) Create an API token at https://www.kaggle.com/settings/account")
            print("2) Download kaggle.json")
            print("3) Place it at C:/Users/" + os.getlogin() + "/.kaggle/kaggle.json")
            print("   or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            return False
    
    def _organize_ham10000(self):
        """Organize HAM10000 dataset into train/validation structure"""
        print("🔄 Organizing HAM10000 dataset...")
        
        # Create directory structure
        for split in ['train', 'validation', 'test']:
            os.makedirs(f'{self.dataset_path}/{split}', exist_ok=True)
        
        # For HAM10000, we would typically use the CSV metadata to organize
        # This is a simplified version
        print("💡 HAM10000 dataset downloaded. Manual organization may be needed.")
        print("📊 Use the HAM_metadata.csv file to organize images by class")
    
    def setup_demo_data(self):
        """Create a realistic demo dataset structure"""
        print("🔄 Setting up demo data structure...")
        
        # Create demo directory structure
        demo_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        for split in ['train', 'validation']:
            for class_name in demo_classes:
                os.makedirs(f'{self.dataset_path}/{split}/{class_name}', exist_ok=True)
        
        # Create a README with instructions
        readme_content = """
# Skin Disease Dataset

This is a demo dataset structure. To use real data:

1. Download real datasets using download_data.py
2. Or manually organize your skin disease images in this structure:

skin_datasets/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...

Recommended datasets:
- HAM10000: 10,000 skin lesion images
- DermNet: 23 skin disease categories
- ISIC: International Skin Imaging Collaboration datasets
"""
        
        with open(f'{self.dataset_path}/README.md', 'w') as f:
            f.write(readme_content)
        
        print("✅ Demo dataset structure created!")
        print("📁 Check skin_datasets/README.md for setup instructions")
        return True
    
    def list_datasets(self):
        """List available datasets"""
        print("📚 Available Skin Disease Datasets:")
        for key, info in self.available_datasets.items():
            print(f"  🔸 {key}: {info['name']}")
            print(f"     {info['description']}")
            print(f"     Kaggle: {info['kaggle_url']}\n")
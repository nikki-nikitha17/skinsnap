# download_data.py
from dataset_manager import SkinDatasetManager
import os

def download_kaggle_data():
    """Download skin disease datasets from Kaggle"""
    print("🎯 Skin Disease Dataset Downloader")
    print("=" * 50)
    
    manager = SkinDatasetManager()
    
    print("Available datasets:")
    for key, info in manager.available_datasets.items():
        print(f"  🔹 {key}: {info['name']} - {info['description']}")
    
    print("\n📥 Downloading HAM10000 dataset (recommended)...")
    
    success = manager.download_dataset('ham10000')
    
    if success:
        print("\n✅ Dataset downloaded successfully!")
        print("📁 Files are in 'skin_datasets/' directory")
        
        # List downloaded files
        if os.path.exists('skin_datasets'):
            print("\n📋 Downloaded files:")
            for root, dirs, files in os.walk('skin_datasets'):
                level = root.replace('skin_datasets', '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}📁 {os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Show first 10 files
                    print(f'{subindent}📄 {file}')
                if len(files) > 10:
                    print(f'{subindent}... and {len(files) - 10} more files')
    else:
        print("\n❌ Download failed.")
        print("\n💡 Setup Kaggle API:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Click on your profile → Account")
        print("3. Scroll to 'API' section → Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print("5. Place kaggle.json in ~/.kaggle/ directory")
        print("6. Run: chmod 600 ~/.kaggle/kaggle.json")
        
        print("\n🔄 Setting up demo data structure instead...")
        manager.setup_demo_data()

def setup_data_directories():
    """Create proper directory structure for training"""
    print("\n🔧 Setting up data directories...")
    
    directories = [
        'skin_datasets/train',
        'skin_datasets/validation', 
        'skin_datasets/test',
        'models',
        'training_logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n📁 Directory structure ready for training!")

if __name__ == "__main__":
    download_kaggle_data()
    setup_data_directories()
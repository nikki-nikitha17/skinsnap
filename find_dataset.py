# find_dataset.py
import os

print("Searching for the dataset...")

# Check current directory
print(f"Current directory: {os.getcwd()}")

# List all files and folders in current directory
print("\n📁 Everything in current folder:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"📁 {item}/")
    else:
        print(f"📄 {item}")

# Search for any folder containing 'skin' or 'ham'
print("\n🔍 Searching for dataset folders...")
for item in os.listdir('.'):
    if os.path.isdir(item) and ('skin' in item.lower() or 'ham' in item.lower()):
        print(f"✅ Found potential dataset: {item}/")
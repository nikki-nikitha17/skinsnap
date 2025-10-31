# SkinSnap – AI Skin Disease Analyzer

A Flask web app that analyzes skin images and returns the single most likely skin condition with a confidence score. Includes 100+ condition labels, OpenCV-based intelligent analysis, and Kaggle dataset integration hooks.

## Features
- Single top prediction with confidence
- 100+ dermatology labels and condition info page
- OpenCV-powered analysis (works without TensorFlow)
- Kaggle dataset download support (HAM10000, DermNet)

## Requirements
- Python 3.11+
- pip

Install dependencies:

```bash
pip install -r requirments.txt
```

## Kaggle Setup (Required for dataset download)
1. Create API token at https://www.kaggle.com/settings/account
2. Download `kaggle.json`
3. Place it at:
   - Windows: `C:\Users\<YOUR_USER>\.kaggle\kaggle.json`
   - Or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`

## Download a Dataset
From Python shell or a script:

```python
from dataset_manager import SkinDatasetManager
SkinDatasetManager().download_dataset('ham10000')  # or 'dermnet'
```

## Run the App
```bash
python app.py
```
Visit `http://localhost:5000` and upload a photo.

## Project Structure
- `app.py` – Flask server and routes
- `real_classifier.py` – Classifier (smart mock by default; TF optional)
- `dataset_manager.py` – Kaggle dataset downloader (lazy auth)
- `templates/` – UI templates
- `skin_datasets/` – Datasets (gitignored)
- `static/uploads/` – Uploaded images (gitignored)

## GitHub
After you create a public repo `skinsnap` under `nikki-nikitha17`, push:

```bash
git branch -M main
git remote add origin https://github.com/nikki-nikitha17/skinsnap.git
git push -u origin main
```

If prompted for a password, use a GitHub Personal Access Token with `repo` scope.

## Notes
- If TensorFlow is not installed, the app uses OpenCV-based intelligent analysis to produce realistic predictions.
- To train a real model, install TensorFlow and integrate a training pipeline in `train_model.py`.

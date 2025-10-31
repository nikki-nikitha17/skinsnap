# train_model.py
import os
import tensorflow as tf
from real_classifier import SkinDiseaseClassifier
from dataset_manager import SkinDatasetManager
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def setup_environment():
    """Setup directories and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    os.makedirs('skin_datasets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)
    
    print("✅ Environment setup completed")

def download_and_prepare_data():
    """Download and prepare Kaggle dataset"""
    print("📥 Downloading dataset...")
    
    manager = SkinDatasetManager()
    
    # Try to download HAM10000 dataset (most popular skin cancer dataset)
    success = manager.download_dataset('ham10000')
    
    if not success:
        print("❌ Kaggle download failed. Setting up demo data structure...")
        manager.setup_demo_data()
        return 'demo'
    
    return 'ham10000'

def load_and_preprocess_data(dataset_name):
    """Load and preprocess the dataset"""
    print(f"🔄 Loading and preprocessing {dataset_name} dataset...")
    
    if dataset_name == 'demo':
        print("💡 Using demo mode - no real training data available")
        return None, None, None, None
    
    # For HAM10000 dataset
    dataset_path = 'skin_datasets'
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found. Please download first.")
        return None, None, None, None
    
    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    try:
        # Load training data
        train_generator = datagen.flow_from_directory(
            os.path.join(dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        validation_generator = datagen.flow_from_directory(
            os.path.join(dataset_path, 'validation'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"✅ Data loaded: {train_generator.samples} training samples, {validation_generator.samples} validation samples")
        return train_generator, validation_generator, train_generator.class_indices, train_generator.num_classes
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Dataset structure might be different than expected")
        return None, None, None, None

def train_real_model():
    """Train the real model with actual data"""
    print("🚀 Starting real model training...")
    
    # Setup environment
    setup_environment()
    
    # Download data
    dataset_name = download_and_prepare_data()
    
    # Load data
    train_gen, val_gen, class_indices, num_classes = load_and_preprocess_data(dataset_name)
    
    if train_gen is None:
        print("❌ Cannot proceed without training data")
        return False
    
    # Initialize classifier
    classifier = SkinDiseaseClassifier(use_real_model=False)
    
    # Build model architecture
    classifier.build_model()
    
    # Update model for actual number of classes in dataset
    if num_classes != len(classifier.class_names):
        print(f"🔄 Adjusting model for {num_classes} classes (from dataset)")
        # You would adjust the last layer here based on actual dataset classes
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/skin_model_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger('training_logs/training_history.csv')
    ]
    
    print("🏋️ Starting model training...")
    
    # Train the model
    history = classifier.model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    classifier.model.save('models/skin_model_final.h5')
    print("✅ Model training completed and saved!")
    
    # Plot training history
    plot_training_history(history)
    
    return True

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_logs/training_history.png')
    plt.show()

def evaluate_model():
    """Evaluate the trained model"""
    print("📊 Evaluating model...")
    
    if not os.path.exists('models/skin_model_final.h5'):
        print("❌ No trained model found. Train first.")
        return
    
    classifier = SkinDiseaseClassifier(use_real_model=True)
    
    # Load test data (you would need to implement this)
    # test_generator = ... 
    
    # Evaluate
    # test_loss, test_accuracy = classifier.model.evaluate(test_generator)
    # print(f"📈 Test Accuracy: {test_accuracy:.2%}")
    # print(f"📈 Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    print("🎯 Skin Disease Model Training Script")
    print("=" * 50)
    
    # Train the model
    success = train_real_model()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Model saved in 'models/' directory")
        print("📊 Training logs saved in 'training_logs/' directory")
        
        # Evaluate the model
        evaluate_model()
    else:
        print("\n❌ Training failed. Check the error messages above.")
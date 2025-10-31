# app.py - Enhanced with Real Dataset Support
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import cv2
import numpy as np
from real_classifier import SkinDiseaseClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize classifier
classifier = SkinDiseaseClassifier(use_real_model=False)

@app.route('/')
def index():
    return render_template('index.html', total_diseases=len(classifier.class_names))

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = f"skin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            disease, confidence = classifier.predict(filepath)
            
            # Get comprehensive disease information
            disease_info = classifier.get_disease_info(disease)
            
            # Prepare single result response
            result = {
                'disease': disease,
                'confidence': confidence,
                'category': disease_info['category'],
                'severity': disease_info['severity'],
                'description': disease_info['description'],
                'recommendation': disease_info['recommendation'],
                'image_url': f'/static/uploads/{filename}',
                'common': disease_info.get('common', False)
            }
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Analysis failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'})

@app.route('/diseases')
def diseases_list():
    """Show all covered diseases with categories"""
    diseases_by_category = {}
    for disease, info in classifier.disease_database.items():
        category = info['category']
        if category not in diseases_by_category:
            diseases_by_category[category] = []
        diseases_by_category[category].append({
            'name': disease,
            'severity': info['severity'],
            'common': info['common']
        })
    
    return render_template('diseases.html', 
                         categories=diseases_by_category,
                         total_diseases=len(classifier.class_names))

@app.route('/dataset/setup', methods=['POST'])
def setup_dataset():
    """Endpoint to setup real dataset (for admin use)"""
    try:
        success = classifier.dataset_manager.setup_demo_data()
        if success:
            return jsonify({'success': True, 'message': 'Demo dataset setup completed'})
        else:
            return jsonify({'success': False, 'message': 'Dataset setup failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Enhanced SkinSnap Web Server...")
    print(f"📋 Comprehensive Coverage: {len(classifier.class_names)} skin conditions")
    print("🏥 Categories: Inflammatory, Infectious, Benign, Malignant, Pigment, etc.")
    print("🌐 Open: http://localhost:5000 in your browser")
    print("💡 Note: Using intelligent analysis with comprehensive disease database")
    app.run(debug=True, host='0.0.0.0', port=5000)
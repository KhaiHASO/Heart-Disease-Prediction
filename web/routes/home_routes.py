from flask import Blueprint, render_template, request, jsonify
import numpy as np
import os
import joblib

main = Blueprint('main', __name__)


# Load the SVM model
def load_model():
    base_path = os.path.dirname(__file__)
    assets_path = os.path.join(base_path, '..', 'assets')
    
    model_path = os.path.join(assets_path, 'best_overall_model_random_forest.pkl')
    scaler_path = os.path.join(assets_path, 'standard_scaler.pkl')
    
    try:
        # Load the SVM model
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
        
        # Try to load preprocessor if it exists
        scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = joblib.load(f)
                print("scaler loaded successfully")
            except Exception as e:
                print(f"Failed to load preprocessor: {e}")
        else:
            print("Preprocessor file not found, will use raw features")

        return model, scaler
        
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Load model and make prediction
model, scaler = load_model()

@main.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        features = {
            'age': int(data['age']),       
            'sex': int(data['sex']),        
            'cp': int(data['cp']),          
            'trestbps': int(data['trestbps']), 
            'chol': int(data['chol']),
            'fbs': int(data['fbs']),  
            'restecg': int(data['restecg']),
            'thalach': int(data['thalach']),
            'exang': int(data['exang']),    
            'oldpeak': float(data['oldpeak']), 
            'slope': int(data['slope']),     
            'ca': int(data['ca']),     
            'thal': int(data['thal'])       
        }
        
        print("Extracted features:", features)

        
        if model is None:
            return jsonify({'error': 'Model không thể tải được'}), 500
        
        # Convert to numpy array and reshape for prediction
        features_array = preprocess_heart_predict(features, scaler)

        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Convert prediction to result
        result = {
            'prediction': int(prediction),
            'risk_level': 'Mắc bệnh' if prediction == 1 else 'Không mắc bệnh',
            'risk_class': 'risk-high' if prediction == 1 else 'risk-low',
            'confidence': float(probability[1] if probability is not None and len(probability) > 1 else 0.5),
            'patient_data': data,
            'features': features,
            'probability_no_disease': float(probability[0]),
            'probability_disease': float(probability[1]),
        }
        
        print("PREDICTION RESULT:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability - No Disease: {result['probability_no_disease']:.2%}")
        print(f"  Probability - Disease: {result['probability_disease']:.2%}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': f'Lỗi trong quá trình phân tích: {str(e)}'}), 500

def preprocess_heart_predict(features, scaler):

    # Apply preprocessing if preprocessor exists
    if scaler is not None:
        try:
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
            # Extract features in correct order
            features = [features[feature] for feature in feature_names]
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            return scaler.transform(features_array)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            # Return original features if preprocessing fails
            return features_array
    
    # Return original features if no preprocessor
    return features_array



# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, redirect
from flask_restx import Api, Resource, fields, reqparse
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def index():
    """Redirect root to Swagger UI."""
    return redirect("/swagger/")

api = Api(app, 
          version='1.0', 
          title='EzaSmart Hydroponics API',
          description='Hydroponics Management System API',
          doc='/swagger/')

# Load models and encoders
try:
    rf_model = joblib.load('Results/random_forest_model.pkl')
    scaler = joblib.load('Results/feature_scaler.pkl')
    le_crop = joblib.load('Results/crop_encoder.pkl')
    le_action = joblib.load('Results/action_encoder.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = None

# Parser with defaults so Swagger "Try it out" is pre-filled
predict_parser = reqparse.RequestParser()
predict_parser.add_argument('crop_id', type=str, required=True, location='json',
    default='Lettuce', help='Crop type: Lettuce, Peppers, or Tomatoes')
predict_parser.add_argument('ph_level', type=float, required=True, location='json',
    default=6.2, help='pH level (4.0-8.5)')
predict_parser.add_argument('ec_value', type=float, required=True, location='json',
    default=1.8, help='EC value in mS/cm (0.5-4.0)')
predict_parser.add_argument('ambient_temp', type=float, required=True, location='json',
    default=24.0, help='Ambient temperature in Celsius (15-32)')

# Request model for docs (same shape as parser)
prediction_model = api.model('PredictionRequest', {
    'crop_id': fields.String(required=True, description='Crop type: Lettuce, Peppers, or Tomatoes'),
    'ph_level': fields.Float(required=True, description='pH level (4.0-8.5)'),
    'ec_value': fields.Float(required=True, description='EC value in mS/cm (0.5-4.0)'),
    'ambient_temp': fields.Float(required=True, description='Ambient temperature in Celsius (15-32)')
})

response_model = api.model('PredictionResponse', {
    'predicted_action': fields.String(description='Recommended action'),
    'confidence': fields.Float(description='Prediction confidence score'),
    'all_probabilities': fields.Raw(description='Probabilities for all actions'),
    'recommendation': fields.String(description='Detailed recommendation')
})

@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_parser)
    @api.marshal_with(response_model)
    def post(self):
        """Get hydroponic management recommendation based on sensor readings"""
        if rf_model is None:
            return {'error': 'Model not loaded'}, 500
        
        data = predict_parser.parse_args()
        
        # Validate inputs
        crop = data.get('crop_id')
        ph = data.get('ph_level')
        ec = data.get('ec_value')
        temp = data.get('ambient_temp')
        
        if crop not in ['Lettuce', 'Peppers', 'Tomatoes']:
            return {'error': 'Invalid crop_id. Must be Lettuce, Peppers, or Tomatoes'}, 400
        
        if not (4.0 <= ph <= 8.5):
            return {'error': 'pH level must be between 4.0 and 8.5'}, 400
        
        if not (0.5 <= ec <= 4.0):
            return {'error': 'EC value must be between 0.5 and 4.0 mS/cm'}, 400
        
        if not (15 <= temp <= 32):
            return {'error': 'Temperature must be between 15 and 32 C'}, 400
        
        # Encode crop
        crop_encoded = le_crop.transform([crop])[0]
        
        # Prepare features
        features = np.array([[crop_encoded, ph, ec, temp]])
        
        # Predict
        prediction = rf_model.predict(features)[0]
        probabilities = rf_model.predict_proba(features)[0]
        
        # Decode action
        predicted_action = le_action.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        all_probs = {
            le_action.classes_[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Generate recommendation
        recommendations = {
            'Add_pH_Down': f'pH level ({ph:.1f}) is too high. Add pH Down solution to lower pH to optimal range (5.5-6.5).',
            'Add_pH_Up': f'pH level ({ph:.1f}) is too low. Add pH Up solution to raise pH to optimal range (5.5-6.5).',
            'Add_Nutrients': f'EC value ({ec:.2f} mS/cm) is too low. Add nutrient solution to increase nutrient concentration.',
            'Dilute': f'EC value ({ec:.2f} mS/cm) is too high. Dilute the solution with fresh water to reduce concentration.',
            'Maintain': f'Current conditions are optimal. pH: {ph:.1f}, EC: {ec:.2f} mS/cm. Continue monitoring.'
        }
        
        recommendation = recommendations.get(predicted_action, 'Monitor your system closely.')
        
        return {
            'predicted_action': predicted_action,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'recommendation': recommendation
        }

@api.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'model_loaded': rf_model is not None
        }

# Inject example into Swagger spec so "Try it out" is pre-filled
_predict_example = {
    'crop_id': 'Lettuce',
    'ph_level': 6.2,
    'ec_value': 1.8,
    'ambient_temp': 24.0
}

@app.after_request
def _inject_swagger_example(response):
    if response.content_type != 'application/json':
        return response
    try:
        data = response.get_json()
        if not data or data.get('swagger') != '2.0' or 'paths' not in data:
            return response
        path = data.get('paths', {}).get('/predict', {})
        for method in ('post', 'put', 'patch'):
            if method not in path:
                continue
            params = path[method].get('parameters') or []
            for p in params:
                if p.get('in') == 'body' and 'schema' in p:
                    p['schema']['example'] = _predict_example
                    break
        response.set_data(__import__('json').dumps(data))
    except Exception:
        pass
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

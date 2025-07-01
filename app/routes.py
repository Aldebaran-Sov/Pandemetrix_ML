# app/routes.py
from flask_restx import Api, Resource, Namespace
from flask import request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from app.models import main as model_trainer

# Variables globales
model = None
metadata = None

def load_model_and_metadata():
    """Charge le modèle et métadonnées"""
    global model, metadata
    
    model_path = "models/covid_polynomial_model.pkl"
    metadata_path = "models/model_metadata.json"
    
    try:
        print(f"Chargement du modèle: {model_path}")
        model = joblib.load(model_path)
        
        print(f"Chargement métadonnées: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        print("Modèle et métadonnées chargés")
        return True
        
    except FileNotFoundError as e:
        print(f"Fichier manquant: {e}")
        return False
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return False

def prepare_features_for_prediction(data):
    """Prépare les features pour prédiction"""
    if not metadata:
        raise ValueError("Métadonnées du modèle non chargées")
    
    # Features requises (tes 5 features)
    required_base_features = ["date", "new_cases", "people_vaccinated", "new_tests", "daily_occupancy_hosp"]
    expected_features = metadata["features"]["all_features"]
    
    # Features de base
    base_features = {
        "date": pd.Timestamp(data["date"]).timestamp(),
        "new_cases": float(data["new_cases"]),
        "people_vaccinated": float(data["people_vaccinated"]),
        "new_tests": float(data["new_tests"]),
        "daily_occupancy_hosp": float(data["daily_occupancy_hosp"])
    }
    
    # One-hot encoding pour le pays
    country = data["country"]
    country_feature = f"country1_{country}"
    
    # Validation du pays
    if country_feature not in expected_features:
        available_countries = metadata["countries_supported"]
        raise ValueError(f"Pays '{country}' non supporté. Pays disponibles: {available_countries}")
    
    # Créer toutes les features pays (one-hot)
    for country_name in metadata["countries_supported"]:
        country_col = f"country1_{country_name}"
        base_features[country_col] = 1 if country_col == country_feature else 0
    
    # Créer DataFrame dans l'ordre des features d'entraînement
    df = pd.DataFrame([base_features])
    df = df[expected_features]  # Réorganiser selon l'ordre d'entraînement
    
    return df

def validate_input_data(data):
    """Valide les données d'entrée"""
    errors = []
    
    required_fields = ["country", "date", "new_cases"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Champ manquant: {field}")
    
    if "new_cases" in data:
        try:
            new_cases = float(data["new_cases"])
            if new_cases < 0:
                errors.append("new_cases doit être positif")
        except (ValueError, TypeError):
            errors.append("new_cases doit être un nombre")
    
    if "date" in data:
        try:
            datetime.strptime(data["date"], "%Y-%m-%d")
        except ValueError:
            errors.append("Format de date invalide. Utilisez YYYY-MM-DD")
    
    return len(errors) == 0, errors

def init_routes(app):
    """Initialise les routes Swagger uniquement"""
    
    # Configuration Swagger
    api = Api(
        app,
        version='1.0.0',
        title='Pandemetrix ML API',
        description='API de prédiction COVID-19 avec Machine Learning pour l\'OMS',
        doc='/',  # Documentation à la racine
        prefix='/api/v1'
    )
    
    # Import des modèles Swagger
    from app.swagger_models import create_swagger_models
    models = create_swagger_models(api)
    
    # Namespace pour organiser les endpoints
    covid_ns = Namespace('covid', description='Prédictions COVID-19')
    api.add_namespace(covid_ns)
    
    # ENDPOINTS SWAGGER
    @covid_ns.route('/info')
    class ApiInfo(Resource):
        @api.marshal_with(models['api_info'])
        @api.doc('get_api_info', description='Informations générales sur l\'API')
        def get(self):
            """Informations générales sur l'API"""
            return {
                "name": "Pandemetrix ML API",
                "version": "1.0.0",
                "description": "API de prédiction COVID-19 utilisant des modèles de Machine Learning",
                "organization": "OMS - Organisation Mondiale de la Santé",
                "model_loaded": model is not None and metadata is not None,
                "model_version": metadata.get("model_info", {}).get("version", "unknown") if metadata else "unknown",
                "available_endpoints": [
                    "/api/v1/covid/info",
                    "/api/v1/covid/health",
                    "/api/v1/covid/countries",
                    "/api/v1/covid/model-info",
                    "/api/v1/covid/predict",
                    "/api/v1/covid/predict-batch"
                ]
            }
    
    @covid_ns.route('/health')
    class HealthCheck(Resource):
        @api.doc('health_check', description='Vérification de l\'état de l\'API')
        def get(self):
            """Vérification de l'état de l'API"""
            return {
                "status": "healthy" if (model is not None and metadata is not None) else "degraded",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": model is not None,
                "metadata_loaded": metadata is not None,
                "ready_for_predictions": model is not None and metadata is not None
            }
    
    @covid_ns.route('/countries')
    class Countries(Resource):
        @api.doc('get_countries', description='Liste des pays supportés par le modèle')
        def get(self):
            """Liste des pays supportés par le modèle"""
            if not metadata:
                return {"error": "Métadonnées non chargées"}, 500
            
            # Récupérer directement depuis les métadonnées
            countries = metadata.get("countries_supported", [])
            
            # Si pas trouvé, fallback sur les features
            if not countries:
                features = metadata.get("training_info", {}).get("features", [])
                countries = []
                for feature in features:
                    if feature.startswith("country1_"):
                        country_name = feature.replace("country1_", "")
                        countries.append(country_name)
            
            return {
                "countries": sorted(countries),
                "total_countries": len(countries),
                "sample_countries": countries[:5] if countries else [],
                "model_performance": {
                    "r2_score": round(metadata.get("performance", {}).get("test_r2", 0), 3),
                    "countries_trained": len(countries)
                },
                "note": "Utilisez exactement ces noms de pays dans vos requêtes"
            }
        
    @covid_ns.route('/model-info')
    class ModelInfo(Resource):
        @api.doc('get_model_info', description='Informations détaillées sur le modèle ML')
        def get(self):
            """Informations détaillées sur le modèle ML"""
            if not metadata:
                return {"error": "Métadonnées non chargées"}, 500
                
            return metadata
    
    @covid_ns.route('/predict-batch')
    class PredictBatch(Resource):
        @api.expect(models['batch_input'])
        @api.doc('make_batch_prediction', description='Effectue des prédictions multiples')
        def post(self):
            """Effectue des prédictions multiples"""
            try:
                if not model or not metadata:
                    return {"error": "Modèle ou métadonnées non chargés"}, 500
                
                data = request.get_json()
                if not data or "predictions" not in data:
                    return {"error": "Format invalide. Utilisez: {'predictions': [...]}"}, 400
                
                predictions_list = data["predictions"]
                if not isinstance(predictions_list, list):
                    return {"error": "Le champ 'predictions' doit être une liste"}, 400
                
                results = []
                errors = []
                
                for i, pred_data in enumerate(predictions_list):
                    try:
                        is_valid, validation_errors = validate_input_data(pred_data)
                        if not is_valid:
                            errors.append({"index": i, "errors": validation_errors})
                            continue
                        
                        features_df = prepare_features_for_prediction(pred_data)
                        prediction = model.predict(features_df)[0]
                        prediction = max(0, prediction)
                        
                        results.append({
                            "index": i,
                            "country": pred_data["country"],
                            "date": pred_data["date"],
                            "new_deaths_predicted": round(prediction, 2)
                        })
                        
                    except Exception as e:
                        errors.append({"index": i, "error": str(e)})
                
                return {
                    "successful_predictions": len(results),
                    "failed_predictions": len(errors),
                    "results": results,
                    "errors": errors if errors else None,
                    "model_version": metadata["model_info"]["version"],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": f"Erreur lors de la prédiction batch: {str(e)}"}, 500
            
    @covid_ns.route('/train_model')
    class TrainModel(Resource):
        def get(self):
            """Entraînement du modèle"""
            try:
                # Appel de la fonction main() qui lance la pipeline
                model_trainer()
                return {"message": "Modèle entraîné avec succès"}, 200
            except Exception as e:
                return {"error": str(e)}, 500
                

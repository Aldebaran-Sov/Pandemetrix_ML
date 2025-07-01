import pandas as pd
import numpy as np
from flask import request
from flask_restx import Api, Resource, Namespace
from datetime import datetime

# Importer les nouvelles fonctions de service
from .services import (
    load_model_and_metadata,
    prepare_features_for_prediction,
    validate_input_data
)

# Charger le modèle et les métadonnées au démarrage
model, metadata = load_model_and_metadata()


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
                countries = sorted(list(set(
                    [f.replace('country_', '') for f in features if f.startswith('country_')]
                )))

            return {"countries": countries}

    @covid_ns.route('/model-info')
    class ModelInfo(Resource):
        @api.doc('get_model_info', description='Informations détaillées sur le modèle')
        def get(self):
            """Informations détaillées sur le modèle"""
            if not metadata:
                return {"error": "Métadonnées non chargées"}, 500

            return metadata.get("model_info", {})

    @covid_ns.route('/predict')
    class Predict(Resource):
        @api.expect(models['prediction_input'])
        @api.doc('predict', description='Faire une seule prédiction')
        def post(self):
            """Faire une seule prédiction"""
            if not model or not metadata:
                return {"error": "Modèle non chargé"}, 503

            data = request.json

            # Valider les données
            error = validate_input_data(data, metadata)
            if error:
                return {"error": error}, 400

            # Préparer les caractéristiques
            features = prepare_features_for_prediction(data, metadata)

            # Faire la prédiction
            prediction = model.predict(features)

            return {
                "prediction_input": data,
                "predicted_cases": int(max(0, prediction[0]))
            }

    @covid_ns.route('/predict-batch')
    class PredictBatch(Resource):
        @api.expect(models['prediction_input_batch'])
        @api.doc('predict_batch', description='Faire des prédictions par lot')
        def post(self):
            """Faire des prédictions par lot"""
            if not model or not metadata:
                return {"error": "Modèle non chargé"}, 503

            data_list = request.json.get('predictions', [])

            # Valider les données pour chaque élément du lot
            error = validate_input_data(data_list, metadata)
            if error:
                return {"error": error}, 400

            # Préparer les caractéristiques
            features = prepare_features_for_prediction(data_list, metadata)

            # Prédictions par lot
            predictions = model.predict(features)

            results = []
            for i, data in enumerate(data_list):
                results.append({
                    "prediction_input": data,
                    "predicted_cases": int(max(0, predictions[i]))
                })

            return {"predictions": results}

    # Endpoint pour recharger le modèle
    @covid_ns.route('/reload-model')
    class ReloadModel(Resource):
        @api.doc('reload_model', description='Recharge le modèle et les métadonnées')
        def post(self):
            """Recharge le modèle et les métadonnées à la demande"""
            global model, metadata
            model, metadata = load_model_and_metadata()
            if model and metadata:
                return {"message": "Modèle rechargé avec succès"}, 200
            else:
                return {"error": "Échec du rechargement du modèle"}, 500

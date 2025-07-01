import joblib
import pandas as pd
import numpy as np
from flask import current_app

model = None
metadata = None

def load_model_and_metadata():
    """Charge le modèle et les métadonnées depuis les fichiers."""
    global model, metadata
    if model is None or metadata is None:
        try:
            model = joblib.load('model/covid_prediction_model.joblib')
            metadata = joblib.load('model/covid_prediction_metadata.joblib')
            current_app.logger.info("Modèle et métadonnées chargés avec succès.")
        except FileNotFoundError:
            model, metadata = None, None
            current_app.logger.error("Fichiers du modèle ou des métadonnées non trouvés.")
        except Exception as e:
            model, metadata = None, None
            current_app.logger.error(f"Erreur lors du chargement du modèle : {e}")
    return model, metadata

def prepare_features_for_prediction(data, metadata):
    """Prépare les caractéristiques pour la prédiction."""
    features_list = metadata.get("training_info", {}).get("features", [])
    
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)

    # Conversion des dates et création de caractéristiques temporelles
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        df = df.drop(columns=['date'])

    # One-Hot Encoding pour les pays
    if 'country' in df.columns:
        all_countries = [f.split('_')[-1] for f in features_list if f.startswith('country_')]
        df['country'] = pd.Categorical(df['country'], categories=all_countries)
        df = pd.get_dummies(df, columns=['country'], prefix='country')

    # Assurer que toutes les colonnes du modèle sont présentes
    model_features_df = pd.DataFrame(columns=features_list)
    combined_df = pd.concat([model_features_df, df], sort=False)
    
    # Remplir les colonnes manquantes avec 0
    for col in features_list:
        if col not in combined_df.columns:
            combined_df[col] = 0
    
    # Remplacer les NaN par 0 (ou une autre stratégie si nécessaire)
    combined_df = combined_df.fillna(0)
    
    # Assurer le bon ordre des colonnes
    final_features = combined_df[features_list]

    return final_features


def validate_input_data(data, metadata):
    """Valide les données d'entrée pour la prédiction."""
    required_fields = {"country", "date"}
    
    if isinstance(data, dict):
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return f"Champs manquants : {', '.join(missing_fields)}"
    elif isinstance(data, list):
        for item in data:
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                return f"Champs manquants dans un des objets : {', '.join(missing_fields)}"
    else:
        return "Le format des données doit être un objet JSON ou une liste d'objets."

    # Validation du pays
    supported_countries = metadata.get("countries_supported", [])
    
    def check_country(country):
        if country not in supported_countries:
            return f"Le pays '{country}' n'est pas supporté."
        return None

    if isinstance(data, dict):
        error = check_country(data.get("country"))
        if error: return error
    else: # list
        for item in data:
            error = check_country(item.get("country"))
            if error: return error
            
    return None

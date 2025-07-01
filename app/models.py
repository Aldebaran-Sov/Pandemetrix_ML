import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime
from app.db import get_db

def load_and_merge_data():
    """
    Charge et merge les données depuis les fichiers CSV sources
    """
    print("Chargement et fusion des données sources...")
    
    try:
        db = get_db()  # Utilise la fonction de connexion à la base de données

        # Charger les collections en DataFrames
        print("   Lecture de la collection cases_deaths...")
        df_cases = pd.DataFrame(list(db.cases_deaths.find()))

        print("   Lecture de la collection vaccinations_global...")
        df_vaccines = pd.DataFrame(list(db.vaccinations_global.find()))

        print("   Lecture de la collection hospital...")
        df_hospital = pd.DataFrame(list(db.hospital.find()))

        print("   Lecture de la collection testing...")
        df_testing = pd.DataFrame(list(db.testing.find()))

        # Supprimer l'ObjectId si présent
        for df in [df_cases, df_vaccines, df_hospital, df_testing]:
            df.drop(columns=["_id"], inplace=True, errors="ignore")
        
        # Fusion séquentielle
        print("Fusion des datasets...")
        df_merge = pd.merge(df_cases, df_vaccines, on=["country", "date"], how="inner")
        print(f"   Après fusion vaccins: {df_merge.shape[0]} lignes")
        
        df_merge = pd.merge(df_merge, df_hospital, on=["country", "date"], how="inner")
        print(f"   Après fusion hospital: {df_merge.shape[0]} lignes")
        
        df_merge = pd.merge(df_merge, df_testing, on=["country", "date"], how="inner")
        print(f"   Après fusion testing: {df_merge.shape[0]} lignes")
        
        # Sauvegarder le résultat
        os.makedirs("data/raw", exist_ok=True)
        df_merge.to_csv("data/raw/merged_data.csv", index=False)
        
        print(f"Données fusionnées et sauvegardées: {df_merge.shape[0]} lignes, {df_merge.shape[1]} colonnes")
        print(f"Pays uniques: {df_merge['country'].nunique()}")
        
        return df_merge
        
    except FileNotFoundError as e:
        print(f"ERREUR - Collection manquante: {e}")
        print("Assure-toi d'avoir importé correctement les collections manquantes")
        raise
    except Exception as e:
        print(f"ERREUR lors du chargement: {e}")
        raise

def prepare_data(df_merge):
    """
    Prépare et encode les données
    """
    print("Préparation et encodage des données...")
    
    # Vérifier les colonnes disponibles
    print(f"   Colonnes disponibles: {list(df_merge.columns)}")
    
    # FEATURES REQUISES UNIQUEMENT
    required_features = ["date", "new_cases", "people_vaccinated", "new_tests", "daily_occupancy_hosp"]
    target = "new_deaths"
    
    # Vérifier que toutes les features requises sont présentes
    missing = [f for f in required_features + [target] if f not in df_merge.columns]
    if missing:
        print(f"ERREUR - Colonnes manquantes: {missing}")
        print(f"Colonnes disponibles: {list(df_merge.columns)}")
        raise ValueError(f"Colonnes manquantes: {missing}")
    
    # Sélectionner UNIQUEMENT les features nécessaires + target + country
    keep_cols = required_features + [target, "country"]
    data_filtered = df_merge[keep_cols].copy()
    
    print(f"   Features sélectionnées: {required_features}")
    print(f"   Target: {target}")
    
    # Encodage des pays avec one-hot
    data_encoded = pd.get_dummies(data_filtered, columns=["country"], prefix="country1")
    
    # Nettoyage
    print(f"   Avant nettoyage NaN: {data_encoded.shape[0]} lignes")
    data_encoded.dropna(inplace=True)  # Supprime lignes avec NaN
    print(f"   Après nettoyage NaN: {data_encoded.shape[0]} lignes")
    
    # Sauvegarder
    data_encoded.to_csv("data/raw/merged_data_encoded.csv", index=False)
    
    print(f"Données encodées: {data_encoded.shape[0]} lignes, {data_encoded.shape[1]} colonnes")
    
    # Afficher les pays détectés
    country_cols = [col for col in data_encoded.columns if col.startswith("country1_")]
    print(f"Pays encodés: {len(country_cols)} pays")
    print(f"   Exemples: {country_cols[:5]}...")
    
    return data_encoded

def prepare_features(data_encoded):
    """
    Prépare les features
    """
    print("Préparation des features...")
    
    # Variable cible
    if "new_deaths" not in data_encoded.columns:
        raise ValueError("ERREUR - Colonne 'new_deaths' introuvable.")
    
    y = data_encoded["new_deaths"]
    print(f"   Variable cible: new_deaths (min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f})")
    
    # Conversion de la date
    if "date" in data_encoded.columns:
        print("   Conversion de la date...")
        data_encoded["date"] = pd.to_datetime(data_encoded["date"])
        data_encoded["date"] = data_encoded["date"].astype('int64') // 10**9  # Timestamp Unix
    else:
        raise ValueError("ERREUR - Colonne 'date' introuvable.")
    
    # FEATURES FIXES - UNIQUEMENT tes 5 features
    base_features = ["date", "new_cases", "people_vaccinated", "new_tests", "daily_occupancy_hosp"]
    country_features = [col for col in data_encoded.columns if col.startswith("country1_")]
    
    print(f"   Features de base: {base_features}")
    print(f"   Features pays: {len(country_features)} pays")
    
    all_features = base_features + country_features
    
    # Vérifier que toutes les features existent
    missing_features = [f for f in all_features if f not in data_encoded.columns]
    if missing_features:
        print(f"   ERREUR - Features manquantes: {missing_features}")
        raise ValueError(f"Features manquantes: {missing_features}")
    
    X = data_encoded[all_features]
    
    # Vérifier qu'il n'y a pas de NaN dans les features finales
    if X.isnull().any().any():
        print("   ATTENTION - Des NaN détectés dans les features, nettoyage...")
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"   Colonnes avec NaN: {nan_cols}")
        
        # Supprimer les lignes avec NaN
        mask = X.isnull().any(axis=1)
        X = X[~mask]
        y = y[~mask]
        
        print(f"   Après nettoyage: {X.shape[0]} échantillons")
    
    # Sauvegarder pour référence
    X.to_csv("data/raw/features_matrix.csv", index=False)
    
    print(f"Matrice de features finale: {X.shape[0]} lignes × {X.shape[1]} colonnes")
    print(f"Features utilisées: {base_features}")
    
    return X, y, all_features

def train_baseline_model(X_train, X_test, y_train, y_test):
    """
    Entraîne le modèle de base (régression linéaire)
    """
    print("Entraînement du modèle baseline...")
    
    # Standardisation pour la régression linéaire
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    y_pred = linear_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Baseline - R²: {r2:.4f}, MAE: {mae:.4f}")
    
    return {
        "model": linear_model,
        "scaler": scaler,
        "r2": r2,
        "mae": mae,
        "predictions": y_pred
    }

def train_polynomial_all_countries(X_train, X_test, y_train, y_test):
    """
    Entraîne le modèle polynomial en gardant tous les pays
    """
    print("Entraînement du modèle polynomial...")
    print(f"   Features d'entrée: {X_train.shape[1]} colonnes")
    
    # Pipeline avec régularisation forte (beaucoup de features)
    polynomial_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('ridge', Ridge())
    ])
    
    # Grille de paramètres prudente
    param_grid = {
        'poly__degree': [1, 2],  # Évite explosion avec 44 pays
        'poly__interaction_only': [False, True],  # Teste avec/sans interactions
        'ridge__alpha': [1.0, 10.0, 100.0, 1000.0]  # Régularisation forte
    }
    
    print("Recherche d'hyperparamètres (GridSearchCV)...")
    grid_search = GridSearchCV(
        polynomial_pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Évaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    cv_score = grid_search.best_score_
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"CV Score: {cv_score:.4f}")
    print(f"Test R²: {r2:.4f}, MAE: {mae:.4f}")
    
    # Diagnostic overfitting
    overfitting_gap = r2 - cv_score
    if overfitting_gap > 0.05:
        print(f"ATTENTION - écart CV/Test = {overfitting_gap:.4f}")
    else:
        print(f"Écart CV/Test acceptable: {overfitting_gap:.4f}")
    
    return {
        "grid_search": grid_search,
        "model": best_model,
        "r2": r2,
        "mae": mae,
        "cv_score": cv_score,
        "best_params": grid_search.best_params_,
        "predictions": y_pred
    }

def save_model_and_metadata(poly_results, features, baseline_results, countries_list):
    """
    Sauvegarde le modèle et ses métadonnées
    """
    print("Sauvegarde du modèle...")
    
    os.makedirs("models", exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = "models/covid_polynomial_model.pkl"
    joblib.dump(poly_results["model"], model_path)
    
    # Features de base (tes 5 features)
    base_features = ["date", "new_cases", "people_vaccinated", "new_tests", "daily_occupancy_hosp"]
    
    # Métadonnées optimisées
    metadata = {
        "model_info": {
            "name": "COVID-19 Deaths Prediction Model - Simplified",
            "type": "polynomial_regression_with_ridge",
            "version": "1.0",
            "training_date": datetime.now().isoformat(),
            "features_count": len(features),
            "base_features_count": len(base_features),
            "countries_count": len(countries_list)
        },
        "features": {
            "all_features": features,
            "base_features": base_features,
            "country_features": countries_list
        },
        "hyperparameters": poly_results["best_params"],
        "performance": {
            "cross_validation_r2": poly_results["cv_score"],
            "test_r2": poly_results["r2"],
            "test_mae": poly_results["mae"],
            "baseline_r2": baseline_results["r2"],
            "baseline_mae": baseline_results["mae"],
            "improvement_r2_percent": ((poly_results["r2"] - baseline_results["r2"]) / baseline_results["r2"] * 100),
            "improvement_mae_percent": ((baseline_results["mae"] - poly_results["mae"]) / baseline_results["mae"] * 100)
        },
        "data_info": {
            "target_variable": "new_deaths",
            "date_encoding": "unix_timestamp",
            "country_encoding": "one_hot",
            "scaling": "StandardScaler",
            "features_used": base_features
        },
        "countries_supported": countries_list
    }
    
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Modèle sauvegardé: {model_path}")
    print(f"Métadonnées: {metadata_path}")
    print(f"Features utilisées: {base_features}")
    
    return model_path, metadata_path

def main():
    """
    Pipeline principal d'entraînement
    """
    print("DÉMARRAGE - ENTRAÎNEMENT MODÈLE COVID-19")
    print("=" * 70)
    
    try:
        # 1. Chargement et fusion des données
        df_merge = load_and_merge_data()
        
        # 2. Préparation et encodage
        data_encoded = prepare_data(df_merge)
        X, y, features = prepare_features(data_encoded)
        
        # 3. Extraction de la liste des pays
        countries_list = [col.replace("country1_", "") for col in features if col.startswith("country1_")]
        print(f"Pays supportés par le modèle: {len(countries_list)} pays")
        
        # 4. Division train/test
        print("\nDivision des données...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=None
        )
        print(f"Train: {X_train.shape[0]} échantillons")
        print(f"Test: {X_test.shape[0]} échantillons")
        
        # 5. Modèle baseline
        baseline_results = train_baseline_model(X_train, X_test, y_train, y_test)
        
        # 6. Modèle polynomial optimisé
        poly_results = train_polynomial_all_countries(X_train, X_test, y_train, y_test)
        
        # 7. Comparaison finale
        print("\n" + "=" * 70)
        print("RÉSULTATS FINAUX")
        print(f"Baseline Linéaire  - R²: {baseline_results['r2']:.4f}, MAE: {baseline_results['mae']:.4f}")
        print(f"Modèle Polynomial  - R²: {poly_results['r2']:.4f}, MAE: {poly_results['mae']:.4f}")
        
        improvement_r2 = ((poly_results['r2'] - baseline_results['r2']) / baseline_results['r2']) * 100
        improvement_mae = ((baseline_results['mae'] - poly_results['mae']) / baseline_results['mae']) * 100
        
        print(f"\nAMÉLIORATION:")
        print(f"   R² amélioré de {improvement_r2:.2f}%")
        print(f"   MAE amélioré de {improvement_mae:.2f}%")
        
        # 8. Sauvegarde
        model_path, metadata_path = save_model_and_metadata(
            poly_results, features, baseline_results, countries_list
        )
        
        print("\nSUCCÈS!")
        print(f"Modèle prêt: {model_path}")
        print(f"Informations: {metadata_path}")
        print(f"Pays supportés: {len(countries_list)}")
        
        return model_path, metadata_path, countries_list
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

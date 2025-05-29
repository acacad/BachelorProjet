from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib
import warnings
from db_utils import get_active_engine

warnings.filterwarnings('ignore')

matplotlib.use("Agg")  # Pour Flask / backend sans GUI

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcul du MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calcul du SMAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def analyze_dataset_for_xgboost(df, target_col='Sales'):
    """
    Analyse les caractéristiques du dataset pour XGBoost

    Returns:
        dict: Caractéristiques du dataset
    """
    characteristics = {}

    # Taille du dataset
    characteristics['n_samples'] = len(df)
    characteristics['n_features'] = len([col for col in df.columns if col not in ['date', target_col]])

    # Ratio samples/features
    characteristics['sample_feature_ratio'] = characteristics['n_samples'] / max(1, characteristics['n_features'])

    # Analyse de la cible
    if target_col in df.columns:
        target_values = df[target_col].values
        characteristics['target_mean'] = np.mean(target_values)
        characteristics['target_std'] = np.std(target_values)
        characteristics['target_skewness'] = pd.Series(target_values).skew()

        # Détection de la non-linéarité
        # Calcul simple : variance des différences successives
        if len(target_values) > 1:
            diff_var = np.var(np.diff(target_values))
            characteristics['volatility'] = diff_var / characteristics['target_std'] if characteristics['target_std'] > 0 else 0
        else:
            characteristics['volatility'] = 0

    # Présence de valeurs catégorielles (après get_dummies)
    characteristics['has_categorical'] = any(col.count('_') > 0 for col in df.columns)

    return characteristics

def get_xgboost_configuration(characteristics):
    """
    Détermine la configuration XGBoost optimale selon les caractéristiques

    4 configurations principales :
    1. LIGHT : Petits datasets simples (<200 samples)
    2. BALANCED : Datasets moyens (200-1000 samples)
    3. COMPLEX : Grands datasets complexes (1000-5000 samples)
    4. HEAVY : Très grands datasets (>5000 samples)
    """

    n_samples = characteristics['n_samples']
    n_features = characteristics['n_features']
    ratio = characteristics['sample_feature_ratio']
    volatility = characteristics.get('volatility', 0.5)

    # Déterminer la configuration
    if n_samples < 200:
        config_name = 'LIGHT'
    elif n_samples < 1000:
        config_name = 'BALANCED'
    elif n_samples < 5000:
        config_name = 'COMPLEX'
    else:
        config_name = 'HEAVY'

    # Ajustements basés sur d'autres facteurs
    if ratio < 10:  # Très peu d'échantillons par feature
        config_name = 'LIGHT'
    elif volatility > 2.0:  # Données très volatiles
        if config_name == 'LIGHT':
            config_name = 'BALANCED'

    # Configurations prédéfinies
    configs = {
        'LIGHT': {
            'n_estimators': 50,        # Peu d'arbres pour éviter l'overfitting
            'max_depth': 3,            # Arbres peu profonds
            'learning_rate': 0.1,      # Taux d'apprentissage plus élevé
            'subsample': 0.8,          # Moins de subsampling
            'colsample_bytree': 0.8,   # Utilise plus de features
            'min_child_weight': 5,     # Plus restrictif
            'reg_alpha': 0.1,          # Régularisation L1 légère
            'reg_lambda': 1.0,         # Régularisation L2 modérée
            'description': 'Configuration légère pour petits datasets (<200 samples)'
        },
        'BALANCED': {
            'n_estimators': 100,       # Nombre modéré d'arbres
            'max_depth': 5,            # Profondeur moyenne
            'learning_rate': 0.05,     # Taux d'apprentissage modéré
            'subsample': 0.7,          # Subsampling équilibré
            'colsample_bytree': 0.7,   # 70% des features par arbre
            'min_child_weight': 3,     # Équilibré
            'reg_alpha': 0.05,         # Régularisation L1 légère
            'reg_lambda': 1.0,         # Régularisation L2 standard
            'description': 'Configuration équilibrée pour datasets moyens (200-1000 samples)'
        },
        'COMPLEX': {
            'n_estimators': 200,       # Plus d'arbres
            'max_depth': 7,            # Arbres plus profonds
            'learning_rate': 0.03,     # Taux plus faible pour stabilité
            'subsample': 0.6,          # Plus de randomisation
            'colsample_bytree': 0.6,   # 60% des features
            'min_child_weight': 1,     # Moins restrictif
            'reg_alpha': 0.01,         # Peu de L1
            'reg_lambda': 0.5,         # L2 modérée
            'description': 'Configuration complexe pour grands datasets (1000-5000 samples)'
        },
        'HEAVY': {
            'n_estimators': 300,       # Beaucoup d'arbres
            'max_depth': 10,           # Arbres profonds
            'learning_rate': 0.01,     # Très faible pour éviter overfitting
            'subsample': 0.5,          # Fort subsampling
            'colsample_bytree': 0.5,   # 50% des features seulement
            'min_child_weight': 0.5,   # Très flexible
            'reg_alpha': 0.001,        # Très peu de L1
            'reg_lambda': 0.1,         # L2 légère
            'description': 'Configuration lourde pour très grands datasets (>5000 samples)'
        }
    }

    # Ajustements spécifiques selon les caractéristiques
    config = configs[config_name].copy()

    # Si beaucoup de features, ajuster colsample_bytree
    if n_features > 20:
        config['colsample_bytree'] = max(0.3, config['colsample_bytree'] - 0.1)

    # Si données très volatiles, plus de régularisation
    if volatility > 1.5:
        config['reg_lambda'] *= 1.5
        config['max_depth'] = max(3, config['max_depth'] - 1)

    # Si très peu d'échantillons par feature
    if ratio < 20:
        config['min_child_weight'] = min(10, config['min_child_weight'] * 2)
        config['subsample'] = min(0.9, config['subsample'] + 0.1)

    config['config_name'] = config_name

    return config

def print_configuration_details(characteristics, config):
    """Affiche les détails de la configuration choisie"""
    print("\n" + "="*70)
    print("ANALYSE ADAPTATIVE XGBOOST")
    print("="*70)

    print("\n📊 Caractéristiques du dataset:")
    print(f"  - Échantillons : {characteristics['n_samples']}")
    print(f"  - Features : {characteristics['n_features']}")
    print(f"  - Ratio samples/features : {characteristics['sample_feature_ratio']:.1f}")
    print(f"  - Volatilité : {characteristics.get('volatility', 0):.3f}")
    print(f"  - Asymétrie (skewness) : {characteristics.get('target_skewness', 0):.3f}")

    print(f"\n🎯 Configuration sélectionnée : {config['config_name']}")
    print(f"   {config['description']}")

    print("\n🔧 Paramètres XGBoost:")
    params_to_show = ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                      'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda']
    for param in params_to_show:
        if param in config:
            print(f"  - {param}: {config[param]}")

    print("\n💡 Justification des choix:")
    if config['config_name'] == 'LIGHT':
        print("  • Peu d'arbres et faible profondeur pour éviter l'overfitting")
        print("  • Learning rate élevé car peu d'itérations")
        print("  • Forte régularisation pour généraliser")
    elif config['config_name'] == 'BALANCED':
        print("  • Paramètres équilibrés pour un bon compromis biais/variance")
        print("  • Subsampling modéré pour la robustesse")
    elif config['config_name'] == 'COMPLEX':
        print("  • Plus d'arbres pour capturer les patterns complexes")
        print("  • Learning rate faible pour convergence stable")
        print("  • Moins de régularisation car plus de données")
    else:  # HEAVY
        print("  • Nombreux arbres peu profonds pour éviter l'overfitting")
        print("  • Fort subsampling pour la diversité")
        print("  • Learning rate très faible pour stabilité")

    print("="*70 + "\n")

def run_xgb_forecast(use_adaptive=True, custom_params=None, perform_grid_search=False):
    """
    Exécute la prévision XGBoost avec paramètres adaptatifs

    Args:
        use_adaptive: Si True, utilise les paramètres adaptatifs
        custom_params: Dict de paramètres personnalisés (override)
        perform_grid_search: Si True, effectue une recherche de grille pour optimiser
    """

    try:
        # Connexion à la base de données
        engine = get_active_engine() if 'get_active_engine' in globals() else \
            create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Encodage des colonnes catégorielles automatiquement
        df = pd.get_dummies(df)

        # Création de features temporelles
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month

        # Création de la variable de décalage
        df['Sales_lag1'] = df['Sales'].shift(1)
        df['Sales_lag7'] = df['Sales'].shift(7)  # Lag hebdomadaire
        df['Sales_rolling_mean_7'] = df['Sales'].rolling(window=7, min_periods=1).mean()
        df['Sales_rolling_std_7'] = df['Sales'].rolling(window=7, min_periods=1).std()

        df = df.dropna().reset_index(drop=True)

        # Analyse du dataset et configuration adaptative
        if use_adaptive:
            characteristics = analyze_dataset_for_xgboost(df)
            config = get_xgboost_configuration(characteristics)
            print_configuration_details(characteristics, config)

            # Extraire les paramètres pour XGBoost
            xgb_params = {k: v for k, v in config.items()
                          if k not in ['description', 'config_name']}

            # Override avec custom_params si fourni
            if custom_params:
                xgb_params.update(custom_params)
                print(f"\n⚠️  Paramètres personnalisés appliqués: {custom_params}")
        else:
            # Paramètres par défaut
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'reg_alpha': 0.05,
                'reg_lambda': 1.0
            }
            if custom_params:
                xgb_params.update(custom_params)

        # Préparation des features
        feature_cols = [col for col in df.columns if col not in ['date', 'Sales']]
        X = df[feature_cols]
        y = df['Sales']

        # Split train/test
        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        test_dates = df['date'].iloc[split_index:]

        # Grid Search optionnel pour optimisation fine
        if perform_grid_search and len(X_train) > 100:
            print("\n🔍 Recherche des hyperparamètres optimaux...")

            # Grille de recherche réduite basée sur la config
            if config['config_name'] == 'LIGHT':
                param_grid = {
                    'max_depth': [2, 3, 4],
                    'n_estimators': [30, 50, 70],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            else:
                param_grid = {
                    'max_depth': [config['max_depth']-1, config['max_depth'], config['max_depth']+1],
                    'n_estimators': [int(config['n_estimators']*0.8), config['n_estimators'], int(config['n_estimators']*1.2)],
                    'learning_rate': [config['learning_rate']*0.5, config['learning_rate'], config['learning_rate']*2]
                }

            # Time Series Split pour validation croisée
            tscv = TimeSeriesSplit(n_splits=3)

            grid_search = GridSearchCV(
                XGBRegressor(random_state=42, **{k: v for k, v in xgb_params.items()
                                                 if k not in param_grid}),
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            xgb_params.update(best_params)
            print(f"✅ Meilleurs paramètres trouvés: {best_params}")

        # Méthode 1 : Configuration dans le constructeur
        xgb_params_with_eval = xgb_params.copy()
        xgb_params_with_eval['eval_metric'] = ['rmse', 'mae']
        # Création et entraînement du modèle
        model = XGBRegressor(random_state=42, **xgb_params)

        # Entraînement
        try:
            # Essayer avec early stopping
            eval_set = [(X_test, y_test)]  # Utiliser seulement le test set
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=True  # Mettre True pour voir la progression
            )
            print("✅ Modèle entraîné avec early stopping")

        except Exception as e:
            print(f"⚠️ Early stopping non supporté: {e}")
            # Fallback : entraînement simple
            model.fit(X_train, y_train)
            print("✅ Modèle entraîné (sans early stopping)")



        # Prédictions
        y_pred = model.predict(X_test)

        # Prévision future (30 jours)
        print("\n📈 Génération des prévisions futures...")
        last_date = df['date'].max()
        future_30 = pd.DataFrame({'date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)})

        # Créer les features temporelles pour le futur
        future_30['day_of_week'] = future_30['date'].dt.dayofweek
        future_30['day_of_month'] = future_30['date'].dt.day
        future_30['month'] = future_30['date'].dt.month

        # Pour les autres features, utiliser des stratégies intelligentes
        for col in feature_cols:
            if col in ['day_of_week', 'day_of_month', 'month']:
                continue  # Déjà créées
            elif 'lag' in col or 'rolling' in col:
                # Pour les lags et rolling, utiliser les dernières valeurs connues
                if 'lag1' in col:
                    future_30[col] = df[col].iloc[-30:].values if len(df) >= 30 else df[col].mean()
                elif 'lag7' in col:
                    future_30[col] = df[col].iloc[-30:].values if len(df) >= 30 else df[col].mean()
                else:
                    future_30[col] = df[col].tail(30).mean()
            else:
                # Pour les autres features, utiliser la moyenne des 30 derniers jours
                future_30[col] = df[col].tail(30).mean()

        y_future_pred = model.predict(future_30[feature_cols])
        forecast_sum = np.sum(y_future_pred)
        future_dates = future_30['date']

        # Métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🎯 Top 10 features les plus importantes:")
        print(feature_importance.head(10).to_string(index=False))

        # Graphiques
        import os
        if not os.path.exists('static'):
            os.makedirs('static')

        # 1. Graphique principal
        plt.figure(figsize=(14, 6))

        # Historique récent
        historical_days = 60
        hist_mask = df.index >= len(df) - historical_days - len(y_test)
        hist_dates = df.loc[hist_mask, 'date']
        hist_sales = df.loc[hist_mask, 'Sales']
        plt.plot(hist_dates, hist_sales, label="Historique", color="gray", alpha=0.5)

        # Prédictions sur le test set
        plt.plot(test_dates, y_test.values, label="Ventes réelles", color="blue", linewidth=2)
        plt.plot(test_dates, y_pred, label="Prédiction XGBoost", color="red", linestyle="--", linewidth=2)

        # Prévisions futures
        plt.plot(future_dates, y_future_pred, label="Prévision 30 jours", color="green", linestyle="--", linewidth=2)

        # Ligne de séparation
        plt.axvline(x=last_date, color='black', linestyle=':', alpha=0.5, label="Aujourd'hui")

        plt.title(f"Prévisions XGBoost - Configuration {config['config_name'] if use_adaptive else 'Manuelle'}")
        plt.xlabel("Date")
        plt.ylabel("Ventes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/pred_xgboost.png")
        plt.close()

        # 2. Graphique des feature importances
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Features Importantes')
        plt.tight_layout()
        plt.savefig("static/xgb_feature_importance.png")
        plt.close()

        print(f"\n✅ Modèle entraîné avec succès!")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²: {r2:.3f}")

        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 2),
            "mape": round(mape, 2),
            "smape": round(smape, 2),
            "future_sum": round(forecast_sum, 2),
            "graph_path": "pred_xgboost.png",
            "feature_importance_path": "xgb_feature_importance.png",
            "config_used": config['config_name'] if use_adaptive else 'Manual',
            "params_used": xgb_params
        }

    except Exception as e:
        print(f"❌ Erreur dans run_xgb_forecast: {str(e)}")
        import traceback
        traceback.print_exc()

        # Retourner des valeurs par défaut en cas d'erreur
        return {
            "rmse": 0,
            "mae": 0,
            "r2": 0,
            "mape": 0,
            "smape": 0,
            "future_sum": 0,
            "graph_path": "pred_xgboost.png",
            "error": str(e)
        }

# Pour test local
if __name__ == "__main__":
    print("Test 1: Mode adaptatif")
    results = run_xgb_forecast(use_adaptive=True)
    print("\n=== Résultats XGBoost Adaptatif ===")
    for k, v in results.items():
        if k not in ['params_used', 'graph_path', 'feature_importance_path']:
            print(f"{k}: {v}")

    print("\n\nTest 2: Mode adaptatif avec grid search")
    results2 = run_xgb_forecast(use_adaptive=True, perform_grid_search=True)
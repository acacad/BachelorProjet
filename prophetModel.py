from sqlalchemy import create_engine
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")  # Pas d'affichage direct, utile pour script/server

def smape(y_true, y_pred):
    """Calcul du Symmetric Mean Absolute Percentage Error"""
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_prophet_forecast():
    """
    Exécute la prévision Prophet sur les données de ventes
    """
    try:
        # Connexion à la base de données
        engine = create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        # Préparation des données pour Prophet
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['Sales']

        # Identification des colonnes numériques pour les régresseurs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exog_cols = [col for col in numeric_cols if col not in ['Sales', 'y'] and col not in ['ds']]

        # Split train/test
        train_df = df.iloc[:-30]
        test_df = df.iloc[-30:]

        # Création et entraînement du modèle Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05  # Régularisation pour éviter l'overfitting
        )

        # Ajout des régresseurs externes
        for col in exog_cols:
            model.add_regressor(col)

        # Entraînement du modèle
        model.fit(train_df[['ds', 'y'] + exog_cols])

        # Prédiction sur le test set
        future_test = test_df[['ds'] + exog_cols].copy()
        forecast_test = model.predict(future_test)

        # Préparation des données pour les métriques
        df_compare = test_df[['ds', 'Sales']].copy()
        df_compare = df_compare.merge(forecast_test[['ds', 'yhat']], on='ds')

        y_true = df_compare['Sales'].values
        y_pred = df_compare['yhat'].values

        # Calcul des métriques sur le test set (30 derniers jours)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        smape_value = smape(y_true, y_pred)

        # Métriques sur les 200 dernières lignes pour validation
        df_200 = df.tail(200)
        future_200 = df_200[['ds'] + exog_cols].copy()
        forecast_200 = model.predict(future_200)
        df_200_compare = df_200[['ds', 'Sales']].copy().merge(forecast_200[['ds', 'yhat']], on='ds')

        y_true_200 = df_200_compare['Sales'].values
        y_pred_200 = df_200_compare['yhat'].values

        rmse_200 = np.sqrt(mean_squared_error(y_true_200, y_pred_200))
        mae_200 = mean_absolute_error(y_true_200, y_pred_200)
        r2_200 = r2_score(y_true_200, y_pred_200)
        mape_200 = mean_absolute_percentage_error(y_true_200, y_pred_200) * 100
        smape_200 = smape(y_true_200, y_pred_200)

        # Prévision future (30 jours)
        last_date = df['ds'].max()
        future_30 = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)})

        # CORRECTION : Utiliser les moyennes des features au lieu de 0
        for col in exog_cols:
            # Option 1 : Moyennes des 30 derniers jours
            future_30[col] = df[col].tail(30).mean()

            # Option 2 : Extrapolation linéaire (décommentez si préféré)
            # future_30[col] = extrapolate_feature(df, col, 30)

        # Prévision sur les 30 jours futurs
        forecast_30 = model.predict(future_30)
        forecast_sum = forecast_30['yhat'].sum()

        # CORRECTION : Créer le dossier static s'il n'existe pas
        if not os.path.exists('static'):
            os.makedirs('static')

        # Graphique des prévisions
        plt.figure(figsize=(14, 6))

        # Historique récent pour contexte
        historical_days = 60
        hist_df = df.tail(historical_days + 30)  # +30 pour inclure le test set
        plt.plot(hist_df['ds'], hist_df['Sales'],
                 label="Ventes historiques", color='gray', alpha=0.5, linewidth=1)

        # Test set (30 derniers jours)
        plt.plot(df_compare['ds'], y_true,
                 label="Ventes réelles (test)", color='blue', linewidth=2)
        plt.plot(df_compare['ds'], y_pred,
                 label="Prédictions Prophet", color='orange', linestyle='--', linewidth=2)

        # Prévisions futures
        plt.plot(forecast_30['ds'], forecast_30['yhat'],
                 label="Prévision 30 jours", color='green', linestyle='--', linewidth=2)

        # Intervalles de confiance pour les prévisions futures
        if 'yhat_lower' in forecast_30.columns and 'yhat_upper' in forecast_30.columns:
            plt.fill_between(forecast_30['ds'],
                             forecast_30['yhat_lower'],
                             forecast_30['yhat_upper'],
                             alpha=0.2, color='green', label='Intervalle de confiance')

        # Ligne verticale pour séparer historique/futur
        plt.axvline(x=last_date, color='red', linestyle=':', alpha=0.5, label='Aujourd\'hui')

        plt.title("Prévisions de ventes - Prophet", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Ventes", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # CORRECTION : Chemin absolu pour le fichier
        graph_path = os.path.join('static', 'pred_prophet.png')
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Graphique sauvegardé dans : {graph_path}")

        # Retour des résultats avec le bon chemin
        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 2),
            "mape": round(mape, 2),
            "smape": round(smape_value, 2),
            "rmse_200": round(rmse_200, 2),
            "mae_200": round(mae_200, 2),
            "r2_200": round(r2_200, 2),
            "mape_200": round(mape_200, 2),
            "smape_200": round(smape_200, 2),
            "future_sum": round(forecast_sum, 2),
            "graph_path": "pred_prophet.png"  # CORRECTION : Chemin relatif pour Flask
        }

    except Exception as e:
        print(f"Erreur dans run_prophet_forecast: {str(e)}")
        import traceback
        traceback.print_exc()

        # Créer un graphique d'erreur
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Erreur: {str(e)}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=12, color='red')
        plt.title("Erreur lors de la génération du graphique")

        if not os.path.exists('static'):
            os.makedirs('static')

        error_path = os.path.join('static', 'pred_prophet.png')
        plt.savefig(error_path)
        plt.close()

        # Retourner des valeurs par défaut avec l'erreur
        return {
            "rmse": 0,
            "mae": 0,
            "r2": 0,
            "mape": 0,
            "smape": 0,
            "rmse_200": 0,
            "mae_200": 0,
            "r2_200": 0,
            "mape_200": 0,
            "smape_200": 0,
            "future_sum": 0,
            "graph_path": "pred_prophet.png",
            "error": str(e)
        }

def extrapolate_feature(df, feature_col, days_ahead):
    """
    Extrapole une feature basée sur la tendance récente

    Args:
        df: DataFrame contenant les données
        feature_col: Nom de la colonne à extrapoler
        days_ahead: Nombre de jours à extrapoler

    Returns:
        Valeur extrapolée pour la feature
    """
    from sklearn.linear_model import LinearRegression

    # Prendre les 30 derniers jours
    recent_data = df.tail(30)
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data[feature_col].values

    # Entraîner un modèle de régression simple
    model = LinearRegression()
    model.fit(X, y)

    # Extrapoler
    future_X = np.array([[len(recent_data) + days_ahead / 2]])  # Milieu de la période
    predicted_value = model.predict(future_X)[0]

    # S'assurer que la valeur est positive
    return max(0, predicted_value)

# Pour test local
if __name__ == "__main__":
    results = run_prophet_forecast()
    print("\n=== Résultats Prophet ===")
    for metric, value in results.items():
        if metric != "graph_path" and metric != "error":
            print(f"{metric}: {value}")
    if "error" in results:
        print(f"\nErreur: {results['error']}")
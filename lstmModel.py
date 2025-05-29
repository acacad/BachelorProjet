import os

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib

from app import get_active_engine

matplotlib.use("Agg")  # Pour Flask ou scripts headless

def smape(y_true, y_pred):
    """Calcul du Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def analyze_dataset_characteristics(df, target_col='Sales'):
    """
    Analyse les caractéristiques du dataset pour ajuster les paramètres

    Retourne un dictionnaire avec les caractéristiques clés
    """
    characteristics = {}

    # Taille du dataset
    characteristics['n_samples'] = len(df)
    characteristics['n_features'] = len([col for col in df.columns if col not in ['date', target_col]])

    # Analyse de la variance et de la complexité
    if target_col in df.columns:
        target_values = df[target_col].values
        characteristics['target_mean'] = np.mean(target_values)
        characteristics['target_std'] = np.std(target_values)
        characteristics['cv'] = characteristics['target_std'] / characteristics['target_mean']  # Coefficient de variation

        # Analyse de la tendance (complexité temporelle)
        from scipy import stats
        x = np.arange(len(target_values))
        slope, _, r_value, _, _ = stats.linregress(x, target_values)
        characteristics['trend_strength'] = abs(r_value)  # 0-1, proche de 1 = forte tendance

        # Analyse de la saisonnalité (simple)
        if len(target_values) > 14:
            # Autocorrélation à 7 jours (hebdomadaire)
            autocorr_7 = pd.Series(target_values).autocorr(lag=7)
            characteristics['weekly_seasonality'] = abs(autocorr_7) if not np.isnan(autocorr_7) else 0
        else:
            characteristics['weekly_seasonality'] = 0

    return characteristics

def get_adaptive_parameters(characteristics):
    """
    Détermine les paramètres optimaux basés sur les caractéristiques du dataset

    Basé sur les recommandations de:
    - "Deep Learning for Time Series Forecasting" (Brownlee, 2018)
    - "On the difficulty of training RNNs" (Pascanu et al., 2013)
    - Expériences empiriques sur différentes tailles de datasets
    """
    params = {}

    n_samples = characteristics['n_samples']
    n_features = characteristics['n_features']
    cv = characteristics.get('cv', 0.5)
    trend_strength = characteristics.get('trend_strength', 0.5)
    weekly_seasonality = characteristics.get('weekly_seasonality', 0)

    # 1. TIME_STEPS - Règles adaptatives
    if n_samples < 100:
        params['time_steps'] = min(7, n_samples // 10)
    elif n_samples < 500:
        if weekly_seasonality > 0.3:
            params['time_steps'] = 14  # 2 semaines pour capturer la saisonnalité
        else:
            params['time_steps'] = 10
    elif n_samples < 1000:
        params['time_steps'] = 21  # 3 semaines
    else:
        params['time_steps'] = min(30, n_samples // 50)

    # 2. LSTM_UNITS - Basé sur la complexité
    # Règle : units = α * sqrt(n_features * time_steps) + β * complexity
    complexity_factor = (cv + trend_strength) / 2
    base_units = int(np.sqrt(n_features * params['time_steps']) * 4)

    if complexity_factor < 0.3:  # Données simples
        params['lstm_units'] = max(8, min(base_units, 24))
    elif complexity_factor < 0.6:  # Complexité moyenne
        params['lstm_units'] = max(16, min(base_units * 1.5, 48))
    else:  # Données complexes
        params['lstm_units'] = max(24, min(base_units * 2, 64))

    # 3. BATCH_SIZE - Basé sur la taille du dataset
    train_size = int(n_samples * 0.8) - params['time_steps']

    if train_size < 100:
        params['batch_size'] = 8
    elif train_size < 500:
        params['batch_size'] = 16
    elif train_size < 1000:
        params['batch_size'] = 32
    else:
        params['batch_size'] = min(64, train_size // 20)

    # 4. EPOCHS - Avec early stopping, on peut se permettre plus
    if n_samples < 200:
        params['epochs'] = 100
    elif n_samples < 1000:
        params['epochs'] = 150
    else:
        params['epochs'] = 200

    # 5. DROPOUT - Basé sur le risque d'overfitting
    overfitting_risk = min(1.0, n_features / (n_samples / 100))

    if overfitting_risk < 0.2:
        params['dropout'] = 0.1
    elif overfitting_risk < 0.5:
        params['dropout'] = 0.2
    else:
        params['dropout'] = min(0.3, overfitting_risk * 0.4)

    # 6. LEARNING_RATE - Adaptatif
    if n_samples < 500:
        params['learning_rate'] = 0.001
    else:
        params['learning_rate'] = 0.0005

    # 7. ARCHITECTURE - Simple ou complexe
    if n_samples < 300 or n_features < 3:
        params['architecture'] = 'simple'  # Une seule couche LSTM
    else:
        params['architecture'] = 'stacked'  # Deux couches LSTM

    # 8. ACTIVATION
    params['activation'] = 'tanh'  # Standard pour LSTM

    # 9. EARLY STOPPING PATIENCE
    params['patience'] = max(10, min(20, params['epochs'] // 10))

    return params

def print_adaptive_parameters(characteristics, params):
    """Affiche les paramètres choisis et les raisons"""
    print("\n" + "="*60)
    print("ANALYSE ADAPTATIVE DU DATASET")
    print("="*60)

    print("\n📊 Caractéristiques du dataset:")
    print(f"  - Échantillons : {characteristics['n_samples']}")
    print(f"  - Features : {characteristics['n_features']}")
    print(f"  - Coefficient de variation : {characteristics.get('cv', 0):.3f}")
    print(f"  - Force de tendance : {characteristics.get('trend_strength', 0):.3f}")
    print(f"  - Saisonnalité hebdomadaire : {characteristics.get('weekly_seasonality', 0):.3f}")

    print("\n🔧 Paramètres adaptatifs sélectionnés:")
    print(f"  - time_steps : {params['time_steps']} jours")
    print(f"  - lstm_units : {params['lstm_units']} neurones")
    print(f"  - batch_size : {params['batch_size']}")
    print(f"  - epochs : {params['epochs']} (avec early stopping)")
    print(f"  - dropout : {params['dropout']}")
    print(f"  - learning_rate : {params['learning_rate']}")
    print(f"  - architecture : {params['architecture']}")
    print(f"  - patience : {params['patience']}")
    print("="*60 + "\n")

def create_adaptive_model(time_steps, n_features, params):
    """
    Crée un modèle LSTM avec une architecture adaptative
    """
    model = Sequential()

    if params['architecture'] == 'simple':
        # Architecture simple pour petits datasets
        model.add(Input(shape=(time_steps, n_features)))
        model.add(LSTM(params['lstm_units'],
                       activation=params['activation'],
                       dropout=params['dropout'],
                       recurrent_dropout=params['dropout']/2))
        model.add(Dense(1))

    else:  # 'stacked'
        # Architecture plus complexe pour grands datasets
        model.add(Input(shape=(time_steps, n_features)))

        # Première couche LSTM
        model.add(LSTM(params['lstm_units'],
                       activation=params['activation'],
                       return_sequences=True,
                       dropout=params['dropout'],
                       recurrent_dropout=params['dropout']/2))
        model.add(BatchNormalization())

        # Deuxième couche LSTM (moitié moins d'unités)
        model.add(LSTM(params['lstm_units']//2,
                       activation=params['activation'],
                       dropout=params['dropout'],
                       recurrent_dropout=params['dropout']/2))
        model.add(BatchNormalization())

        # Couche dense intermédiaire
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(params['dropout']))

        # Sortie
        model.add(Dense(1))

    # Compilation avec learning rate adaptatif
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_lstm_model(time_steps=None, lstm_units=None, epochs=None,
                     batch_size=None, activation=None, use_adaptive=True):
    """
    Entraîne un modèle LSTM avec paramètres adaptatifs ou manuels

    Si use_adaptive=True, les paramètres sont automatiquement déterminés
    """
    # Connexion à la base de données
    engine = get_active_engine() if 'get_active_engine' in globals() else \
        create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
    df = pd.read_sql("SELECT * FROM sales_data", engine)

    # Préparation des données
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    target_col = 'Sales'
    feature_cols = [col for col in df.columns if col not in ['date', 'Sales']]

    # Analyse du dataset et paramètres adaptatifs
    if use_adaptive:
        characteristics = analyze_dataset_characteristics(df, target_col)
        params = get_adaptive_parameters(characteristics)
        print_adaptive_parameters(characteristics, params)

        # Utiliser les paramètres adaptatifs sauf si spécifiés manuellement
        time_steps = time_steps or params['time_steps']
        lstm_units = lstm_units or params['lstm_units']
        epochs = epochs or params['epochs']
        batch_size = batch_size or params['batch_size']
        activation = activation or params['activation']
    else:
        # Valeurs par défaut si non spécifiées
        time_steps = time_steps or 21
        lstm_units = lstm_units or 24
        epochs = epochs or 75
        batch_size = batch_size or 32
        activation = activation or 'tanh'
        params = {
            'dropout': 0.2,
            'learning_rate': 0.001,
            'architecture': 'simple',
            'patience': 15
        }

    # Préparation des données
    X_raw = df[feature_cols].values
    y_raw = df[[target_col]].values

    # Fonction pour créer les séquences temporelles
    def create_sequences(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:i + time_steps])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    # Création des séquences
    X_seq, y_seq = create_sequences(X_raw, y_raw, time_steps)

    # Split train/test
    split_index = int(len(X_seq) * 0.8)
    X_train_raw, X_test_raw = X_seq[:split_index], X_seq[split_index:]
    y_train_raw, y_test_raw = y_seq[:split_index], y_seq[split_index:]

    # Normalisation
    scaler_X = MinMaxScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    scaler_y = MinMaxScaler().fit(y_train_raw)

    X_train = scaler_X.transform(X_train_raw.reshape(-1, X_train_raw.shape[2])).reshape(X_train_raw.shape)
    X_test = scaler_X.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)
    y_train = scaler_y.transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    # Création du modèle adaptatif
    model = create_adaptive_model(time_steps, X_train.shape[2], params)

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=params['patience']//2,
        min_lr=0.00001,
        verbose=1
    )

    # Entraînement avec suivi détaillé
    print("\n🚀 Début de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Nombre d'époques réellement utilisées
    actual_epochs = len(history.history['loss'])
    print(f"\n✅ Entraînement terminé après {actual_epochs} époques (early stopping)")

    # Évaluation
    y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
    y_test_actual = scaler_y.inverse_transform(y_test)
    test_dates = df['date'].iloc[-len(y_test_actual):]

    # Métriques
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred) * 100
    smape_val = smape(y_test_actual, y_pred)

    # Prévision future
    future_preds = []
    recent_feature_means = df[feature_cols].tail(30).mean().values
    last_seq = X_raw[-time_steps:]
    current_seq = scaler_X.transform(last_seq).reshape(1, time_steps, -1)

    for day in range(30):
        pred_scaled = model.predict(current_seq, verbose=0)[0, 0]
        future_preds.append(pred_scaled)

        new_features = scaler_X.transform([recent_feature_means])[0]
        new_input = current_seq[0, 1:, :].tolist()
        new_input.append(new_features.tolist())
        current_seq = np.array(new_input).reshape(1, time_steps, -1)

    future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    forecast_sum = np.sum(future_preds)
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=30)

    # Graphiques
    if not os.path.exists('static'):
        os.makedirs('static')

    # 1. Graphique des prédictions
    plt.figure(figsize=(12, 6))
    historical_days = 60
    hist_dates = df['date'].tail(historical_days)
    hist_sales = df['Sales'].tail(historical_days)
    plt.plot(hist_dates, hist_sales, label='Historique récent', color='gray', alpha=0.5)
    plt.plot(test_dates, y_test_actual, label='Ventes réelles (test)', color='blue', linewidth=2)
    plt.plot(test_dates, y_pred, label='Prédictions LSTM', linestyle='--', color='green', linewidth=2)
    plt.plot(future_dates, future_preds, label='Prévision 30 jours', linestyle='--', color='orange', linewidth=2)
    plt.axvline(x=df['date'].max(), color='red', linestyle=':', alpha=0.5, label='Aujourd\'hui')
    plt.legend()
    plt.title("Prédictions LSTM Adaptatives")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/pred_lstm.png")
    plt.close()

    # 2. Graphique de l'historique d'entraînement
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Évolution du MAE')
    plt.xlabel('Époque')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("static/training_history.png")
    plt.close()

    # Retour des résultats avec infos sur les paramètres utilisés
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 2),
        "mape": round(mape, 2),
        "smape": round(smape_val, 2),
        "future_sum": round(forecast_sum, 2),
        "graph_path": "pred_lstm.png",
        "rmse_epoch_path": "training_history.png",
        "params_used": {
            "time_steps": time_steps,
            "lstm_units": lstm_units,
            "epochs": actual_epochs,
            "batch_size": batch_size,
            "architecture": params['architecture'],
            "dropout": params['dropout']
        }
    }

# Pour compatibilité avec l'ancien code
run_lstm_forecast = train_lstm_model

if __name__ == "__main__":
    # Test avec paramètres adaptatifs
    print("Test avec paramètres adaptatifs:")
    result = train_lstm_model(use_adaptive=True)
    print("\n=== Résultats ===")
    for k, v in result.items():
        if k not in ['graph_path', 'rmse_epoch_path']:
            print(f"{k}: {v}")
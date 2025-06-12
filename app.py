from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import mysql.connector
from matplotlib import pyplot as plt
from mysql.connector import Error
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import prophetModel
import os
import XGBModel
import lstmModel
import numpy as np
from flask import send_file  # Pour le téléchargement des fichiers
from werkzeug.utils import secure_filename
import os
import shutil
import seaborn as sns
from db_utils import get_active_engine
import LinearReg


app = Flask(__name__)
app.secret_key = "a_secret_key_for_sessions"  # À sécuriser en prod

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="ecommerce"
    )

# Page d'accueil
@app.route("/")
def index():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Niche")
        niches = cursor.fetchall()

        # Par défaut, on peut afficher la première niche si elle existe
        selected_niche = niches[0] if niches else None
        if selected_niche:
            niche_id = selected_niche.get("id")
            if not niche_id:
                influenceurs = []
            else:
                cursor.execute("SELECT * FROM Influenceur WHERE niche_id = %s", (niche_id,))
                influenceurs = cursor.fetchall()
        else:
            influenceurs = []

        return render_template("index.html",
                               niches=niches,
                               influenceurs=influenceurs,
                               selected_niche=selected_niche,
                               logged_in=session.get("logged_in", False))
    except Error as e:
        return f"Erreur : {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Connexion admin
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "jesuistonadmin":
            session["logged_in"] = True
            return redirect(url_for("index"))
        return "Identifiants incorrects", 401
    return render_template("login.html")

# Déconnexion
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("index"))

# Voir les niches
@app.route("/niches_page")
def niches_page():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Niche")
        niches = cursor.fetchall()
        return render_template("niches.html", niches=niches)
    except Error as e:
        return f"Erreur : {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Ajouter une niche (admin uniquement)
@app.route("/add_niche_page", methods=["GET", "POST"])
def add_niche_page():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    if request.method == "POST":
        nom = request.form.get("nom")
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Niche (nom) VALUES (%s)", (nom,))
            conn.commit()
            return redirect(url_for("niches_page"))
        except Error as e:
            return f"Erreur : {e}"
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    return render_template("add_niche.html")

@app.route("/prophet_forecast")
def prophet_forecast():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        results = prophetModel.run_prophet_forecast()

        # Vérifier si une erreur s'est produite
        if "error" in results:
            print(f"Erreur Prophet: {results['error']}")
            return f"Erreur lors de la prévision Prophet: {results['error']}", 500

        return render_template("prophet_result.html",
                               image_path=results["graph_path"],
                               rmse=results["rmse"],
                               mae=results["mae"],
                               r2=results["r2"],
                               mape=results["mape"],
                               smape=results["smape"],
                               rmse_200=results["rmse_200"],
                               mae_200=results["mae_200"],
                               r2_200=results["r2_200"],
                               mape_200=results["mape_200"],
                               smape_200=results["smape_200"],
                               future_sum=results["future_sum"])
    except Exception as e:
        print(f"Erreur dans prophet_forecast: {str(e)}")
        return f"Erreur lors de l'exécution de Prophet: {str(e)}", 500


@app.route("/lstm_forecast")
def lstm_forecast():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        # CORRECTION : Utiliser le bon nom de fonction
        results = lstmModel.train_lstm_model()  # Au lieu de run_lstm_forecast()

        # Vérifier si les clés attendues sont présentes
        required_keys = ["rmse", "mae", "r2", "mape", "smape", "future_sum", "graph_path", "rmse_epoch_path"]
        missing_keys = [key for key in required_keys if key not in results]

        if missing_keys:
            print(f"Clés manquantes dans les résultats LSTM: {missing_keys}")
            return f"Erreur: résultats LSTM incomplets. Clés manquantes: {missing_keys}", 500

        return render_template("lstm_result.html",
                               image_path=results["graph_path"],
                               rmse_epoch_path=results["rmse_epoch_path"],  # Utiliser la bonne clé
                               rmse=results["rmse"],
                               mae=results["mae"],
                               mape=results["mape"],
                               smape=results["smape"],
                               r2=results["r2"],
                               future_sum=results["future_sum"])
    except Exception as e:
        print(f"Erreur dans lstm_forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Erreur lors de l'exécution du modèle LSTM: {str(e)}", 500

@app.route("/variables")
def view_variables():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        variables = [col for col in df.columns if col not in ["date", "id"]]

        # Statistiques globales existantes
        global_stats = df[variables].agg(['mean', 'std', 'min', 'max', 'median']).transpose().reset_index()
        global_stats.columns = ['Variable', 'Moyenne', 'Écart-type', 'Min', 'Max', 'Médiane']

        # Statistiques sur les 50 derniers jours
        last_50 = df.tail(50)
        stats_50 = last_50[variables].agg(['mean', 'std', 'min', 'max', 'median']).transpose().reset_index()
        stats_50.columns = ['Variable', 'Moyenne_50j', 'Écart-type_50j', 'Min_50j', 'Max_50j', 'Médiane_50j']

        # NOUVELLE ANALYSE 1: Statistiques avancées avec percentiles et coefficient de variation
        advanced_stats = []
        for var in variables:
            var_stats = {
                'Variable': var,
                'Q1': df[var].quantile(0.25),
                'Q3': df[var].quantile(0.75),
                'IQR': df[var].quantile(0.75) - df[var].quantile(0.25),
                'CV': (df[var].std() / df[var].mean() * 100) if df[var].mean() != 0 else 0,  # Coefficient de variation en %
                'Skewness': df[var].skew(),
                'Kurtosis': df[var].kurtosis(),
                'Zeros_count': (df[var] == 0).sum(),
                'Zeros_pct': (df[var] == 0).sum() / len(df) * 100
            }
            advanced_stats.append(var_stats)

        # NOUVELLE ANALYSE 2: Analyse de tendance (croissance/décroissance)
        trend_analysis = []
        for var in variables:
            # Calculer la tendance via régression linéaire simple
            from scipy import stats
            x = np.arange(len(df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, df[var])

            # Comparer première et dernière période
            first_quarter = df[var].iloc[:len(df)//4].mean()
            last_quarter = df[var].iloc[-len(df)//4:].mean()
            growth_rate = ((last_quarter - first_quarter) / first_quarter * 100) if first_quarter != 0 else 0

            trend_analysis.append({
                'Variable': var,
                'Slope': slope,
                'R_squared': r_value**2,
                'P_value': p_value,
                'Growth_rate': growth_rate,
                'Trend': 'Croissant' if slope > 0 else 'Décroissant',
                'Significant': 'Oui' if p_value < 0.05 else 'Non'
            })

        # NOUVELLE ANALYSE 3: Corrélations avec Sales uniquement
        sales_correlations = []
        if 'Sales' in variables:
            for var in variables:
                if var != 'Sales':
                    corr = df['Sales'].corr(df[var])
                    # Corrélation avec décalage temporel
                    lag1_corr = df['Sales'].corr(df[var].shift(1))
                    lag7_corr = df['Sales'].corr(df[var].shift(7))

                    sales_correlations.append({
                        'Variable': var,
                        'Correlation': corr,
                        'Lag1_Correlation': lag1_corr,
                        'Lag7_Correlation': lag7_corr,
                        'Best_Lag': 0 if abs(corr) >= max(abs(lag1_corr), abs(lag7_corr)) else (1 if abs(lag1_corr) >= abs(lag7_corr) else 7)
                    })

        # NOUVELLE ANALYSE 4: Analyse de saisonnalité
        seasonality_analysis = []
        if len(df) >= 30:  # Besoin d'au moins 30 jours pour analyser
            for var in variables:
                # Autocorrélation manuelle
                def manual_autocorr(series, lag):
                    """Calcul manuel de l'autocorrélation"""
                    if len(series) <= lag:
                        return np.nan
                    c0 = np.var(series)
                    if c0 == 0:
                        return np.nan
                    c_lag = np.cov(series[:-lag], series[lag:])[0, 1]
                    return c_lag / c0

                series = df[var].values
                acf_7 = manual_autocorr(series, 7) if len(df) > 7 else np.nan
                acf_30 = manual_autocorr(series, 30) if len(df) > 30 else np.nan

                # Variance par jour de la semaine
                df['day_of_week'] = df['date'].dt.dayofweek
                weekly_var = df.groupby('day_of_week')[var].mean().std()

                seasonality_analysis.append({
                    'Variable': var,
                    'ACF_7days': acf_7,
                    'ACF_30days': acf_30,
                    'Weekly_variance': weekly_var,
                    'Has_weekly_pattern': 'Oui' if abs(acf_7) > 0.3 else 'Non'
                })

        # Créer le dossier static s'il n'existe pas
        if not os.path.exists('static'):
            os.makedirs('static')

        # Graphiques existants (1-4)...
        # Graphe 1 : uniquement les ventes
        plt.figure(figsize=(14, 5))
        plt.plot(df['date'], df['Sales'], label="Sales", color="blue", linewidth=2)
        plt.title("Évolution des ventes")
        plt.xlabel("Date")
        plt.ylabel("Ventes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/sales_only.png")
        plt.close()

        # Graphe 2 : ventes + variables
        plt.figure(figsize=(14, 6))
        for var in variables:
            if var != 'Sales':
                plt.plot(df['date'], df[var], label=var)
        plt.plot(df['date'], df['Sales'], label="Sales", linewidth=2, color="black")
        plt.title("Ventes et variables exogènes")
        plt.xlabel("Date")
        plt.ylabel("Valeurs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/sales_with_vars.png")
        plt.close()

        # Graphe 3 : Évolution logarithmique des ventes
        df['log_sales'] = np.log1p(df['Sales'])
        plt.figure(figsize=(14, 5))
        plt.plot(df['date'], df['log_sales'], color='purple', linewidth=2)
        plt.title("Évolution logarithmique des ventes (log(Sales + 1))")
        plt.xlabel("Date")
        plt.ylabel("Logarithme des ventes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/sales_log.png")
        plt.close()

        # Graphe 4 : matrice de corrélation
        import seaborn as sns
        correlation = df[variables].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.savefig("static/correlation_matrix.png")
        plt.close()

        # NOUVEAU GRAPHE 5: Box plots pour visualiser les distributions
        n_vars = len(variables)
        if n_vars <= 10:  # Si peu de variables, un seul graphique
            plt.figure(figsize=(14, 8))
            # Normaliser les données pour la comparaison
            df_normalized = df[variables].copy()
            for var in variables:
                if df_normalized[var].std() > 0:
                    df_normalized[var] = (df_normalized[var] - df_normalized[var].mean()) / df_normalized[var].std()

            df_normalized.boxplot(rot=45)
            plt.title("Distribution des variables (normalisées)")
            plt.ylabel("Valeur normalisée (z-score)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("static/variables_boxplot.png")
            plt.close()
        else:
            # Si beaucoup de variables, créer un graphique simplifié
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Box plot non généré : {n_vars} variables (max 10)",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig("static/variables_boxplot.png")
            plt.close()

        # NOUVEAU GRAPHE 6: Évolution des moyennes mobiles
        plt.figure(figsize=(14, 6))
        if 'Sales' in df.columns:
            df['MA7'] = df['Sales'].rolling(window=7, min_periods=1).mean()
            df['MA30'] = df['Sales'].rolling(window=30, min_periods=1).mean()

            plt.plot(df['date'], df['Sales'], label="Ventes", alpha=0.5, color='gray')
            plt.plot(df['date'], df['MA7'], label="Moyenne mobile 7j", color='blue', linewidth=2)
            plt.plot(df['date'], df['MA30'], label="Moyenne mobile 30j", color='red', linewidth=2)
            plt.title("Ventes avec moyennes mobiles")
            plt.xlabel("Date")
            plt.ylabel("Ventes")
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/moving_averages.png")
        plt.close()

        # NOUVEAU GRAPHE 7: Décomposition temporelle des ventes (version simplifiée sans statsmodels)
        if len(df) >= 60 and 'Sales' in df.columns:  # Besoin de suffisamment de données
            try:
                # Version manuelle de la décomposition sans statsmodels
                fig, axes = plt.subplots(4, 1, figsize=(14, 10))

                # 1. Données originales
                axes[0].plot(df['date'], df['Sales'], color='blue', linewidth=1)
                axes[0].set_title('Ventes originales')
                axes[0].grid(True, alpha=0.3)

                # 2. Tendance (moyenne mobile sur 30 jours)
                window = 30 if len(df) >= 90 else max(7, len(df) // 3)
                df['trend'] = df['Sales'].rolling(window=window, center=True, min_periods=1).mean()
                axes[1].plot(df['date'], df['trend'], color='red', linewidth=2)
                axes[1].set_title(f'Tendance (moyenne mobile {window}j)')
                axes[1].grid(True, alpha=0.3)

                # 3. Calcul de la saisonnalité
                # Retirer la tendance
                df['detrended'] = df['Sales'] - df['trend']

                # Calculer la saisonnalité hebdomadaire
                df['day_of_week'] = df['date'].dt.dayofweek
                seasonal_pattern = df.groupby('day_of_week')['detrended'].mean()
                df['seasonal'] = df['day_of_week'].map(seasonal_pattern)

                # Normaliser la saisonnalité
                df['seasonal'] = df['seasonal'] - df['seasonal'].mean()

                axes[2].plot(df['date'], df['seasonal'], color='green', linewidth=1)
                axes[2].set_title('Saisonnalité hebdomadaire')
                axes[2].grid(True, alpha=0.3)

                # 4. Résidus
                df['residual'] = df['Sales'] - df['trend'] - df['seasonal']
                axes[3].plot(df['date'], df['residual'], color='gray', linewidth=1, alpha=0.7)
                axes[3].set_title('Résidus')
                axes[3].grid(True, alpha=0.3)

                # Ajuster les labels
                for ax in axes:
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Valeur')

                plt.tight_layout()
                plt.savefig("static/seasonal_decomposition.png")
                plt.close()

                # Nettoyer les colonnes temporaires
                df.drop(['trend', 'detrended', 'seasonal', 'residual'], axis=1, inplace=True, errors='ignore')

                decomposition_available = True
            except Exception as e:
                # Si la décomposition échoue, créer un placeholder
                print(f"Erreur lors de la décomposition: {e}")
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Décomposition non disponible\n(erreur lors du calcul)",
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.savefig("static/seasonal_decomposition.png")
                plt.close()
                decomposition_available = False
        else:
            # Pas assez de données
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Décomposition non disponible\n(minimum 60 observations, actuellement {len(df)})",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig("static/seasonal_decomposition.png")
            plt.close()
            decomposition_available = False

        # NOUVEAU GRAPHE 8: Scatter plots des top corrélations avec Sales
        if sales_correlations and len(sales_correlations) > 0:
            # Trier par corrélation absolue
            top_corr = sorted(sales_correlations, key=lambda x: abs(x['Correlation']), reverse=True)[:4]

            if len(top_corr) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()

                for i, corr_info in enumerate(top_corr):
                    if i < 4:
                        var = corr_info['Variable']
                        ax = axes[i]
                        ax.scatter(df[var], df['Sales'], alpha=0.5)
                        ax.set_xlabel(var)
                        ax.set_ylabel('Sales')
                        ax.set_title(f"Corrélation: {corr_info['Correlation']:.3f}")

                        # Ajouter ligne de tendance
                        z = np.polyfit(df[var], df['Sales'], 1)
                        p = np.poly1d(z)
                        ax.plot(df[var].sort_values(), p(df[var].sort_values()), "r--", alpha=0.8)
                        ax.grid(True, alpha=0.3)

                # Masquer les axes non utilisés
                for i in range(len(top_corr), 4):
                    axes[i].set_visible(False)

                plt.tight_layout()
                plt.savefig("static/correlation_scatter.png")
                plt.close()
            else:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Pas de corrélations à afficher",
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.savefig("static/correlation_scatter.png")
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Pas de corrélations disponibles",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig("static/correlation_scatter.png")
            plt.close()

        # Préparer les données pour le template
        editable_variables = [var for var in variables if var != 'Sales']

        return render_template("variables.html",
                               variables=variables,
                               global_stats=global_stats.to_dict(orient='records'),
                               stats_50=stats_50.to_dict(orient='records'),
                               advanced_stats=advanced_stats,
                               trend_analysis=trend_analysis,
                               sales_correlations=sales_correlations,
                               seasonality_analysis=seasonality_analysis,
                               sales_only_path="sales_only.png",
                               sales_with_vars_path="sales_with_vars.png",
                               sales_log_path="sales_log.png",
                               correlation_path="correlation_matrix.png",
                               boxplot_path="variables_boxplot.png",
                               moving_avg_path="moving_averages.png",
                               decomposition_path="seasonal_decomposition.png",
                               scatter_path="correlation_scatter.png",
                               editable_variables=editable_variables,
                               dataset_info={
                                   'n_observations': len(df),
                                   'n_variables': len(variables),
                                   'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} à {df['date'].max().strftime('%Y-%m-%d')}",
                                   'n_days': (df['date'].max() - df['date'].min()).days + 1
                               })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erreur : {e}", 500


@app.route("/add_variable", methods=["GET", "POST"])
def add_variable():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    # Récupérer la liste des variables existantes pour permettre leur modification
    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Variables existantes (hors colonnes fixes)
        fixed_cols = ["date", "Sales", "id"]
        existing_variables = [col for col in df.columns if col not in fixed_cols]

    except Exception as e:
        existing_variables = []
        flash(f"Erreur lors du chargement des variables: {str(e)}", "error")
        # Créer un DataFrame vide pour éviter les erreurs
        df = pd.DataFrame()

    if request.method == "POST":
        action = request.form.get("action", "add")

        if action == "add":
            # Code existant pour ajouter une variable
            name = request.form.get("name")
            start_date_str = request.form.get("start_date")
            end_date_str = request.form.get("end_date")
            repeat = request.form.get("repeat")
            effect_type = request.form.get("effect_type")

            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

                if name not in df.columns:
                    df[name] = 0.0  # Valeur continue

                def generate_effect_values(dates, effect_type):
                    n = len(dates)
                    if n == 0:
                        return []
                    if effect_type == "constant":
                        return [1.0] * n
                    elif effect_type == "linear_down":
                        return np.linspace(1.0, 0.0, n).tolist()
                    elif effect_type == "linear_up":
                        return np.linspace(0.0, 1.0, n).tolist()
                    elif effect_type == "middle_peak":
                        peak = n // 2
                        return [i / peak if i <= peak else (n - i - 1) / (n - peak - 1) for i in range(n)]
                    else:
                        return [1.0] * n

                def get_repeated_periods(start, end, repeat_type):
                    periods = []
                    current = start
                    while current <= max(df['date']):
                        next_end = current + (end - start)
                        periods.append((current, next_end))
                        if repeat_type == "weekly":
                            current += timedelta(weeks=1)
                        elif repeat_type == "monthly":
                            month = current.month + 1
                            year = current.year + (month - 1) // 12
                            month = (month - 1) % 12 + 1
                            current = current.replace(year=year, month=month)
                        else:
                            break
                    return periods

                periods = [(start_date, end_date)]
                if repeat in ("weekly", "monthly"):
                    periods = get_repeated_periods(start_date, end_date, repeat)

                for start, end in periods:
                    mask = (df['date'] >= start) & (df['date'] <= end)
                    affected_dates = df.loc[mask, 'date'].sort_values()
                    values = generate_effect_values(affected_dates, effect_type)
                    df.loc[mask, name] = values

                df.to_sql("sales_data", engine, if_exists="replace", index=False)
                flash(f"Variable '{name}' ajoutée avec succès!", "success")
                return redirect(url_for("add_variable"))

            except Exception as e:
                flash(f"Erreur lors de l'ajout: {str(e)}", "error")

        elif action == "modify":
            # Nouvelle fonctionnalité : modifier une variable existante
            variable_name = request.form.get("variable_to_modify")
            modify_action = request.form.get("modify_action")

            try:
                if modify_action == "set_period":
                    # Modifier une période spécifique
                    start_date_str = request.form.get("modify_start_date")
                    end_date_str = request.form.get("modify_end_date")
                    effect_type = request.form.get("modify_effect_type")
                    value = float(request.form.get("modify_value", 1.0))

                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    affected_dates = df.loc[mask, 'date'].sort_values()

                    if effect_type == "constant":
                        df.loc[mask, variable_name] = value
                    elif effect_type == "linear_up":
                        n = len(affected_dates)
                        values = np.linspace(0, value, n)
                        df.loc[mask, variable_name] = values
                    elif effect_type == "linear_down":
                        n = len(affected_dates)
                        values = np.linspace(value, 0, n)
                        df.loc[mask, variable_name] = values
                    elif effect_type == "multiply":
                        df.loc[mask, variable_name] *= value
                    elif effect_type == "add":
                        df.loc[mask, variable_name] += value

                elif modify_action == "set_all":
                    # Définir toutes les valeurs de la variable
                    value = float(request.form.get("modify_value", 0))
                    df[variable_name] = value

                elif modify_action == "reset":
                    # Remettre toutes les valeurs à zéro
                    df[variable_name] = 0

                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                flash(f"Variable '{variable_name}' modifiée avec succès!", "success")
                return redirect(url_for("add_variable"))

            except Exception as e:
                flash(f"Erreur lors de la modification: {str(e)}", "error")

    # Données à passer au template
    template_data = {
        'existing_variables': existing_variables,
        'min_date': df['date'].min().strftime('%Y-%m-%d') if not df.empty and len(df) > 0 else '',
        'max_date': df['date'].max().strftime('%Y-%m-%d') if not df.empty and len(df) > 0 else ''
    }

    return render_template("add_variable.html", **template_data)

@app.route("/delete_variable", methods=["GET", "POST"])
def delete_variable():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        # Variables existantes (hors colonnes fixes)
        fixed_cols = ["date", "Sales"]
        variables = [col for col in df.columns if col not in fixed_cols]

        if request.method == "POST":
            variable_to_delete = request.form.get("variable")
            if variable_to_delete in df.columns:
                df.drop(columns=[variable_to_delete], inplace=True)
                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                return redirect(url_for("view_variables"))

        return render_template("delete_variable.html", variables=variables)

    except Exception as e:
        return f"Erreur : {e}", 500


@app.route("/xgb_forecast")
def xgb_forecast():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:

        results = XGBModel.run_xgb_forecast()

        # Vérifier si une erreur s'est produite
        if "error" in results:
            print(f"Erreur XGBoost: {results['error']}")
            return f"Erreur lors de la prévision XGBoost: {results['error']}", 500

        return render_template("xgb_result.html",
                               image_path=results["graph_path"],
                               rmse=results["rmse"],
                               mae=results["mae"],
                               r2=results["r2"],
                               mape=results["mape"],
                               smape=results["smape"],
                               future_sum=results["future_sum"])
    except Exception as e:
        print(f"Erreur dans xgb_forecast: {str(e)}")
        return f"Erreur lors de l'exécution de XGBoost: {str(e)}", 500


# Route de debug pour vérifier les fichiers static
@app.route("/debug_static")
def debug_static():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    import os
    static_files = []
    if os.path.exists('static'):
        static_files = os.listdir('static')

    return f"<h1>Fichiers dans le dossier static:</h1><ul>{''.join([f'<li>{f}</li>' for f in static_files])}</ul>"



# Configuration pour l'upload de fichiers
UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'csv'}

# Ajouter cette configuration après app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataset(df):
    """
    Valide que le dataset contient les colonnes nécessaires
    et est dans le bon format
    """
    errors = []

    # Ne plus exiger la colonne date
    # if 'date' not in df.columns:
    #     errors.append("Le dataset doit contenir une colonne 'date'")

    if 'Sales' not in df.columns and 'sales' not in df.columns:
        errors.append("Le dataset doit contenir une colonne 'Sales' ou 'sales'")

    # Vérifier le format de date SI elle existe
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            errors.append("La colonne 'date' doit être dans un format de date valide")

    # Vérifier qu'il y a au moins quelques lignes
    if len(df) < 10:
        errors.append("Le dataset doit contenir au moins 10 lignes")

    return errors

@app.route("/datasets")
def list_datasets():
    """Liste tous les datasets disponibles"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    # Créer le dossier s'il n'existe pas
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Sauvegarder le dataset original au premier accès
    backup_original_dataset()

    # Lister tous les fichiers CSV (sauf la sauvegarde)
    datasets = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.csv') and not filename.startswith('_'):  # Ignorer les fichiers de sauvegarde
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                df = pd.read_csv(filepath, nrows=5)
                full_df = pd.read_csv(filepath)
                info = {
                    'name': filename,
                    'rows': len(full_df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'size': os.path.getsize(filepath) / 1024,
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M')
                }
                datasets.append(info)
            except Exception as e:
                print(f"Erreur lors de la lecture de {filename}: {e}")

    # Obtenir le dataset actif
    active_dataset = session.get('active_dataset', 'database')

    # Informations sur le dataset original
    try:
        engine = create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        original_df = pd.read_sql("SELECT * FROM sales_data LIMIT 5", engine)
        original_info = {
            'exists': True,
            'columns': len(original_df.columns),
            'column_names': list(original_df.columns)
        }
        # Compter les lignes séparément pour éviter de charger tout le dataset
        count_result = pd.read_sql("SELECT COUNT(*) as count FROM sales_data", engine)
        original_info['rows'] = count_result['count'].iloc[0]
    except:
        original_info = {'exists': False}

    return render_template("datasets.html",
                           datasets=datasets,
                           active_dataset=active_dataset,
                           original_info=original_info)

# Ajouter une fonction pour exporter le dataset actuel
@app.route("/export_current_dataset")
def export_current_dataset():
    """Exporte le dataset actuellement actif"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        # Nom du fichier basé sur le dataset actif
        active = session.get('active_dataset', 'database')
        if active == 'database':
            filename = f"dataset_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filename = f"dataset_{active.replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Créer un fichier temporaire
        temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{filename}')
        df.to_csv(temp_path, index=False)

        return send_file(temp_path,
                         as_attachment=True,
                         download_name=filename,
                         mimetype='text/csv')
    except Exception as e:
        flash(f"Erreur lors de l'export: {str(e)}", "error")
        return redirect(url_for("list_datasets"))

@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    """Upload un nouveau dataset CSV"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        # Vérifier si un fichier a été uploadé
        if 'file' not in request.files:
            return render_template("upload_dataset.html",
                                   error="Aucun fichier sélectionné")

        file = request.files['file']
        dataset_name = request.form.get('dataset_name', '').strip()

        if file.filename == '':
            return render_template("upload_dataset.html",
                                   error="Aucun fichier sélectionné")

        if file and allowed_file(file.filename):
            # Sécuriser le nom du fichier
            if dataset_name:
                filename = secure_filename(dataset_name + '.csv')
            else:
                filename = secure_filename(file.filename)

            # Créer le dossier s'il n'existe pas
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            # Chemin temporaire pour validation
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_' + filename)
            file.save(temp_path)

            try:
                # Lire et nettoyer le dataset
                df = pd.read_csv(temp_path)

                # Normaliser les noms de colonnes
                df.columns = df.columns.str.strip()
                if 'sales' in df.columns:
                    df.rename(columns={'sales': 'Sales'}, inplace=True)

                # NOUVELLE FONCTIONNALITÉ : Créer une colonne date si elle n'existe pas
                if 'date' not in df.columns:
                    print("⚠️ Pas de colonne 'date' détectée, création automatique...")
                    # Créer des dates en partant d'aujourd'hui et en remontant
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=len(df)-1)
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    df['date'] = date_range
                    flash(f"Colonne 'date' créée automatiquement (du {start_date} au {end_date})", "info")

                # Convertir la colonne date au bon format
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        # Reformater en string pour la sauvegarde CSV
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    except:
                        pass

                # Remplacer toutes les valeurs NaN par 0
                nan_count_before = df.isna().sum().sum()
                df = df.fillna(0)
                nan_count_after = df.isna().sum().sum()
                nan_replaced = nan_count_before - nan_count_after

                # Valider le dataset
                errors = validate_dataset(df)

                if errors:
                    os.remove(temp_path)
                    return render_template("upload_dataset.html",
                                           error="Erreurs de validation: " + ", ".join(errors))

                # Sauvegarder le dataset validé et nettoyé
                final_path = os.path.join(UPLOAD_FOLDER, filename)

                # Si le fichier existe déjà, demander confirmation
                if os.path.exists(final_path):
                    os.remove(temp_path)
                    return render_template("upload_dataset.html",
                                           error=f"Un dataset nommé '{filename}' existe déjà")

                # Sauvegarder le dataset nettoyé
                df.to_csv(final_path, index=False)
                os.remove(temp_path)

                # Message de succès avec info sur les modifications
                success_msg = f"Dataset '{filename}' importé avec succès!"
                if nan_replaced > 0:
                    success_msg += f" ({nan_replaced} valeurs manquantes remplacées par 0)"
                flash(success_msg, "success")

                # Optionnel : charger aussi dans la base de données
                if request.form.get('load_to_db'):
                    engine = get_active_engine()
                    df.to_sql('sales_data', engine, if_exists='replace', index=False)
                    session['active_dataset'] = filename
                    flash("Dataset chargé et activé dans la base de données", "success")

                return redirect(url_for("list_datasets"))

            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return render_template("upload_dataset.html",
                                       error=f"Erreur lors du traitement du fichier: {str(e)}")
        else:
            return render_template("upload_dataset.html",
                                   error="Type de fichier non autorisé. Utilisez un fichier CSV.")

    return render_template("upload_dataset.html")


# Fonction pour nettoyer les fichiers temporaires
def cleanup_temp_files():
    """Nettoie les fichiers temporaires du dossier datasets"""
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.startswith('temp_'):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, filename))
                except:
                    pass

@app.route("/activate_dataset/<dataset_name>")
def activate_dataset(dataset_name):
    """Active un dataset pour l'utiliser dans les analyses"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()

        if dataset_name == 'database':
            # Restaurer le dataset original depuis la sauvegarde
            backup_path = os.path.join(UPLOAD_FOLDER, '_original_database_backup.csv')

            if os.path.exists(backup_path):
                # Charger la sauvegarde
                df = pd.read_csv(backup_path)
                # S'assurer qu'il n'y a pas de NaN même dans la sauvegarde
                df = df.fillna(0)
                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                session['active_dataset'] = 'database'
                flash("Dataset original restauré avec succès", "success")
            else:
                # Si pas de sauvegarde, on suppose que c'est déjà le dataset original
                session['active_dataset'] = 'database'
                flash("Dataset original déjà actif", "info")

            return redirect(url_for("list_datasets"))

        # Pour les autres datasets
        filepath = os.path.join(UPLOAD_FOLDER, dataset_name)
        if not os.path.exists(filepath):
            flash("Dataset non trouvé", "error")
            return redirect(url_for("list_datasets"))

        # Charger le nouveau dataset
        df = pd.read_csv(filepath)

        # S'assurer qu'il n'y a pas de NaN
        nan_count = df.isna().sum().sum()
        df = df.fillna(0)

        # Charger dans MySQL
        df.to_sql('sales_data', engine, if_exists='replace', index=False)

        # Marquer comme actif
        session['active_dataset'] = dataset_name

        msg = f"Dataset '{dataset_name}' activé avec succès"
        if nan_count > 0:
            msg += f" ({nan_count} valeurs manquantes remplacées par 0)"
        flash(msg, "success")

        return redirect(url_for("list_datasets"))

    except Exception as e:
        flash(f"Erreur lors de l'activation du dataset: {str(e)}", "error")
        return redirect(url_for("list_datasets"))

@app.route("/edit_variable/<variable_name>", methods=["GET", "POST"])
def edit_variable(variable_name):
    """Modifier les valeurs d'une variable existante"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if variable_name not in df.columns:
            flash(f"Variable '{variable_name}' non trouvée", "error")
            return redirect(url_for("view_variables"))

        if request.method == "POST":
            action = request.form.get("action")

            if action == "set_period":
                # Modifier les valeurs sur une période
                start_date_str = request.form.get("start_date")
                end_date_str = request.form.get("end_date")
                effect_type = request.form.get("effect_type")
                value = float(request.form.get("value", 1.0))

                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)

                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                affected_dates = df.loc[mask, 'date'].sort_values()

                if effect_type == "constant":
                    df.loc[mask, variable_name] = value
                elif effect_type == "linear_up":
                    n = len(affected_dates)
                    values = np.linspace(0, value, n)
                    df.loc[mask, variable_name] = values
                elif effect_type == "linear_down":
                    n = len(affected_dates)
                    values = np.linspace(value, 0, n)
                    df.loc[mask, variable_name] = values
                elif effect_type == "multiply":
                    df.loc[mask, variable_name] *= value
                elif effect_type == "add":
                    df.loc[mask, variable_name] += value

                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                flash(f"Variable '{variable_name}' modifiée avec succès", "success")

            elif action == "set_all":
                # Mettre toutes les valeurs à une constante
                value = float(request.form.get("value", 0))
                df[variable_name] = value
                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                flash(f"Toutes les valeurs de '{variable_name}' mises à {value}", "success")

            elif action == "reset":
                # Remettre à zéro
                df[variable_name] = 0
                df.to_sql('sales_data', engine, if_exists='replace', index=False)
                flash(f"Variable '{variable_name}' réinitialisée à 0", "success")

            return redirect(url_for("view_variables"))

        # Préparer les données pour l'affichage
        current_values = df[['date', variable_name]].to_dict('records')
        stats = {
            'mean': df[variable_name].mean(),
            'std': df[variable_name].std(),
            'min': df[variable_name].min(),
            'max': df[variable_name].max(),
            'count_non_zero': (df[variable_name] != 0).sum()
        }

        return render_template("edit_variable.html",
                               variable_name=variable_name,
                               current_values=current_values,
                               stats=stats,
                               min_date=df['date'].min().strftime('%Y-%m-%d'),
                               max_date=df['date'].max().strftime('%Y-%m-%d'))

    except Exception as e:
        flash(f"Erreur: {str(e)}", "error")
        return redirect(url_for("view_variables"))

# 4. NOUVELLE ROUTE : Visualiser la valeur d'une variable à une date donnée
@app.route("/view_variable_value", methods=["GET", "POST"])
def view_variable_value():
    """Voir la valeur d'une variable pour une date spécifique"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)
        df['date'] = pd.to_datetime(df['date'])

        variables = [col for col in df.columns if col not in ["date", "id"]]

        result = None
        if request.method == "POST":
            selected_date = request.form.get("selected_date")
            selected_var = request.form.get("selected_variable")

            if selected_date and selected_var:
                selected_date = pd.to_datetime(selected_date)

                # Trouver la valeur exacte ou la plus proche
                exact_match = df[df['date'] == selected_date]

                if not exact_match.empty:
                    value = exact_match[selected_var].iloc[0]
                    result = {
                        'date': selected_date.strftime('%Y-%m-%d'),
                        'variable': selected_var,
                        'value': value,
                        'exact': True
                    }
                else:
                    # Trouver la date la plus proche
                    df['date_diff'] = abs(df['date'] - selected_date)
                    closest_row = df.loc[df['date_diff'].idxmin()]
                    value = closest_row[selected_var]
                    result = {
                        'date': selected_date.strftime('%Y-%m-%d'),
                        'closest_date': closest_row['date'].strftime('%Y-%m-%d'),
                        'variable': selected_var,
                        'value': value,
                        'exact': False
                    }

                # Ajouter le contexte (valeurs autour)
                window_start = selected_date - timedelta(days=7)
                window_end = selected_date + timedelta(days=7)
                context_df = df[(df['date'] >= window_start) & (df['date'] <= window_end)]

                if not context_df.empty:
                    result['context'] = context_df[['date', selected_var]].to_dict('records')
                    for item in result['context']:
                        item['date'] = item['date'].strftime('%Y-%m-%d')

        return render_template("view_variable_value.html",
                               variables=variables,
                               result=result,
                               min_date=df['date'].min().strftime('%Y-%m-%d'),
                               max_date=df['date'].max().strftime('%Y-%m-%d'))

    except Exception as e:
        flash(f"Erreur: {str(e)}", "error")
        return redirect(url_for("view_variables"))

@app.route("/delete_dataset/<dataset_name>")
def delete_dataset(dataset_name):
    """Supprime un dataset"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    filepath = os.path.join(UPLOAD_FOLDER, dataset_name)

    try:
        if os.path.exists(filepath):
            os.remove(filepath)

            # Si c'était le dataset actif, revenir à la base de données
            if session.get('active_dataset') == dataset_name:
                session['active_dataset'] = 'database'

        return redirect(url_for("list_datasets"))

    except Exception as e:
        return f"Erreur lors de la suppression: {str(e)}", 500

@app.route("/download_dataset/<dataset_name>")
def download_dataset(dataset_name):
    """Télécharge un dataset"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if dataset_name == 'current':
        # Télécharger le dataset actuellement dans la base
        try:
            engine = get_active_engine()
            df = pd.read_sql("SELECT * FROM sales_data", engine)

            # Créer un fichier temporaire
            temp_path = os.path.join(UPLOAD_FOLDER, 'temp_download.csv')
            df.to_csv(temp_path, index=False)

            return send_file(temp_path,
                             as_attachment=True,
                             download_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                             mimetype='text/csv')
        except Exception as e:
            return f"Erreur: {str(e)}", 500
    else:
        filepath = os.path.join(UPLOAD_FOLDER, dataset_name)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return "Fichier non trouvé", 404

# Ajouter ce filtre pour l'utiliser dans les templates
@app.template_filter('format_size')
def format_size(size_kb):
    """Formate la taille du fichier de manière lisible"""
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    else:
        return f"{size_kb/1024:.1f} MB"

# fonction pour sauvegarder le dataset original
def backup_original_dataset():
    """Sauvegarde le dataset original de la base de données"""
    backup_path = os.path.join(UPLOAD_FOLDER, '_original_database_backup.csv')

    # Vérifier si la sauvegarde existe déjà
    if not os.path.exists(backup_path):
        try:
            engine = create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
            df = pd.read_sql("SELECT * FROM sales_data", engine)

            # Sauvegarder le dataset original
            df.to_csv(backup_path, index=False)
            print(f"Dataset original sauvegardé dans : {backup_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du dataset original : {e}")
# Modifier légèrement la fonction create_engine pour utiliser le bon dataset

'''
def get_active_engine():
    """Retourne l'engine pour le dataset actif"""
    active = session.get('active_dataset', 'database')

    if active == 'database':
        return create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
    else:
        # Le dataset a déjà été chargé dans la base lors de l'activation
        return create_engine("mysql+mysqlconnector://root:root@localhost/salesmarket")
        
        
'''

# Fonction utilitaire pour analyser les NaN dans un dataset
def analyze_nan_in_dataset(df):
    """Analyse les valeurs manquantes dans un DataFrame"""
    nan_summary = {}
    total_cells = df.shape[0] * df.shape[1]
    total_nan = df.isna().sum().sum()

    nan_summary['total_cells'] = total_cells
    nan_summary['total_nan'] = total_nan
    nan_summary['percentage'] = (total_nan / total_cells * 100) if total_cells > 0 else 0

    # Détail par colonne
    nan_by_column = {}
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_by_column[col] = {
                'count': nan_count,
                'percentage': (nan_count / len(df) * 100)
            }

    nan_summary['by_column'] = nan_by_column

    return nan_summary

# Optionnel : Route pour vérifier les NaN dans le dataset actuel
@app.route("/check_dataset_quality")
def check_dataset_quality():
    """Vérifie la qualité du dataset actuel"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        engine = get_active_engine()
        df = pd.read_sql("SELECT * FROM sales_data", engine)

        # Analyser les NaN
        nan_analysis = analyze_nan_in_dataset(df)

        # Autres statistiques de qualité
        quality_report = {
            'rows': len(df),
            'columns': len(df.columns),
            'nan_analysis': nan_analysis,
            'duplicates': df.duplicated().sum(),
            'column_types': df.dtypes.astype(str).to_dict()
        }

        return render_template("dataset_quality.html",
                               quality_report=quality_report,
                               active_dataset=session.get('active_dataset', 'database'))

    except Exception as e:
        flash(f"Erreur lors de l'analyse: {str(e)}", "error")
        return redirect(url_for("list_datasets"))



@app.route("/linear_regression_result")
def linear_regression_forecast():
    """Route pour la régression linéaire simple"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        # Exécuter la régression linéaire simple
        results = LinearReg.run_linear_regression()

        # Vérifier si une erreur s'est produite
        if "error" in results:
            print(f"Erreur Régression Linéaire: {results['error']}")
            return f"Erreur lors de la régression linéaire: {results['error']}", 500

        return render_template("linear_regression_result.html",
                               image_path=results["graph_path"],
                               rmse=results["rmse"],
                               mae=results["mae"],
                               r2=results["r2"],
                               mape=results["mape"],
                               smape=results["smape"],
                               # Nouvelles métriques sur 100 valeurs
                               rmse_100=results.get("rmse_100", "N/A"),
                               mae_100=results.get("mae_100", "N/A"),
                               r2_100=results.get("r2_100", "N/A"),
                               mape_100=results.get("mape_100", "N/A"),
                               smape_100=results.get("smape_100", "N/A"),
                               future_sum=results["future_sum"],
                               # Informations sur le modèle
                               coefficients=results["coefficients"],
                               intercept=results["intercept"],
                               model_name="Régression Linéaire Simple")

    except Exception as e:
        print(f"Erreur dans linear_regression_forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Erreur lors de l'exécution de la régression linéaire: {str(e)}", 500


if __name__ == "__main__":
    # Créer les dossiers nécessaires
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Nettoyer les fichiers temporaires
    cleanup_temp_files()

    # Sauvegarder le dataset original au démarrage
    with app.app_context():
        backup_original_dataset()

    app.run(debug=True)

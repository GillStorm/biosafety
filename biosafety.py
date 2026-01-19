import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
import pymannkendall as mk
import numpy as np

from phylogeny_weighting import add_phylogeny_weights, analyze_weights

# ---------------------------------------------------------
# BASIC PAGE CONFIG (no external animations -> more stable)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AMR Biosafety Dashboard",
    page_icon="🦠",
    layout="wide",
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🧬 AI-Driven AMR Biosafety & Novelty Detection System (Research Grade)")
st.write(
    "Upload AMR surveillance data to get: "
    "1) resistance trends, 2) ML predictions, 3) novelty / anomaly detection, "
    "and 4) biosafety risk scores."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📂 Data Source")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Upload CSV", "WHO GLASS AMR 2022 (Official)")
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
else:
    uploaded_file = None

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
show_plots = st.sidebar.checkbox("Show Visualizations", True, help="Trends & heatmaps")
show_risk = st.sidebar.checkbox("Show Multidimensional Risk Scores", True, help="Composite risk scoring")
show_novelty = st.sidebar.checkbox("Show Novelty Detection", True, help="Spike/novelty alerts")
show_shap = st.sidebar.checkbox("Show SHAP Explanations", True, help="Explainable AI for model predictions")
show_stats = st.sidebar.checkbox("Show Statistical Trends", True, help="Mann-Kendall Trend Tests")

st.sidebar.markdown("---")
st.sidebar.info("Expected columns: location, year, pathogen, antibiotic, n_tested, n_resistant (+ optional gene).")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
df = None

if data_source == "Upload CSV":
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif data_source == "WHO GLASS AMR 2022 (Official)":
    try:
        df = pd.read_csv("who_glass_amr.csv")
        st.info("Using official WHO GLASS AMR 2022 dataset.")
    except FileNotFoundError:
        st.error("File 'who_glass_amr.csv' not found. Please run preprocessing.")

if df is not None:

    # ---------- LOAD DATA ----------
    # df is already loaded above

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------- CHECK REQUIRED COLUMNS ----------
    required_cols = ["location", "year", "pathogen", "antibiotic", "n_tested", "n_resistant"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.stop()

    # ---------- BASIC CLEANING ----------
    # Enforce numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["n_tested"] = pd.to_numeric(df["n_tested"], errors="coerce")
    df["n_resistant"] = pd.to_numeric(df["n_resistant"], errors="coerce")

    # Drop invalid numeric rows
    df = df.dropna(subset=["year", "n_tested", "n_resistant"])
    df = df[df["n_tested"] > 0]

        # Compute resistance rate
    df["resistance_rate"] = df["n_resistant"] / df["n_tested"]
    df = df[(df["resistance_rate"] >= 0) & (df["resistance_rate"] <= 1)]

    if df.empty:
        st.error("All rows were invalid after cleaning (check n_tested/n_resistant).")
        st.stop()

    # ===== NEW: Add Phylogeny Weighting =====
    st.info("🧬 Calculating phylogeny-based sample weights...")
    df = add_phylogeny_weights(df)
    st.success("✅ Phylogeny weights calculated!")

    st.subheader("⚙️ Processed Data (with Phylogeny Weights)")
    st.dataframe(df.head())

    # ---------------------------------------------------------
    # MACHINE LEARNING MODEL (XGBoost)
    # ---------------------------------------------------------
    st.subheader("🤖 AMR Prediction Model (XGBoost)")

    X = df[["location", "year", "pathogen", "antibiotic"]].copy()
    y = df["resistance_rate"].copy()

    # Preprocessing: one-hot encode categorical, passthrough year
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["location", "pathogen", "antibiotic"]),
            ("num", "passthrough", ["year"]),
        ]
    )

    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

        # Guard: need at least 5 rows to split for ML
    if len(df) < 5:
        st.warning("Not enough rows (<5) for train/test split. Showing basic stats only.")
        pipeline.fit(X, y)
        y_pred_full = pipeline.predict(X)
        st.write("Mean predicted resistance rate:", float(y_pred_full.mean()))
    else:
                 # ===== NEW: Use Phylogeny Weights =====
        weights = df['phylogeny_weight'].values
        
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights,
            test_size=0.2, random_state=42
        )
        
        # Train model WITH sample weights
        pipeline.fit(X_train, y_train, **{"model__sample_weight": w_train})


        y_pred = pipeline.predict(X_test)


        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📈 R² Score", f"{r2:.3f}")
        with c2:
            st.metric("📉 RMSE", f"{rmse:.3f}")

    # ---------------------------------------------------------
    # SHAP EXPLANATIONS
    # ---------------------------------------------------------
    if show_shap:
        st.subheader("🧠 Explainable AI (SHAP)")
        st.write("Explaining why the model predicts high/low resistance.")
        
        # We need the transformed feature names for SHAP
        # Fit preprocessor first to get feature names
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["location", "pathogen", "antibiotic"])
        feature_names = list(cat_features) + ["year"]
        
        # Create a dataframe for SHAP
        X_shap_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Explain the model
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        shap_values = explainer(X_shap_df)
        
        # Summary Plot
        st.write("#### Feature Importance (SHAP Summary)")
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, X_shap_df, show=False)
        st.pyplot(fig_shap)

    # ---------------------------------------------------------
    # 5. STATISTICAL TRENDS (Mann-Kendall)
    # ---------------------------------------------------------
    if show_stats:
        st.header("5. 📉 Statistical Trend Analysis (Mann-Kendall)")
        st.write("Validating trends statistically (p < 0.05 indicates significance).")
        
        # We need time series: Location + Pathogen + Antibiotic
        # Group by year
        trend_results = []
        
        # Get unique combinations
        combos = df[["location", "pathogen", "antibiotic"]].drop_duplicates()
        
        for idx, row in combos.iterrows():
            loc, path, ab = row["location"], row["pathogen"], row["antibiotic"]
            subset = df[(df["location"] == loc) & (df["pathogen"] == path) & (df["antibiotic"] == ab)]
            subset = subset.sort_values("year")
            
            if len(subset) >= 3: # MK test needs at least a few points
                # Calculate resistance rate
                subset["rate"] = subset["n_resistant"] / subset["n_tested"]
                try:
                    result = mk.original_test(subset["rate"])
                    trend_results.append({
                        "Location": loc,
                        "Pathogen": path,
                        "Antibiotic": ab,
                        "Trend": result.trend,
                        "Slope": result.slope,
                        "P-Value": result.p,
                        "Significant": result.p < 0.05
                    })
                except:
                    pass
        
        if trend_results:
            trend_df = pd.DataFrame(trend_results)
            st.dataframe(trend_df)
            
            # Highlight significant increasing trends
            worsening = trend_df[(trend_df["Significant"] == True) & (trend_df["Slope"] > 0)]
            if not worsening.empty:
                st.error(f"⚠️ Found {len(worsening)} statistically significant WORSENING trends!")
                st.dataframe(worsening)
            else:
                st.success("No statistically significant worsening trends found in this dataset.")
        else:
            st.info("Not enough data points for trend analysis.")

    # ---------------------------------------------------------
    # 6. LITERATURE-STYLE VISUALIZATION (Lancet 2024 Style)
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("6. 📊 Global Burden Analysis (Lancet Style)")
    st.info("Visualizing resistance rates by pathogen, mimicking the 'Global burden of bacterial AMR' (Lancet, 2024) chart styles using the official WHO GLASS 2022 data.")
    
    # Calculate aggregated resistance rates by pathogen
    # We want to show the median resistance rate with an interquartile range (IQR) to show variability across countries
    df["resistance_rate"] = df["n_resistant"] / df["n_tested"]
    
    # Group by Pathogen
    pathogen_stats = df.groupby("pathogen")["resistance_rate"].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        count="count"
    ).sort_values("median", ascending=True).reset_index()
    
    # Filter out pathogens with very few data points
    pathogen_stats = pathogen_stats[pathogen_stats["count"] > 10]
    
    if not pathogen_stats.empty:
        fig_lit, ax_lit = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart with error bars (representing IQR)
        # Calculate error bars: [median - q1, q3 - median]
        errors = [
            pathogen_stats["median"] - pathogen_stats["q1"],
            pathogen_stats["q3"] - pathogen_stats["median"]
        ]
        
        y_pos = np.arange(len(pathogen_stats))
        
        ax_lit.errorbar(
            pathogen_stats["median"], 
            y_pos, 
            xerr=errors, 
            fmt='o', 
            color='black', 
            ecolor='gray', 
            capsize=5, 
            label='Median Rate (with IQR)'
        )
        ax_lit.barh(y_pos, pathogen_stats["median"], align='center', alpha=0.6, color='#d62728')
        
        ax_lit.set_yticks(y_pos)
        ax_lit.set_yticklabels(pathogen_stats["pathogen"], fontsize=10, fontweight='bold')
        ax_lit.set_xlabel('Resistance Rate (Proportion)', fontsize=12)
        ax_lit.set_title('Global Resistance Burden by Pathogen (Median & IQR)', fontsize=14)
        ax_lit.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add data labels
        for i, v in enumerate(pathogen_stats["median"]):
            ax_lit.text(v + 0.02, i, f"{v:.2f}", color='black', va='center', fontsize=9)
            
        st.pyplot(fig_lit)
        st.caption("Figure 1: Median resistance rates for priority pathogens across all reporting countries. Error bars represent the Interquartile Range (IQR), showing the variability in resistance burdens globally. Source: WHO GLASS 2022.")
    else:
        st.warning("Not enough data to generate the literature-style plot.")

    # ---------------------------------------------------------
    # MULTIDIMENSIONAL RISK SCORE
    # ---------------------------------------------------------
    if show_risk:
        st.subheader("⚠️ Multidimensional Biosafety Risk Score")
        st.write("Risk = (Predicted Rate * 0.6) + (Trend Slope * 0.4) + Penalty")

        # Predict for all rows (full dataset)
        try:
            df["predicted_rate"] = pipeline.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
            
        if "slope" not in df.columns:
             df["slope"] = 0

        # Normalize slope (simple min-max or sigmoid could be better, but linear for now)
        # Slope is usually small (e.g. 0.05 per year). We scale it up.
        
        # Formula: 
        # Base Risk: Predicted Rate (0-1) * 100
        # Trend Bonus: Slope * 500 (e.g., 0.1 slope -> +50 risk)
        # Clip to 0-100
        
        df["base_risk"] = df["predicted_rate"] * 100
        df["trend_risk"] = df["slope"] * 500
        
        df["risk_score"] = (df["base_risk"] + df["trend_risk"]).clip(0, 100)

        st.write("Sample of risk-scored rows:")
        st.dataframe(
            df[["location", "year", "pathogen", "antibiotic", "risk_score", "slope"]].head()
        )

        high_risk = df.sort_values("risk_score", ascending=False).head(10)
        st.write("### 🔥 Top 10 Highest-Risk Combinations")
        st.table(high_risk[["location", "year", "pathogen", "antibiotic", "risk_score", "slope"]])

    # ---------------------------------------------------------
    # NOVELTY / ANOMALY DETECTION
    # ---------------------------------------------------------
    if show_novelty:

        st.subheader("🧪 Novelty & Anomaly Detection")

        # ---------- 1) STATISTICAL NOVELTY (Z-SCORE) ----------
        st.markdown("#### 1️⃣ Statistical Novelty (Z-score spikes)")
        if df["resistance_rate"].std() == 0 or len(df) < 3:
            st.info("Not enough variability for statistical anomaly detection.")
        else:
            df["z_score"] = (
                (df["resistance_rate"] - df["resistance_rate"].mean())
                / df["resistance_rate"].std()
            )
            df["statistical_novelty"] = df["z_score"].abs() > 2.5
            novel_stats = df[df["statistical_novelty"]]

            if not novel_stats.empty:
                st.error("⚠ Statistical anomalies detected (unusual resistance spikes):")
                st.dataframe(
                    novel_stats[
                        ["location", "year", "pathogen", "antibiotic", "resistance_rate", "z_score"]
                    ]
                )
            else:
                st.success("No strong statistical resistance spikes detected.")

        # ---------- 2) ML-BASED NOVELTY (Isolation Forest) ----------
        st.markdown("#### 2️⃣ ML Novelty (Isolation Forest)")
        if len(df) < 10:
            st.info("Need at least 10 rows for Isolation Forest. Skipping ML-based novelty.")
        else:
            iso_features = df[["year", "resistance_rate"]].copy()

            try:
                iso = IsolationForest(
                    contamination=0.1, random_state=42
                )  # 10% assumed anomaly
                df["ml_novelty_score"] = iso.fit_predict(iso_features)
                ml_novelty = df[df["ml_novelty_score"] == -1]

                if not ml_novelty.empty:
                    st.error("⚠ ML-based novel patterns detected (multidimensional anomalies):")
                    st.dataframe(
                        ml_novelty[
                            [
                                "location",
                                "year",
                                "pathogen",
                                "antibiotic",
                                "resistance_rate",
                            ]
                        ]
                    )
                else:
                    st.success("No ML-detected anomalies in the current dataset.")
            except Exception as e:
                st.warning(f"Isolation Forest could not run: {e}")

        # ---------- 3) GENOMIC NOVELTY (NEW AMR GENES) ----------
        st.markdown("#### 3️⃣ Genomic Novelty (new / unusual AMR genes)")
        if "gene" not in df.columns:
            st.info("No 'gene' column found. Add a 'gene' column to enable genomic novelty detection.")
        else:
            # Simple reference set of frequently reported AMR genes
            reference_gene_list = {
                "mcr-1", "mcr-2", "blaNDM", "blaCTX-M", "blaKPC",
                "tetA", "vanA", "OXA-48"
            }

            df["genomic_novelty"] = ~df["gene"].isin(reference_gene_list)
            novel_genes = df[df["genomic_novelty"]]

            if not novel_genes.empty:
                st.error("⚠ Potential novel or uncommon AMR genes detected:")
                st.dataframe(
                    novel_genes[
                        ["location", "year", "pathogen", "antibiotic", "gene"]
                    ].drop_duplicates()
                )
            else:
                st.success("No novel AMR genes detected compared to the reference list.")

    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
        # ---------------------------------------------------------
    # PHYLOGENY WEIGHT ANALYSIS (NEW TAB)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🧬 Phylogeny Weight Analysis")
    st.write("Understanding sample importance based on epidemiological patterns (location, year, pathogen).")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_weight = df['phylogeny_weight'].mean()
        st.metric(
            "Avg Weight",
            f"{avg_weight:.3f}",
            help="Closer to 1.0 = more unique samples"
        )
    
    with col2:
        high_weight = (df['phylogeny_weight'] > 0.7).sum()
        st.metric("Unique Samples", f"{high_weight}", help="High importance samples")
    
    with col3:
        low_weight = (df['phylogeny_weight'] < 0.3).sum()
        st.metric("Outbreak Samples", f"{low_weight}", help="Redundant/clonal samples")
    
    with col4:
        st.metric(
            "Total Samples",
            f"{len(df)}",
            help="All surveillance records"
        )
    
    # Show weight distribution by location
    st.markdown("#### Weight Distribution by Location")
    try:
        location_weights = df.groupby('location')['phylogeny_weight'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        fig_loc, ax_loc = plt.subplots(figsize=(10, 5))
        location_weights['mean'].plot(kind='barh', ax=ax_loc, color='steelblue')
        ax_loc.set_xlabel('Average Phylogeny Weight')
        ax_loc.set_title('Sample Importance by Location')
        st.pyplot(fig_loc)
    except Exception as e:
        st.warning(f"Could not create location weights plot: {e}")
    
    # Show detected outbreak clusters
    st.markdown("#### 🚨 Detected Outbreak Clusters (weight < 0.3)")
    outbreak_data = df[df['phylogeny_weight'] < 0.3].groupby(['location', 'pathogen', 'year']).agg({
        'phylogeny_weight': 'mean',
        'resistance_rate': 'mean',
        'n_tested': 'sum'
    }).sort_values('phylogeny_weight')
    
    if not outbreak_data.empty:
        st.error(f"⚠️ Found {len(outbreak_data)} potential outbreak clusters!")
        st.dataframe(outbreak_data)
    else:
        st.success("✅ No significant outbreak clusters detected.")
    
    # Show weight statistics
    st.markdown("#### Weight Statistics by Pathogen")
    try:
        pathogen_weights = df.groupby('pathogen')['phylogeny_weight'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        st.dataframe(pathogen_weights)
    except Exception as e:
        st.warning(f"Could not generate pathogen statistics: {e}")




    if show_plots:
        st.subheader("📊 Visual Analytics")

        # ---------- LINE PLOT ----------
        st.markdown("#### Resistance over time by pathogen")
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            sns.lineplot(
                data=df.sort_values("year"),
                x="year",
                y="resistance_rate",
                hue="pathogen",
                marker="o",
                ax=ax,
            )
            ax.set_ylabel("Resistance rate")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create line plot: {e}")

        # ---------- HEATMAP ----------
        st.markdown("#### Pathogen × Antibiotic mean resistance heatmap")
        try:
            pivot = df.pivot_table(
                values="resistance_rate",
                index="pathogen",
                columns="antibiotic",
                aggfunc="mean",
            )
            if pivot.empty:
                st.info("Not enough data to build a heatmap.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot, annot=False, cmap="viridis")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create heatmap: {e}")

else:
    st.info("📥 Upload a CSV file to begin analysis. Use the sample datasets I provided.")
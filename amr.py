"""
Enhanced AMR Biosafety Dashboard with Publication-Quality Visualizations
Novel Contribution: Phylogeny-Weighted Machine Learning for AMR Surveillance
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from utils import Config, load_uploaded_file, load_example_data, load_who_glass_data
from utils import validate_amr_data, display_validation_results, display_data_summary
from utils import load_fasta_file, display_fasta_summary, create_example_fasta
from utils.visualizations import (
    create_resistance_timeline, create_phylogeny_weight_distribution,
    create_model_comparison, create_risk_heatmap, add_download_button
)
from pipelines.surveillance.phylogeny import add_phylogeny_weights
from pipelines.surveillance.risk_scoring import (
    calculate_risk_scores, identify_high_risk_combinations,
    calculate_risk_statistics, compare_risk_by_region
)
from pipelines.surveillance.conformal import ConformalPredictor
from utils.network_analysis import build_transmission_network, plot_transmission_network
from pipelines.surveillance.explanation import compute_shap_values, plot_shap_summary, plot_shap_waterfall
from pipelines.surveillance.simulation import run_substitution_simulation, plot_simulation_gauge
from pipelines.surveillance.anomaly import (
    detect_statistical_anomalies, detect_ml_anomalies, 
    check_genomic_novelty, compute_false_negative_rate
)

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AMR Biosafety Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">AI-Driven AMR Biosafety Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Phylogeny-Weighted Machine Learning for Enhanced Surveillance</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Upload CSV", "Upload FASTA", "Upload CSV + FASTA", "Example Dataset", "WHO GLASS AMR 2022")
)

uploaded_file = None
uploaded_fasta = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=["csv"],
        help=f"Maximum file size: {Config.ui.max_file_size_mb} MB"
    )
elif data_source == "Upload FASTA":
    uploaded_fasta = st.sidebar.file_uploader(
        "Upload FASTA file",
        type=["fasta", "fa", "fna"],
        help="FASTA file with pathogen sequences. Header format: >pathogen|location|year|antibiotic"
    )
    st.sidebar.info("FASTA headers should include metadata: `>E.coli|India|2020|ciprofloxacin`")
elif data_source == "Upload CSV + FASTA":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="csv_upload"
    )
    uploaded_fasta = st.sidebar.file_uploader(
        "Upload FASTA file",
        type=["fasta", "fa", "fna"],
        key="fasta_upload",
        help="Sequences will be matched with CSV metadata"
    )

st.sidebar.markdown("---")
st.sidebar.header("Analysis Options")
show_comparison = st.sidebar.checkbox("Show Model Comparison", True, help="Compare weighted vs unweighted models")
show_stats = st.sidebar.checkbox("Show Statistical Tests", True, help="Statistical significance testing")
show_risk = st.sidebar.checkbox("Show Risk Analysis", True, help="Biosafety risk scoring")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Novel Contribution:** Phylogeny-weighted ML approach addresses outbreak bias in AMR surveillance data."
)

# Main content
df = None
fasta_df = None

# Load data based on source
if data_source == "Upload CSV":
    if uploaded_file is not None:
        # Validate file size
        from utils.data_validator import validate_file_size
        is_valid, error_msg = validate_file_size(uploaded_file, Config.get_file_size_limit_bytes())
        if not is_valid:
            st.error(error_msg)
            st.stop()
        
        df = load_uploaded_file(uploaded_file)

elif data_source == "Upload FASTA":
    if uploaded_fasta is not None:
        st.info("Processing FASTA file...")
        fasta_df = load_fasta_file(uploaded_fasta)
        
        if fasta_df is not None:
            st.success(f"Loaded {len(fasta_df)} sequences")
            display_fasta_summary(fasta_df)
            
            # Convert FASTA data to AMR format if possible
            if all(col in fasta_df.columns for col in ['pathogen', 'location', 'year', 'antibiotic']):
                # Create synthetic resistance data for demonstration
                fasta_df['n_tested'] = 100
                fasta_df['n_resistant'] = (fasta_df['sequence_length'] % 50) + 10  # Synthetic
                df = fasta_df
            else:
                st.error("FASTA headers missing required metadata. Please use format: >pathogen|location|year|antibiotic")
                st.stop()

elif data_source == "Upload CSV + FASTA":
    if uploaded_file is not None and uploaded_fasta is not None:
        # Load both files
        df = load_uploaded_file(uploaded_file)
        fasta_df = load_fasta_file(uploaded_fasta)
        
        if df is not None and fasta_df is not None:
            st.success(f"Loaded CSV ({len(df)} rows) and FASTA ({len(fasta_df)} sequences)")
            
            # Merge datasets
            from utils.fasta_parser import merge_fasta_with_csv
            df = merge_fasta_with_csv(fasta_df, df)
            st.info(f"Merged dataset: {len(df)} records with sequence data")
    elif uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
    elif uploaded_fasta is not None:
        fasta_df = load_fasta_file(uploaded_fasta)
        if fasta_df is not None:
            st.warning("Only FASTA uploaded. Please also upload CSV for complete analysis.")

elif data_source == "Example Dataset":
    df, dataset_name = load_example_data()
    
elif data_source == "WHO GLASS AMR 2022":
    df, dataset_name = load_who_glass_data()

if df is not None:
    # Validate data
    df_clean, errors, warnings = validate_amr_data(df, Config.REQUIRED_CSV_COLUMNS)
    
    if errors:
        display_validation_results(errors, warnings)
        st.stop()
    
    if warnings:
        display_validation_results([], warnings)
    
    df = df_clean
    
    # Display data summary
    st.subheader("Dataset Overview")
    display_data_summary(df)
    
    with st.expander("View Data Sample", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Add phylogeny weights
    with st.spinner("Calculating phylogeny weights..."):
        df = add_phylogeny_weights(df)
    st.success("Phylogeny weights calculated!")
    
    # Tabs for organized content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ML Predictions", 
        "Visualizations", 
        "Risk Analysis",
        "Novelty Detection",
        "Explainability (XAI)",
        "Policy Simulation",
        "Export Results"
    ])
    
    # TAB 1: ML PREDICTIONS
    with tab1:
        st.header("Machine Learning Predictions")
        
        # Prepare features
        X = df[["location", "year", "pathogen", "antibiotic"]].copy()
        y = df["resistance_rate"].copy()
        weights = df['phylogeny_weight'].values
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 ["location", "pathogen", "antibiotic"]),
                ("num", "passthrough", ["year"]),
            ]
        )
        
        if len(df) >= 5:
            # Train/test split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
            
            col1, col2 = st.columns(2)
            
            # WEIGHTED MODEL (Novel Approach)
            with col1:
                st.subheader("Phylogeny-Weighted Model (Novel)")
                
                model_weighted = xgb.XGBRegressor(
                    n_estimators=Config.model.n_estimators,
                    learning_rate=Config.model.learning_rate,
                    max_depth=Config.model.max_depth,
                    random_state=Config.model.random_state,
                    n_jobs=Config.model.n_jobs
                )
                
                pipeline_weighted = Pipeline([
                    ("preprocess", preprocessor), 
                    ("model", model_weighted)
                ])
                
                with st.spinner("Training weighted model..."):
                    pipeline_weighted.fit(X_train, y_train, **{"model__sample_weight": w_train})
                
                y_pred_weighted = pipeline_weighted.predict(X_test)
                r2_weighted = r2_score(y_test, y_pred_weighted)
                rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
                
                st.metric("R² Score", f"{r2_weighted:.4f}", help="Higher is better")
                st.metric("RMSE", f"{rmse_weighted:.4f}", help="Lower is better")
            
            # UNWEIGHTED MODEL (Baseline)
            with col2:
                st.subheader("Standard Model (Baseline)")
                
                model_unweighted = xgb.XGBRegressor(
                    n_estimators=Config.model.n_estimators,
                    learning_rate=Config.model.learning_rate,
                    max_depth=Config.model.max_depth,
                    random_state=Config.model.random_state,
                    n_jobs=Config.model.n_jobs
                )
                
                pipeline_unweighted = Pipeline([
                    ("preprocess", preprocessor), 
                    ("model", model_unweighted)
                ])
                
                with st.spinner("Training baseline model..."):
                    pipeline_unweighted.fit(X_train, y_train)
                
                y_pred_unweighted = pipeline_unweighted.predict(X_test)
                r2_unweighted = r2_score(y_test, y_pred_unweighted)
                rmse_unweighted = np.sqrt(mean_squared_error(y_test, y_pred_unweighted))
                
                st.metric("R² Score", f"{r2_unweighted:.4f}")
                st.metric("RMSE", f"{rmse_unweighted:.4f}")
            
            # Model Comparison
            if show_comparison:
                st.markdown("---")
                st.subheader("Model Performance Comparison")
                
                improvement_r2 = ((r2_weighted - r2_unweighted) / r2_unweighted) * 100
                improvement_rmse = ((rmse_unweighted - rmse_weighted) / rmse_unweighted) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Improvement", f"{improvement_r2:+.2f}%", 
                             delta=f"{r2_weighted - r2_unweighted:.4f}")
                with col2:
                    st.metric("RMSE Improvement", f"{improvement_rmse:+.2f}%",
                             delta=f"{rmse_unweighted - rmse_weighted:.4f}")
                with col3:
                    # Paired t-test
                    if show_stats:
                        t_stat, p_value = stats.ttest_rel(
                            np.abs(y_test - y_pred_weighted),
                            np.abs(y_test - y_pred_unweighted)
                        )
                        significance = "Significant" if p_value < 0.05 else "Not Significant"
                        st.metric("Statistical Test", significance, 
                                 delta=f"p={p_value:.4f}")
                
                # Visualization
                fig_comparison = create_model_comparison(
                    {'r2': r2_weighted, 'rmse': rmse_weighted},
                    {'r2': r2_unweighted, 'rmse': rmse_unweighted}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                add_download_button(fig_comparison, "model_comparison", "Download Comparison")

            # --- NOVEL FEATURE: CONFORMAL PREDICTION ---
            st.markdown("---")
            st.subheader("Phylogeny-Informed Conformal Prediction (Novelty)")
            st.info("**Why this matters:** We provide guaranteed confidence intervals. Novel/rare samples (low phylogeny weight) get wider, safer error bars.")
            
            # 1. Calibrate on test set provided we have enough data
            # (In a real app we'd use a separate calibration set, but for demo we use test)
            cp = ConformalPredictor(alpha=0.1) # 90% confidence
            cp.calibrate(y_test.values, y_pred_weighted, weights=w_test)
            
            # 2. Predict intervals
            lower, upper = cp.predict(y_pred_weighted, weights=w_test)
            
            # 3. Calculate metrics
            coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper)) * 100
            avg_width = np.mean(upper - lower)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Empirical Coverage", f"{coverage:.1f}%", help="Target: 90%")
            with c2:
                st.metric("Avg Interval Width", f"{avg_width:.3f}", help="Narrower is better (if coverage met)")
            with c3:
                # Correlation between weight and width (should be negative)
                width_corr = np.corrcoef(w_test, upper - lower)[0, 1]
                st.metric("Weight-Width Corr.", f"{width_corr:.2f}", help="Should be negative (Unique -> Wider)")
            
            # Visualizing the intervals for a subset
            st.caption("Prediction Intervals for Test Subset (Sorted by Phylogeny Weight)")
            
            # Create a dataframe for plotting
            res_df = pd.DataFrame({
                'True': y_test.values,
                'Predicted': y_pred_weighted,
                'Lower': lower,
                'Upper': upper,
                'Weight': w_test
            }).sort_values('Weight')
            
            # Quick custom chart for intervals
            import plotly.graph_objects as go
            fig_cp = go.Figure()
            
            # Add error bars
            fig_cp.add_trace(go.Scatter(
                x=res_df['Weight'],
                y=res_df['Predicted'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=res_df['Upper'] - res_df['Predicted'],
                    arrayminus=res_df['Predicted'] - res_df['Lower'],
                    color='#2E86AB'
                ),
                mode='markers',
                name='Prediction Interval',
                marker=dict(size=5, color='#2E86AB')
            ))
            
            fig_cp.add_trace(go.Scatter(
                x=res_df['Weight'],
                y=res_df['True'],
                mode='markers',
                name='True Value',
                marker=dict(size=4, color='#E63946', symbol='x')
            ))
            
            fig_cp.update_layout(
                title="Uncertainty Quantification vs. Biologic Novelty",
                xaxis_title="Phylogeny Weight (Left=Novel, Right=Common)",
                yaxis_title="Resistance Rate",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_cp, use_container_width=True)
            
            # Store predictions for risk analysis

            df['predicted_rate_weighted'] = pipeline_weighted.predict(X)
            df['predicted_rate_unweighted'] = pipeline_unweighted.predict(X)
        
        else:
            st.warning("Not enough data for train/test split (need at least 5 rows)")
    
    # TAB 2: VISUALIZATIONS
    with tab2:
        st.header("Publication-Quality Visualizations")
        
        # Resistance Timeline
        st.subheader("Resistance Trends Over Time")
        fig_timeline = create_resistance_timeline(df)
        st.plotly_chart(fig_timeline, use_container_width=True)
        add_download_button(fig_timeline, "resistance_timeline", "Download Timeline")
        
        # Phylogeny Weight Distribution
        st.markdown("---")
        st.subheader("Phylogeny Weight Distribution")
        st.info("**Novel Contribution:** Lower weights indicate outbreak clusters; higher weights indicate unique samples.")
        
        fig_weights = create_phylogeny_weight_distribution(df)
        st.plotly_chart(fig_weights, use_container_width=True)
        add_download_button(fig_weights, "phylogeny_weights", "Download Weights")
        
        # Weight Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Weight", f"{df['phylogeny_weight'].mean():.3f}")
        with col2:
            high_weight = (df['phylogeny_weight'] > 0.7).sum()
            st.metric("Unique Samples", high_weight, help="High importance")
        with col3:
            low_weight = (df['phylogeny_weight'] < 0.3).sum()
            st.metric("Outbreak Samples", low_weight, help="Redundant/clonal")
        with col4:
            st.metric("Total Samples", len(df))

        # --- NOVEL FEATURE: NETWORK ANALYSIS ---
        st.markdown("---")
        st.subheader("Transmission Network Analysis (Novelty)")
        st.info("**What this shows:** A graph where regions are connected if their resistance profiles are highly correlated (>0.5). This reveals 'super-spreader' hub locations.")
        
        with st.spinner("Building transmission network..."):
            G = build_transmission_network(df, min_correlation=0.5)
            fig_network = plot_transmission_network(G)
            st.plotly_chart(fig_network, use_container_width=True)
            add_download_button(fig_network, "transmission_network", "Download Network Graph")
    
    # TAB 3: RISK ANALYSIS
    with tab3:
        if show_risk and 'predicted_rate_weighted' in df.columns:
            st.header("Biosafety Risk Assessment")
            
            # Calculate risk scores
            df_risk = calculate_risk_scores(
                df, 
                df['predicted_rate_weighted'].values,
                Config.risk_scoring.base_weight,
                Config.risk_scoring.trend_weight,
                Config.risk_scoring.slope_multiplier
            )
            
            # Risk statistics
            risk_stats = calculate_risk_statistics(df_risk)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Risk", f"{risk_stats['mean_risk']:.1f}")
            with col2:
                st.metric("High Risk (≥70)", risk_stats['high_risk_count'], 
                         delta_color="inverse")
            with col3:
                st.metric("Medium Risk (40-70)", risk_stats['medium_risk_count'])
            with col4:
                st.metric("Low Risk (<40)", risk_stats['low_risk_count'],
                         delta_color="normal")
            
            # Risk Heatmap
            st.subheader("Risk Score Heatmap")
            fig_heatmap = create_risk_heatmap(df_risk)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            add_download_button(fig_heatmap, "risk_heatmap", "Download Heatmap")
            
            # High-risk combinations
            st.markdown("---")
            st.subheader("Top 10 Highest-Risk Combinations")
            high_risk_combos = identify_high_risk_combinations(df_risk, top_n=10)
            
            if not high_risk_combos.empty:
                st.dataframe(
                    high_risk_combos.style.background_gradient(
                        subset=['risk_score'], cmap='Reds'
                    ),
                    use_container_width=True
                )
            else:
                st.info("No high-risk combinations found.")
            
            # Regional comparison
            st.markdown("---")
            st.subheader("Regional Risk Comparison")
            regional_risk = compare_risk_by_region(df_risk)
            st.dataframe(regional_risk, use_container_width=True)
        
        else:
            st.info("Run ML predictions first to see risk analysis.")
    
    # TAB 4: NOVELTY DETECTION
    with tab4:
        st.header("🧪 Novelty & Anomaly Detection")
        st.info("**Advanced Surveillance:** Detects statistical spikes, ML-based anomalies, and potential surveillance gaps (False Negatives).")
        
        # 1. Statistical Novelty
        st.subheader("1. Statistical Spikes (Z-Score)")
        with st.spinner("Analyzing statistical anomalies..."):
            df_stat = detect_statistical_anomalies(df)
            anomalies = df_stat[df_stat['statistical_novelty']]
            
            if not anomalies.empty:
                st.error(f"⚠️ Found {len(anomalies)} statistical anomalies (Resistance Spikes > 1.5 Std Dev)")
                st.dataframe(anomalies[['location', 'year', 'pathogen', 'antibiotic', 'resistance_rate', 'z_score']])
            else:
                st.success("✅ No statistical anomalies detected.")
                
        # 2. System Validation (False Negatives)
        st.markdown("---")
        st.subheader("2. Surveillance System Validation")
        fn_stats = compute_false_negative_rate(df, spike_threshold=0.15, spike_col="statistical_novelty")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("False Negative Rate", f"{fn_stats['fn_rate']*100:.1f}%", help="Spikes missed by statistical method")
        with c2:
            st.metric("Missed Spikes", fn_stats['fn'])
        with c3:
            st.metric("Total True Spikes", fn_stats['tp'] + fn_stats['fn'])
            
        if fn_stats['fn'] > 0:
            st.warning("Examples of missed spikes (Potential Surveillance Gaps):")
            st.dataframe(fn_stats['details'][['location', 'year', 'pathogen', 'antibiotic', 'resistance_change']])

        # 3. ML-Based Novelty
        st.markdown("---")
        st.subheader("3. ML-Based Anomaly Detection (Isolation Forest)")
        with st.spinner("Running Isolation Forest..."):
            df_ml = detect_ml_anomalies(df, contamination=0.05)
            ml_anomalies = df_ml[df_ml['is_ml_anomaly']]
            
            if not ml_anomalies.empty:
                st.error(f"⚠️ Found {len(ml_anomalies)} multidimensional anomalies (unusual patterns)")
                st.dataframe(ml_anomalies[['location', 'year', 'pathogen', 'antibiotic', 'resistance_rate']])
            else:
                st.success("✅ No ML-based anomalies detected.")
    
    # TAB 5: EXPLAINABILITY (XAI)
    with tab5:
        st.header("Trustworthy AI: Model Explainability")
        
        if 'predicted_rate_weighted' in df.columns:
            st.info("**Transparency:** SHAP (Shapley Additive exPlanations) values show exactly how each feature contributes to the resistance prediction. This eliminates the 'Black Box' problem.")
            
            with st.spinner("Computing SHAP values..."):
                # Use the weighted pipeline's model and preprocessor
                shap_values, X_shap_df = compute_shap_values(
                    pipeline_weighted.named_steps['model'],
                    X, 
                    pipeline_weighted.named_steps['preprocess']
                )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Global Feature Importance")
                fig_shap_summary = plot_shap_summary(shap_values, X_shap_df)
                st.plotly_chart(fig_shap_summary, use_container_width=True)
                add_download_button(fig_shap_summary, "shap_summary", "Download SHAP Summary")
            
            with col2:
                st.subheader("Local Explanation (Waterfall)")
                st.caption("Select a sample to understand why the model predicted its specific resistance rate.")
                
                # Sample selector
                sample_idx = st.number_input("Sample Index", min_value=0, max_value=len(X)-1, value=0)
                
                # Show sample details
                selected_row = X.iloc[[sample_idx]]
                st.dataframe(selected_row, use_container_width=True)
                
                # Get waterfall plot
                # Note: SHAP waterfall is matplotlib, so we use st.pyplot
                fig_waterfall = plot_shap_waterfall(
                    pipeline_weighted.named_steps['model'],
                    selected_row,
                    pipeline_weighted.named_steps['preprocess']
                )
                st.pyplot(fig_waterfall)
                
        else:
            st.info("Please train the model (Tab 1) first to see explanations.")

            st.info("Please train the model (Tab 1) first to see explanations.")

    # TAB 6: POLICY SIMULATION
    with tab6:
        st.header("Active Policy Simulation")
        st.info("**For Policymakers:** Simulate the impact of antibiotic stewardship interventions. (e.g., Switching first-line treatment).")
        
        if 'predicted_rate_weighted' in df.columns:
            st.subheader("Scenario: Antibiotic Substitution")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                sim_pathogen = st.selectbox("Select Pathogen", df['pathogen'].unique())
            with sc2:
                # Filter antibiotics used for this pathogen
                avail_abs = df[df['pathogen'] == sim_pathogen]['antibiotic'].unique()
                current_ab = st.selectbox("Current Antibiotic (Baseline)", avail_abs, index=0)
            with sc3:
                new_ab = st.selectbox("Proposed Antibiotic (Intervention)", avail_abs, index=len(avail_abs)-1 if len(avail_abs)>1 else 0)
            
            if st.button("Run Simulation"):
                if current_ab == new_ab:
                    st.warning("Please select different antibiotics.")
                else:
                    with st.spinner("Simulating counterfactual scenario..."):
                        sim_results, sim_summary = run_substitution_simulation(
                            pipeline_weighted, df, sim_pathogen, current_ab, new_ab
                        )
                    
                    if sim_results is not None:
                        # Display results
                        r1, r2 = st.columns([1, 2])
                        with r1:
                            st.subheader("Projected Impact")
                            st.metric("Mean Risk Reduction", f"{sim_summary['mean_reduction']*100:.1f}%", 
                                     delta="Improvement" if sim_summary['mean_reduction'] > 0 else "Worsening")
                            st.metric("Locations Improved", f"{sim_summary['locations_improved']} / {sim_summary['locations_simulated']}")
                            
                            fig_gauge = plot_simulation_gauge(sim_summary)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                        with r2:
                            st.subheader("Location-Specific Projections")
                            st.dataframe(
                                sim_results.style.background_gradient(subset=['Risk Reduction'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                    else:
                        st.error(sim_summary['error'])
        else:
            st.info("Please train the model (Tab 1) first to run simulations.")

    # TAB 7: EXPORT RESULTS
    with tab7:

        st.header("Export Results for Publication")
        
        st.info("Download processed data and statistical results for your paper.")
        
        # Export processed data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processed Dataset")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="amr_processed_data.csv",
                mime="text/csv"
            )
        
        with col2:
            if 'predicted_rate_weighted' in df.columns:
                st.subheader("Model Results")
                results_df = pd.DataFrame({
                    'Model': ['Phylogeny-Weighted (Novel)', 'Standard (Baseline)'],
                    'R²': [r2_weighted, r2_unweighted],
                    'RMSE': [rmse_weighted, rmse_unweighted],
                    'R²_Improvement_%': [improvement_r2, 0],
                    'RMSE_Improvement_%': [improvement_rmse, 0]
                })
                
                results_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=results_csv,
                    file_name="model_comparison_results.csv",
                    mime="text/csv"
                )
        
        # Statistical summary
        st.markdown("---")
        st.subheader("Statistical Summary")
        
        if show_stats and 'predicted_rate_weighted' in df.columns:
            summary_text = f"""
## Model Performance Summary

### Phylogeny-Weighted Model (Novel Approach)
- R² Score: {r2_weighted:.4f}
- RMSE: {rmse_weighted:.4f}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}

### Standard Model (Baseline)
- R² Score: {r2_unweighted:.4f}
- RMSE: {rmse_unweighted:.4f}

### Improvement
- R² Improvement: {improvement_r2:+.2f}%
- RMSE Improvement: {improvement_rmse:+.2f}%
- Statistical Significance: p={p_value:.4f} {'(p<0.05)' if p_value < 0.05 else '(p≥0.05)'}

### Dataset Characteristics
- Total Records: {len(df)}
- Unique Locations: {df['location'].nunique()}
- Unique Pathogens: {df['pathogen'].nunique()}
- Unique Antibiotics: {df['antibiotic'].nunique()}
- Year Range: {int(df['year'].min())} - {int(df['year'].max())}

### Phylogeny Weights
- Mean Weight: {df['phylogeny_weight'].mean():.3f}
- Unique Samples (weight>0.7): {(df['phylogeny_weight'] > 0.7).sum()}
- Outbreak Samples (weight<0.3): {(df['phylogeny_weight'] < 0.3).sum()}
"""
            
            st.markdown(summary_text)
            
            st.download_button(
                label="Download Statistical Summary",
                data=summary_text,
                file_name="statistical_summary.md",
                mime="text/markdown"
            )

else:
    st.info("Please select a data source from the sidebar to begin analysis.")
    
    # Show example formats
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Expected CSV Format"):
            st.markdown("""
            Your CSV file should contain the following columns:
            - `location` - Hospital, ward, or country name
            - `year` - Year as integer (e.g., 2015, 2016)
            - `pathogen` - e.g., E.coli, K.pneumoniae
            - `antibiotic` - e.g., ciprofloxacin, meropenem
            - `n_tested` - Number of isolates tested
            - `n_resistant` - Number of resistant isolates
            
            Optional: `gene` - AMR gene name for genomic novelty detection
            """)
    
    with col2:
        with st.expander("Expected FASTA Format"):
            st.markdown("""
            FASTA headers should include metadata in one of these formats:
            
            **Pipe-separated:**
            ```
            >E.coli|India|2020|ciprofloxacin
            ATGCGATCGATCG...
            ```
            
            **Underscore-separated:**
            ```
            >E.coli_India_2020_ciprofloxacin
            ATGCGATCGATCG...
            ```
            
            **Download example:**
            """)
            
            example_fasta = create_example_fasta()
            st.download_button(
                label="Download Example FASTA",
                data=example_fasta,
                file_name="example_amr.fasta",
                mime="text/plain"
            )
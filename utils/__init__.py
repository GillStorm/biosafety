import pandas as pd
import streamlit as st
import numpy as np

class Config:
    class ui:
        max_file_size_mb = 200
        
    class model:
        n_estimators = 200
        learning_rate = 0.05
        max_depth = 6
        random_state = 42
        n_jobs = -1
        
    class risk_scoring:
        base_weight = 0.6
        trend_weight = 0.4
        slope_multiplier = 500
        
    REQUIRED_CSV_COLUMNS = ["location", "year", "pathogen", "antibiotic", "n_tested", "n_resistant"]
    
    @staticmethod
    def get_file_size_limit_bytes():
        return Config.ui.max_file_size_mb * 1024 * 1024

def load_uploaded_file(uploaded_file):
    return pd.read_csv(uploaded_file)

def load_example_data():
    # Create simple dummy data
    data = {
        "location": ["Hospital A", "Hospital A", "Hospital B", "Hospital B"] * 5,
        "year": [2020, 2021, 2020, 2021] * 5,
        "pathogen": ["E.coli", "E.coli", "K.pneumoniae", "K.pneumoniae"] * 5,
        "antibiotic": ["Ciprofloxacin", "Ciprofloxacin", "Meropenem", "Meropenem"] * 5,
        "n_tested": np.random.randint(50, 200, 20),
        "n_resistant": np.random.randint(0, 50, 20)
    }
    df = pd.DataFrame(data)
    df['resistance_rate'] = df['n_resistant'] / df['n_tested']
    return df, "Example Dataset"

def load_who_glass_data():
    try:
        df = pd.read_csv("who_glass_amr.csv")
        return df, "WHO GLASS AMR 2022"
    except FileNotFoundError:
        st.error("who_glass_amr.csv not found.")
        return None, None

def validate_amr_data(df, required_columns):
    # Basic validation
    missing = [c for c in required_columns if c not in df.columns]
    errors = []
    warnings = []
    
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return df, errors, warnings
        
    # Validation logic from biosafety.py
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "n_tested", "n_resistant"])
    df = df[df["n_tested"] > 0]
    df["resistance_rate"] = df["n_resistant"] / df["n_tested"]
    df = df[(df["resistance_rate"] >= 0) & (df["resistance_rate"] <= 1)]
    
    return df, errors, warnings

def display_validation_results(errors, warnings):
    if errors:
        for e in errors:
            st.error(e)
    if warnings:
        for w in warnings:
            st.warning(w)

def display_data_summary(df):
    st.write(f"Total Records: {len(df)}")
    st.write(f"Locations: {df['location'].nunique()}")
    st.write(f"Pathogens: {df['pathogen'].nunique()}")
    st.write(f"Antibiotics: {df['antibiotic'].nunique()}")

# Fasta placeholders
def load_fasta_file(file):
    # Placeholder
    st.warning("FASTA loading not fully implemented.")
    return pd.DataFrame()

def display_fasta_summary(df):
    st.write("FASTA Summary placeholder")

def create_example_fasta():
    return "AGCT..."

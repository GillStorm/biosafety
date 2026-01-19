import pandas as pd
import numpy as np

def calculate_risk_scores(df, predicted_rates, base_weight, trend_weight, slope_multiplier):
    df = df.copy()
    # Simplified logic: Risk = predicted_rate * 100
    df['risk_score'] = predicted_rates * 100 
    return df

def calculate_risk_statistics(df_risk):
    if 'risk_score' not in df_risk:
        return {'mean_risk':0, 'high_risk_count':0,'medium_risk_count':0,'low_risk_count':0}
    return {
        'mean_risk': df_risk['risk_score'].mean(),
        'high_risk_count': (df_risk['risk_score'] >= 70).sum(),
        'medium_risk_count': ((df_risk['risk_score'] >= 40) & (df_risk['risk_score'] < 70)).sum(),
        'low_risk_count': (df_risk['risk_score'] < 40).sum()
    }

def identify_high_risk_combinations(df_risk, top_n=10):
    if 'risk_score' in df_risk:
        return df_risk.sort_values('risk_score', ascending=False).head(top_n)
    return pd.DataFrame()

def compare_risk_by_region(df_risk):
    if 'risk_score' in df_risk:
        return df_risk.groupby('location')['risk_score'].mean().reset_index()
    return pd.DataFrame()

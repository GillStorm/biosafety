import plotly.graph_objects as go

def run_substitution_simulation(pipeline, df, pathogen, current_ab, new_ab):
    # Placeholder
    summary = {
        'mean_reduction': 0.15, 
        'locations_improved': 3, 
        'locations_simulated': 5,
        'error': None
    }
    return df.head(5).copy(), summary

def plot_simulation_gauge(summary):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = summary['mean_reduction'] * 100,
        title = {'text': "Risk Reduction (%)"}
    ))
    return fig

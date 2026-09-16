import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data_loader import VolveDataLoader
from src.dca_engine import ArpsDCAEngine
from src.sensitivity import InstitutionalSensitivityEngine, get_cached_loader

# Apply publication-ready style defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


class ValuationVisualizer:
    """
    Tier-1 A&D / Investment Banking Visualizer.
    Generates presentation-grade DCA plots and Sensitivity Heatmaps.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_sensitivity_heatmap(self, sensitivity_df: pd.DataFrame, filename: str = "npv10_sensitivity_heatmap.png"):
        """
        Renders a publication-quality 2D Valuation Sensitivity Heatmap ($ MM NPV10).
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Plot heatmap
        sns.heatmap(
            sensitivity_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            cbar_kws={'label': 'Field NPV10 ($ Millions)'},
            linewidths=1,
            linecolor='white',
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )

        ax.set_title("Volve Field Valuation Sensitivity Matrix: Field NPV10 ($ MM)", fontsize=13, pad=15, weight='bold')
        ax.set_xlabel("Oil Price Scenario ($/bbl & % Base)", fontsize=10, labelpad=10, weight='bold')
        ax.set_ylabel("OPEX Scenario ($/bbl & % Base)", fontsize=10, labelpad=10, weight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[Saved] Heatmap figure saved to: {filepath}")

    def plot_well_dca_fit(
        self, 
        t_hist: np.ndarray, 
        q_hist: np.ndarray, 
        t_forecast: np.ndarray, 
        q_forecast: np.ndarray, 
        well_name: str,
        filename: str = None
    ):
        """
        Plots historical production alongside Arps DCA forecast on Semi-Log scale.
        """
        if filename is None:
            filename = f"dca_fit_{well_name.replace('/', '_')}.png"

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

        # Historical vs Forecast
        ax.semilogy(t_hist, q_hist, 'o', color='#1f77b4', alpha=0.6, label='Historical Oil Prod (BBL/mo)', markersize=4)
        ax.semilogy(t_forecast, q_forecast, '--', color='#d62728', linewidth=2, label='Arps Decline Forecast')

        ax.set_title(f"Arps DCA Production Forecast - Well {well_name}", fontsize=12, pad=12, weight='bold')
        ax.set_xlabel("Time (Months)", fontsize=10, labelpad=8)
        ax.set_ylabel("Monthly Oil Production (BBL/mo) - Log Scale", fontsize=10, labelpad=8)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[Saved] DCA plot saved to: {filepath}")


if __name__ == "__main__":
    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")
    loader = get_cached_loader(dataset_path)
    
    visualizer = ValuationVisualizer()

    # 1. Generate Sensitivity Matrix Heatmap
    print("Generating Field Sensitivity Heatmap...")
    engine = InstitutionalSensitivityEngine(
        loader=loader,
        base_oil_price=75.0,
        base_fixed_opex=15000.0,
        base_var_opex=12.0,
        discount_rate=0.10
    )
    matrix_df = engine.run_field_sensitivity_matrix()
    visualizer.plot_sensitivity_heatmap(matrix_df)

    # 2. Generate Sample DCA Curve Plot for Primary Well
    wells = loader.get_available_wells()
    if len(wells) > 0:
        sample_well = wells[0]
        print(f"Generating Sample DCA Fit Plot for Well: {sample_well}...")
        df_well = loader.clean_well_data(sample_well)
        
        dca_engine = ArpsDCAEngine()
        params = dca_engine.fit_decline_curve(df_well['t_month'].values, df_well['monthly_oil_bbl'].values)
        forecast_df = dca_engine.forecast_production(params['qi'], params['Di'], params['b'], forecast_months=120)

        visualizer.plot_well_dca_fit(
            t_hist=df_well['t_month'].values,
            q_hist=df_well['monthly_oil_bbl'].values,
            t_forecast=forecast_df['t_month'].values,
            q_forecast=forecast_df['forecast_monthly_bbl'].values,
            well_name=sample_well
        )
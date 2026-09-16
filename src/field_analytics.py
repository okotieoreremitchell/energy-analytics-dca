import os
import pandas as pd
import numpy as np
from src.data_loader import VolveDataLoader
from src.dca_engine import ArpsDCAEngine
from src.dcf_engine import UpstreamDCFEngine

class VolveFieldAnalytics:
    """
    Field-wide production analytics engine.
    Automates multi-well Arps curve fitting, cash flow valuation,
    field aggregation, and report generation across all active wellbores.
    """

    def __init__(self, dataset_path: str):
        self.loader = VolveDataLoader(dataset_path)
        self.dca = ArpsDCAEngine()
        self.dcf = UpstreamDCFEngine()

    def run_field_evaluation(self, forecast_months: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes every well in the Volve field dataset and aggregates cash flows and EURs.
        """
        available_wells = self.loader.get_available_wells()
        field_summary = []
        all_forecasts = []

        print(f"Starting field-wide analytics across {len(available_wells)} wells...\n")

        for well in available_wells:
            try:
                # 1. Clean data for individual well
                df_well = self.loader.clean_well_data(well)
                if df_well.empty or len(df_well) < 3:
                    print(f"Skipping well {well}: insufficient production history.")
                    continue

                # 2. Fit DCA parameters
                fit = self.dca.fit_decline_curve(
                    df_well['t_month'].values, 
                    df_well['monthly_oil_bbl'].values
                )

                # 3. Forecast production
                forecast_df = self.dca.forecast_production(
                    fit['qi'], fit['Di'], fit['b'], forecast_months=forecast_months
                )
                forecast_df['well_name'] = well

                # 4. Run DCF Model
                val = self.dcf.run_cash_flow_model(forecast_df)
                metrics = val['metrics']

                # Append well-level summary
                field_summary.append({
                    "Well Name": well,
                    "qi (BBL/mo)": round(fit['qi'], 2),
                    "Di (%/mo)": round(fit['Di'] * 100, 2),
                    "b Exponent": round(fit['b'], 4),
                    "10-Yr EUR (BBL)": int(forecast_df['cum_oil_bbl'].iloc[-1]),
                    "Gross Revenue ($)": round(metrics['gross_revenue_usd'], 2),
                    "Total OPEX ($)": round(metrics['total_opex_usd'], 2),
                    "Net Cash Flow ($)": round(metrics['total_ncf_usd'], 2),
                    "NPV10 ($)": round(metrics['npv10_usd'], 2)
                })

                all_forecasts.append(val['cash_flow_table'])

            except Exception as e:
                print(f"Error processing well {well}: {e}")

        summary_df = pd.DataFrame(field_summary)
        consolidated_forecasts = pd.concat(all_forecasts, ignore_index=True)

        return summary_df, consolidated_forecasts


if __name__ == "__main__":
    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")
    analytics = VolveFieldAnalytics(dataset_path)

    summary_df, field_forecasts = analytics.run_field_evaluation(forecast_months=120)

    print("=" * 70)
    print("                VOLVE FIELD ASSET VALUATION SUMMARY               ")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    total_field_eur = summary_df['10-Yr EUR (BBL)'].sum()
    total_field_npv10 = summary_df['NPV10 ($)'].sum()

    print("\n" + "=" * 70)
    print(f"Total Volve Field 10-Yr Estimated Ultimate Recovery (EUR): {total_field_eur:,.0f} BBLs")
    print(f"Total Volve Field Asset Valuation (NPV10):                ${total_field_npv10:,.2f}")
    print("=" * 70)
import pandas as pd
import numpy as np
import numpy_financial as npf
from src.dca_engine import ArpsDCAEngine
from src.dcf_engine import UpstreamDCFEngine


class FieldAggregatorEngine:
    """Aggregates individual wellbore DCA forecasts and cash flows into a consolidated field model."""

    def __init__(self, data_loader, dcf_params: dict):
        self.loader = data_loader
        self.dcf_params = dcf_params
        self.dca_engine = ArpsDCAEngine()

    @staticmethod
    def _get_col_values(df: pd.DataFrame, candidates: list, default_val: float = 0.0) -> np.ndarray:
        """Safely extracts numpy values for the first matching column candidate in a DataFrame."""
        for col in candidates:
            if col in df.columns:
                return df[col].values
        return np.full(len(df), default_val)

    def aggregate_field(
        self,
        well_list: list,
        forecast_months: int = 120,
        d_lim_annual: float = 0.06,
        total_field_capex: float = 0.0,
    ) -> dict:
        if not well_list:
            raise ValueError(
                "At least one wellbore must be selected for aggregation."
            )

        # Consolidated tracking streams
        field_oil_bbl = np.zeros(forecast_months)
        field_gross_rev = np.zeros(forecast_months)
        field_opex = np.zeros(forecast_months)
        field_tax = np.zeros(forecast_months)
        field_ncf = np.zeros(forecast_months)

        well_summaries = []

        dcf = UpstreamDCFEngine(**self.dcf_params)

        for well in well_list:
            df_clean = self.loader.clean_well_data(well)
            if len(df_clean) < 3:
                continue

            # 1. Fit Arps Decline & Forecast
            fit_results = self.dca_engine.fit_decline_curve(
                df_clean["t_month"].values, df_clean["monthly_oil_bbl"].values
            )
            forecast_df = self.dca_engine.forecast_production(
                qi=fit_results["qi"],
                Di=fit_results["Di"],
                b=fit_results["b"],
                forecast_months=forecast_months,
                d_lim_annual=d_lim_annual,
            )

            # Standardize column name
            oil_col = "oil_prod_bbl" if "oil_prod_bbl" in forecast_df.columns else forecast_df.columns[1]
            forecast_df["oil_prod_bbl"] = forecast_df[oil_col]

            # 2. Run DCF per Well (assigning 0 initial CAPEX at well-level)
            well_val = dcf.run_cash_flow_model(forecast_df, initial_capex=0.0)
            well_cf = well_val["cash_flow_table"]

            n_len = min(forecast_months, len(well_cf))

            # 3. Dynamic Safe Extraction (Handles any DCF column naming variant)
            oil_vals = self._get_col_values(well_cf, ["oil_prod_bbl", "monthly_oil_bbl", "oil_bbl"])
            rev_vals = self._get_col_values(well_cf, ["total_gross_revenue", "gross_revenue", "gross_rev"])
            opex_vals = self._get_col_values(well_cf, ["total_opex", "var_opex", "opex"])
            tax_vals = self._get_col_values(well_cf, ["tax_paid", "tax_liability", "tax"])
            ncf_vals = self._get_col_values(well_cf, ["net_cash_flow", "ncf", "discounted_ncf"])

            # Accumulate field totals
            field_oil_bbl[:n_len] += oil_vals[:n_len]
            field_gross_rev[:n_len] += rev_vals[:n_len]
            field_opex[:n_len] += opex_vals[:n_len]
            field_tax[:n_len] += tax_vals[:n_len]
            field_ncf[:n_len] += ncf_vals[:n_len]

            well_summaries.append(
                {
                    "well_name": well,
                    "qi": fit_results["qi"],
                    "b": fit_results["b"],
                    "eur_bbl": forecast_df["oil_prod_bbl"].sum(),
                    "well_npv10": well_val["metrics"].get("npv10_usd", 0.0),
                }
            )

        # Apply Field-Level CAPEX in Month 0 / Month 1 NCF
        field_ncf[0] -= total_field_capex

        # 4. Consolidated Cash Flow Table
        months = np.arange(1, forecast_months + 1)
        consolidated_df = pd.DataFrame(
            {
                "t_month": months,
                "oil_prod_bbl": field_oil_bbl,
                "gross_revenue": field_gross_rev,
                "total_opex": field_opex,
                "tax_paid": field_tax,
                "net_cash_flow": field_ncf,
            }
        )

        # 5. Calculate Consolidated Field Key Metrics
        discount_rate = self.dcf_params.get("discount_rate", 0.10)
        discount_factors = (1 + discount_rate / 12) ** (
            -consolidated_df["t_month"]
        )
        field_npv = np.sum(consolidated_df["net_cash_flow"] * discount_factors)

        # Unlevered IRR computation via numpy_financial
        try:
            cf_stream = np.insert(
                consolidated_df["net_cash_flow"].values[1:], 0, -total_field_capex + consolidated_df["net_cash_flow"].values[0]
            )
            irr_monthly = npf.irr(cf_stream)
            field_irr = (
                (1 + irr_monthly) ** 12 - 1 if not np.isnan(irr_monthly) else 0.0
            )
        except Exception:
            field_irr = 0.0

        return {
            "field_df": consolidated_df,
            "well_summaries": pd.DataFrame(well_summaries),
            "field_metrics": {
                "field_npv": field_npv,
                "field_irr": field_irr,
                "total_eur_bbl": field_oil_bbl.sum(),
                "total_ncf_usd": field_ncf.sum(),
                "active_wells": len(well_summaries),
            },
        }
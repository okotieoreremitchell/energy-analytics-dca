import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.data_loader import VolveDataLoader
from src.dca_engine import ArpsDCAEngine
from src.dcf_engine import UpstreamDCFEngine


class InstitutionalSensitivityEngine:
    """
    Tier-1 Investment Banking & Energy PE Sensitivity Engine.
    
    Combines cached physical DCA curves with well-level UpstreamDCFEngine
    evaluations to guarantee exact financial accuracy and sub-second performance.
    """

    def __init__(
        self,
        loader: VolveDataLoader,
        base_oil_price: float = 75.0,
        base_fixed_opex: float = 15000.0,
        base_var_opex: float = 12.0,
        discount_rate: float = 0.10,
        net_revenue_interest: float = 0.85,
        forecast_months: int = 120
    ):
        self.loader = loader
        self.base_oil_price = base_oil_price
        self.base_fixed_opex = base_fixed_opex
        self.base_var_opex = base_var_opex
        self.discount_rate = discount_rate
        self.nri = net_revenue_interest
        self.royalty_rate = 1.0 - net_revenue_interest
        self.forecast_months = forecast_months
        self.dca_engine = ArpsDCAEngine()
        
        # Build and cache physical decline curves once on init
        self.asset_cache = self._build_physical_curve_cache()

    def calculate_economic_limit(self, oil_price: float, var_opex: float, fixed_opex: float) -> float:
        """
        Calculates Economic Limit Cutoff rate (q_ec in BBL/month).
        q_ec = Fixed OPEX / (Oil_Price * NRI - Var_OPEX)
        """
        net_realized = (oil_price * self.nri) - var_opex
        if net_realized <= 0:
            return float('inf')
        return fixed_opex / net_realized

    def _build_physical_curve_cache(self) -> List[Dict[str, Any]]:
        wells = self.loader.get_available_wells()
        cache = []

        for well in wells:
            try:
                df = self.loader.clean_well_data(well)
                if len(df) < 6:
                    continue

                # Fit physical curve once per well
                fit_params = self.dca_engine.fit_decline_curve(
                    df['t_month'].values, 
                    df['monthly_oil_bbl'].values
                )

                forecast_df = self.dca_engine.forecast_production(
                    fit_params['qi'],
                    fit_params['Di'],
                    fit_params['b'],
                    forecast_months=self.forecast_months
                )

                cache.append({
                    'well_name': well,
                    'forecast_df': forecast_df
                })
            except Exception:
                continue

        return cache

    def run_field_sensitivity_matrix(
        self,
        price_multipliers: List[float] = [0.70, 0.85, 1.00, 1.15, 1.30],
        opex_multipliers: List[float] = [0.70, 0.85, 1.00, 1.15, 1.30]
    ) -> pd.DataFrame:
        """
        Executes exact 2D Valuation Sensitivity Matrix ($ MM NPV10).
        Evaluates each cell through UpstreamDCFEngine per well.
        """
        matrix_results = np.zeros((len(opex_multipliers), len(price_multipliers)))

        for i, opex_mult in enumerate(opex_multipliers):
            curr_fixed_opex = self.base_fixed_opex * opex_mult
            curr_var_opex = self.base_var_opex * opex_mult

            for j, price_mult in enumerate(price_multipliers):
                curr_price = self.base_oil_price * price_mult
                field_npv = 0.0

                # Clean DCF model for current macro scenario
                dcf_engine = UpstreamDCFEngine(
                    oil_price_usd=curr_price,
                    royalty_rate=self.royalty_rate,
                    variable_opex_bbl=curr_var_opex,
                    fixed_opex_month=curr_fixed_opex,
                    discount_rate=self.discount_rate,
                    severance_tax_rate=0.0,
                    corporate_tax_rate=0.0
                )

                # Evaluate cached physical curves against scenario economic limit
                q_ec = self.calculate_economic_limit(curr_price, curr_var_opex, curr_fixed_opex)

                for asset in self.asset_cache:
                    df_run = asset['forecast_df'].copy()
                    df_run['forecast_monthly_bbl'] = np.where(
                        df_run['forecast_monthly_bbl'] >= q_ec,
                        df_run['forecast_monthly_bbl'],
                        0.0
                    )
                    
                    res = dcf_engine.run_cash_flow_model(df_run)
                    field_npv += res['metrics']['npv10_usd']

                matrix_results[i, j] = field_npv / 1e6

        row_labels = [f"OPEX {int(m*100)}% (${(self.base_var_opex*m):.2f}/bbl)" for m in opex_multipliers]
        col_labels = [f"Oil ${ (self.base_oil_price*m):.1f}/bbl ({int(m*100)}%)" for m in price_multipliers]

        return pd.DataFrame(matrix_results, index=row_labels, columns=col_labels)


def get_cached_loader(excel_path: str, cache_dir: str = "data/processed") -> VolveDataLoader:
    """
    Loads VolveDataLoader using Feather binary caching to eliminate Excel I/O overhead.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "volve_production_cached.feather")

    if os.path.exists(cache_path):
        # Load directly from cached Feather file
        loader = VolveDataLoader(excel_path)
        loader.df = pd.read_feather(cache_path)
        return loader
    
    # First-time load from Excel and save Feather binary
    loader = VolveDataLoader(excel_path)
    if hasattr(loader, 'df') and loader.df is not None:
        loader.df.to_feather(cache_path)
    return loader


if __name__ == "__main__":
    t0 = time.time()
    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")
    
    # Fast loader initialization
    loader = get_cached_loader(dataset_path)
    
    print("======================================================================")
    print("      INITIALIZING PE/A&D ENGINE & CACHING ENGINEERING PROFILES       ")
    print("======================================================================")
    
    engine = InstitutionalSensitivityEngine(
        loader=loader,
        base_oil_price=75.0,
        base_fixed_opex=15000.0,
        base_var_opex=12.0,
        discount_rate=0.10
    )

    q_ec_base = engine.calculate_economic_limit(75.0, 12.0, 15000.0)
    print(f"Base Case Realized Net Oil Price:  ${(75.0 * 0.85 - 12.0):.2f} / BBL")
    print(f"Economic Limit Cutoff Rate (q_ec): {q_ec_base:.2f} BBL / Month")
    print("======================================================================\n")

    print("Executing Institutional Two-Way Field NPV10 Matrix ($ MM)...")
    matrix_df = engine.run_field_sensitivity_matrix()
    
    print("\n======================================================================")
    print("       VOLVE FIELD TWO-WAY NPV10 SENSITIVITY MATRIX ($ MILLIONS)       ")
    print("======================================================================")
    print(matrix_df.round(2).to_string())
    print("======================================================================")
    print(f"\nExecution Completed in {time.time() - t0:.2f} seconds.")
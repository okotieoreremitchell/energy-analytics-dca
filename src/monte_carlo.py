import numpy as np
import pandas as pd
from src.dca_engine import ArpsDCAEngine
from src.dcf_engine import UpstreamDCFEngine

class MonteCarloEngine:
    """Executes probabilistic risk analysis over DCA forecasts and DCF valuations."""

    def __init__(self, base_fit: dict, base_params: dict):
        self.base_fit = base_fit
        self.base_params = base_params
        self.dca_engine = ArpsDCAEngine()

    def run_simulation(
        self,
        iterations: int = 1000,
        oil_price_std: float = 10.0,
        qi_std_pct: float = 0.10,
        opex_std_pct: float = 0.15,
        forecast_months: int = 120,
        d_lim_annual: float = 0.06,
        initial_capex: float = 5000000.0
    ) -> pd.DataFrame:
        """Runs N iterations sampling from normal distributions around base parameters."""
        
        np.random.seed(42)  # For reproducible institutional auditability
        
        # Sample parameters
        oil_prices = np.random.normal(self.base_params['oil_price_usd'], oil_price_std, iterations)
        oil_prices = np.maximum(oil_prices, 20.0)  # Floor price at $20/BBL
        
        qis = np.random.normal(self.base_fit['qi'], self.base_fit['qi'] * qi_std_pct, iterations)
        b_factors = np.random.uniform(0.1, 1.2, iterations)  # Uniform bound for b-factor
        
        var_opexs = np.random.normal(
            self.base_params['variable_opex_bbl'], 
            self.base_params['variable_opex_bbl'] * opex_std_pct, 
            iterations
        )

        npv_results = []
        irr_results = []
        eur_results = []

        for i in range(iterations):
            # 1. Forecast production with sampled subsurface params
            fc_df = self.dca_engine.forecast_production(
                qi=qis[i],
                Di=self.base_fit['Di'],
                b=b_factors[i],
                forecast_months=forecast_months,
                d_lim_annual=d_lim_annual
            )
            
            oil_col = 'oil_prod_bbl' if 'oil_prod_bbl' in fc_df.columns else fc_df.columns[1]
            fc_df['oil_prod_bbl'] = fc_df[oil_col]

            # 2. Valuation with sampled financial params
            dcf = UpstreamDCFEngine(
                oil_price_usd=oil_prices[i],
                gas_price_mcf=self.base_params['gas_price_mcf'],
                variable_opex_bbl=var_opexs[i],
                fixed_opex_month=self.base_params['fixed_opex_month'],
                discount_rate=self.base_params['discount_rate'],
                apply_ncs_tax=self.base_params['apply_ncs_tax']
            )

            res = dcf.run_cash_flow_model(fc_df, initial_capex=initial_capex)
            
            npv_results.append(res['metrics']['npv10_usd'] / 1e6)  # Store in $MM
            
            irr_val = res['metrics']['irr_annual']
            irr_results.append(irr_val if isinstance(irr_val, float) and not np.isnan(irr_val) else 0.0)
            eur_results.append(fc_df['oil_prod_bbl'].sum() / 1e6)  # Store in MMbbl

        sim_df = pd.DataFrame({
            'npv10_mm': npv_results,
            'irr_annual': irr_results,
            'eur_mmbbl': eur_results,
            'oil_price': oil_prices,
            'sampled_qi': qis,
            'sampled_b': b_factors
        })

        return sim_df
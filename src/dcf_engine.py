import numpy as np
import numpy_financial as npf
import pandas as pd


class UpstreamDCFEngine:
    """
    Valuation Engine for Unlevered Cash Flow Analysis of Upstream Oil & Gas Assets.
    Implements Norwegian Continental Shelf (NCS) fiscal regime, multi-stream production economics,
    and economic limit truncation.
    """

    def __init__(
        self,
        oil_price_usd: float = 75.0,
        gas_price_mcf: float = 3.50,
        gor_mcf_bbl: float = 0.80,          # Gas-Oil Ratio (MCF associated gas per BBL oil)
        wct_bbl_bbl: float = 1.50,          # Water Cut Ratio (BBL water per BBL oil)
        water_cost_bbl: float = 1.25,       # Water handling/disposal cost ($/BBL water)
        royalty_rate: float = 0.0,          # NCS has 0% royalty rate
        variable_opex_bbl: float = 12.0,    # $/BBL lifting cost
        fixed_opex_month: float = 15000.0,  # $/month fixed well OPEX
        abex_per_well: float = 2_500_000.0, # Plug & Abandonment liability per well ($)
        discount_rate: float = 0.10,        # 10% standard discount rate (NPV10)
        apply_ncs_tax: bool = True,         # Norwegian Continental Shelf tax regime
        corp_tax_rate: float = 0.22,        # NCS Corporate Tax Rate (22%)
        special_tax_rate: float = 0.56      # NCS Special Tax Rate (56%)
    ):
        self.oil_price = oil_price_usd
        self.gas_price = gas_price_mcf
        self.gor = gor_mcf_bbl
        self.wct = wct_bbl_bbl
        self.water_cost = water_cost_bbl
        self.royalty_rate = royalty_rate
        self.variable_opex = variable_opex_bbl
        self.fixed_opex = fixed_opex_month
        self.abex_per_well = abex_per_well
        self.discount_rate = discount_rate
        self.apply_ncs_tax = apply_ncs_tax
        self.corp_tax = corp_tax_rate
        self.special_tax = special_tax_rate

    def calculate_economic_cutoff_rate(self) -> float:
        """
        Calculates minimum monthly oil production (BBL/month) required to break even on monthly OPEX.
        Net Unit Margin = (Oil Price + GOR * Gas Price) - Var OPEX - (WCT * Water Cost)
        """
        unit_rev = self.oil_price + (self.gor * self.gas_price)
        unit_opex = self.variable_opex + (self.wct * self.water_cost)
        net_unit_margin = (unit_rev * (1.0 - self.royalty_rate)) - unit_opex

        if net_unit_margin <= 0:
            return float('inf')
        return self.fixed_opex / net_unit_margin

    def run_cash_flow_model(self, forecast_df: pd.DataFrame, initial_capex: float = 0.0) -> dict:
        """
        Executes monthly financial accounting waterfall over the forecasted production horizon.
        Truncates production at economic limit and applies P&A liability (ABEX).
        """
        df = forecast_df.copy()
        q_ec = self.calculate_economic_cutoff_rate()

        # Robust column detection for production volume
        if 'forecast_monthly_bbl' in df.columns:
            prod_col = 'forecast_monthly_bbl'
        elif 'oil_prod_bbl' in df.columns:
            prod_col = 'oil_prod_bbl'
        else:
            prod_col = df.columns[1]

        # Determine active economic months
        df['is_active'] = df[prod_col] >= q_ec

        # Truncate production after first non-economic month
        if not df['is_active'].all():
            first_cutoff_idx = (~df['is_active']).idxmax()
            df.loc[first_cutoff_idx:, prod_col] = 0.0
            economic_life_months = int(first_cutoff_idx)
        else:
            economic_life_months = len(df)

        # 1. Multi-Stream Production Volumes
        df['oil_prod_bbl'] = df[prod_col]
        df['forecast_monthly_bbl'] = df[prod_col]
        df['gas_prod_mcf'] = df['oil_prod_bbl'] * self.gor
        df['water_prod_bbl'] = df['oil_prod_bbl'] * self.wct

        # 2. Multi-Stream Revenues
        df['gross_oil_rev'] = df['oil_prod_bbl'] * self.oil_price
        df['gross_gas_rev'] = df['gas_prod_mcf'] * self.gas_price
        df['total_gross_revenue'] = df['gross_oil_rev'] + df['gross_gas_rev']
        df['net_revenue'] = df['total_gross_revenue'] * (1.0 - self.royalty_rate)

        # 3. Operating Expenses (LOE & Water Processing)
        df['var_opex'] = df['oil_prod_bbl'] * self.variable_opex
        df['water_opex'] = df['water_prod_bbl'] * self.water_cost
        df['fixed_opex'] = np.where(df['oil_prod_bbl'] > 0, self.fixed_opex, 0.0)
        df['total_opex'] = df['var_opex'] + df['water_opex'] + df['fixed_opex']

        # 4. EBITDA
        df['ebitda'] = df['net_revenue'] - df['total_opex']

        # 5. ABEX / Plug & Abandonment Liability at End of Economic Life
        df['abex'] = 0.0
        if 0 < economic_life_months <= len(df):
            df.loc[economic_life_months - 1, 'abex'] = self.abex_per_well

        # 6. Fiscal Tax Modeling (NCS 78% Combined Marginal Rate)
        if self.apply_ncs_tax:
            df['taxable_income'] = np.maximum(0.0, df['ebitda'] - df['abex'])
            df['tax_liability'] = df['taxable_income'] * (self.corp_tax + self.special_tax)
        else:
            df['tax_liability'] = 0.0

        # 7. Unlevered Net Cash Flow
        df['net_cash_flow'] = df['ebitda'] - df['abex'] - df['tax_liability']

        # Apply initial CAPEX at Month 1 if present
        if initial_capex > 0:
            df.loc[df['t_month'] == 1, 'net_cash_flow'] -= initial_capex

        # 8. Monthly Discounting & NPV
        df['discount_factor'] = np.power(1.0 + self.discount_rate, df['t_month'] / 12.0)
        df['discounted_ncf'] = df['net_cash_flow'] / df['discount_factor']

        df['cum_ncf'] = np.cumsum(df['net_cash_flow'])
        df['cum_discounted_ncf'] = np.cumsum(df['discounted_ncf'])

        # Summary Metrics
        npv10 = float(df['discounted_ncf'].sum())
        total_undiscounted_ncf = float(df['net_cash_flow'].sum())
        total_gross_revenue = float(df['total_gross_revenue'].sum())
        total_opex = float(df['total_opex'].sum())
        total_oil = float(df['oil_prod_bbl'].sum())

        # Returns Metrics (IRR & MOIC)
        cash_flows = df['net_cash_flow'].values
        if initial_capex > 0:
            cf_with_initial = np.insert(cash_flows, 0, -initial_capex)
            try:
                monthly_irr = npf.irr(cf_with_initial)
                annual_irr = ((1.0 + monthly_irr) ** 12.0) - 1.0 if monthly_irr is not None and not np.isnan(monthly_irr) else 0.0
            except Exception:
                annual_irr = 0.0
            
            total_inflows = df[df['net_cash_flow'] > 0]['net_cash_flow'].sum()
            moic = total_inflows / initial_capex if initial_capex > 0 else 0.0
        else:
            annual_irr = 0.0
            moic = 0.0

        return {
            "cash_flow_table": df,
            "metrics": {
                "npv10_usd": npv10,
                "total_ncf_usd": total_undiscounted_ncf,
                "gross_revenue_usd": total_gross_revenue,
                "total_opex_usd": total_opex,
                "total_oil_bbl": total_oil,
                "economic_limit_month": economic_life_months,
                "economic_cutoff_bbl_month": q_ec,
                "irr_annual": annual_irr,
                "moic": moic
            }
        }


# VERIFICATION SCRIPT
if __name__ == "__main__":
    import os
    from src.data_loader import VolveDataLoader
    from src.dca_engine import ArpsDCAEngine

    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")
    loader = VolveDataLoader(dataset_path)
    wells = loader.get_available_wells()
    test_well = wells[0]
    df_clean = loader.clean_well_data(test_well)

    # DCA Forecast
    dca = ArpsDCAEngine()
    fit_results = dca.fit_decline_curve(df_clean['t_month'].values, df_clean['monthly_oil_bbl'].values)
    forecast_df = dca.forecast_production(
        fit_results['qi'], 
        fit_results['Di'], 
        fit_results['b'], 
        forecast_months=120,
        d_lim_annual=0.06
    )

    # Institutional DCF
    dcf = UpstreamDCFEngine(
        oil_price_usd=75.0,
        gas_price_mcf=3.50,
        variable_opex_bbl=12.0,
        fixed_opex_month=15000.0,
        abex_per_well=2_500_000.0,
        apply_ncs_tax=True
    )
    valuation = dcf.run_cash_flow_model(forecast_df, initial_capex=5_000_000.0)

    metrics = valuation['metrics']
    cf_table = valuation['cash_flow_table']

    print(f"=== Institutional Financial Valuation: Well {test_well} ===")
    print(f"Economic Cutoff Threshold:  {metrics['economic_cutoff_bbl_month']:.2f} BBL/month")
    print(f"Economic Life Limit:        Month {metrics['economic_limit_month']}")
    print(f"10-Year Gross Revenue:      ${metrics['gross_revenue_usd']:,.2f}")
    print(f"10-Year Total OPEX:         ${metrics['total_opex_usd']:,.2f}")
    print(f"Post-Tax Asset NPV10:       ${metrics['npv10_usd']:,.2f}")
    print(f"Unlevered Project IRR:      {metrics['irr_annual']*100:.2f}%")
    print(f"MOIC:                       {metrics['moic']:.2f}x")

    print("\n--- Cash Flow Snapshot (Around Cutoff / P&A) ---")
    cols = ['t_month', 'oil_prod_bbl', 'total_gross_revenue', 'total_opex', 'abex', 'tax_liability', 'net_cash_flow']
    cutoff_m = metrics['economic_limit_month']
    print(cf_table[cols].iloc[max(0, cutoff_m - 3):min(len(cf_table), cutoff_m + 2)])
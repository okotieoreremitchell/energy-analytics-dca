import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class ArpsDCAEngine:
    """
    Engine for Arps Decline Curve Analysis (DCA) using hyperbolic and exponential models.
    Provides methods for fitting production data, forecasting future production, and calculating cumulative production.
    """

    @staticmethod
    def hyperbolic_decline(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
        """
        Hyperbolic Arps Decline Equation:
        q(t) = qi / (1 + b * Di * t) ** (1 / b)
        Handles edge cases where b -> 0 (exponential limit).
        """
        if abs(b) < 1e-6:
            return qi * np.exp(-Di * t)
        return qi / np.power(1.0 + b * Di * t, 1.0 / b)

    @staticmethod
    def exponential_decline(t: np.ndarray, q_trans: float, D_lim_monthly: float, t_trans: float) -> np.ndarray:
        """
        Exponential Decline Equation used after transitioning at D_lim:
        q(t) = q_trans * exp(-D_lim * (t - t_trans))
        """
        return q_trans * np.exp(-D_lim_monthly * (t - t_trans))

    def fit_decline_curve(self, t_data: np.ndarray, q_data: np.ndarray, max_b: float = 1.0) -> dict:
        """
        Uses SciPy non-linear least squares optimization to solve for optimal (qi, Di, b).
        Calculates goodness-of-fit metrics (R2, RMSE) for institutional audit requirements.
        """
        if len(t_data) < 3:
            raise ValueError("Insufficient data points for non-linear regression (minimum 3 required).")

        qi_guess = float(np.max(q_data))
        Di_guess = 0.05  # 5% monthly decline initial guess
        b_guess = 0.5    # Standard initial hyperbolic guess

        p0 = [qi_guess, Di_guess, b_guess]

        bounds = (
            [0.0, 1e-4, 0.0],
            [qi_guess * 3.0, 0.5, max_b]
        )

        try:
            popt, pcov = curve_fit(
                self.hyperbolic_decline,
                t_data,
                q_data,
                p0=p0,
                bounds=bounds,
                method='trf'
            )
            
            qi_fit, Di_fit, b_fit = popt

            # Goodness-of-fit calculation
            q_pred = self.hyperbolic_decline(t_data, qi_fit, Di_fit, b_fit)
            ss_res = np.sum((q_data - q_pred) ** 2)
            ss_tot = np.sum((q_data - np.mean(q_data)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean((q_data - q_pred) ** 2))

            return {
                "qi": qi_fit,
                "Di": Di_fit,
                "b": b_fit,
                "r2": r2,
                "rmse": rmse,
                "covariance": pcov
            }

        except Exception as e:
            raise RuntimeError(f"Non-linear decline curve fitting failed: {e}")

    def forecast_production(
        self, 
        qi: float, 
        Di: float, 
        b: float, 
        forecast_months: int = 120,
        d_lim_annual: float = 0.06
    ) -> pd.DataFrame:
        """
        Generates forward production forecast.
        Applies terminal annual decline rate floor (D_lim, default 6% p.a.).
        Switches from hyperbolic to exponential decline once nominal decline D(t) <= D_lim.
        """
        d_lim_monthly = -np.log(1.0 - d_lim_annual) / 12.0
        t_future = np.arange(1, forecast_months + 1)
        q_forecast = np.zeros(forecast_months)

        # Instantaneous nominal monthly decline rate D(t) = Di / (1 + b * Di * t)
        D_t = Di / (1.0 + b * Di * t_future)

        # Identify transition point
        trans_mask = D_t <= d_lim_monthly

        if not np.any(trans_mask):
            # Stays hyperbolic for the entire forecast horizon
            q_forecast = self.hyperbolic_decline(t_future, qi, Di, b)
            is_exponential = np.zeros(forecast_months, dtype=bool)
        else:
            t_trans_idx = np.argmax(trans_mask)
            t_trans = t_future[t_trans_idx]

            # Hyperbolic segment
            q_hyp = self.hyperbolic_decline(t_future[:t_trans_idx], qi, Di, b)
            q_trans = self.hyperbolic_decline(t_trans, qi, Di, b)

            # Exponential segment
            q_exp = self.exponential_decline(t_future[t_trans_idx:], q_trans, d_lim_monthly, t_trans)

            q_forecast = np.concatenate([q_hyp, q_exp])
            is_exponential = t_future >= t_trans

        cum_oil_bbl = np.cumsum(q_forecast)

        return pd.DataFrame({
            "t_month": t_future,
            "forecast_monthly_bbl": q_forecast,
            "cum_oil_bbl": cum_oil_bbl,
            "nominal_decline_monthly": D_t,
            "is_exponential": is_exponential
        })


# VERIFICATION SCRIPT
if __name__ == "__main__":
    import os
    from src.data_loader import VolveDataLoader

    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")
    loader = VolveDataLoader(dataset_path)

    wells = loader.get_available_wells()
    test_well = wells[0]
    df_clean = loader.clean_well_data(test_well)

    t_data = df_clean['t_month'].values
    q_data = df_clean['monthly_oil_bbl'].values

    dca = ArpsDCAEngine()
    fit_results = dca.fit_decline_curve(t_data, q_data)

    print(f"=== Arps Decline Curve Fit: Well {test_well} ===")
    print(f"Initial Rate (qi):     {fit_results['qi']:.2f} BBL/month")
    print(f"Initial Decline (Di): {fit_results['Di']*100:.2f}% per month")
    print(f"Arps b-exponent (b):  {fit_results['b']:.4f}")
    print(f"Fit Quality (R2):     {fit_results['r2']:.4f}")

    forecast = dca.forecast_production(
        fit_results['qi'],
        fit_results['Di'],
        fit_results['b'],
        forecast_months=120,
        d_lim_annual=0.06
    )

    print("\n=== Forecast with D_lim (6% p.a. Floor) ===")
    print(forecast.head())
    print(f"\n10-Yr Cumulative EUR: {forecast['cum_oil_bbl'].iloc[-1]:,.0f} BBLs")
    print(f"Switched to Exponential Decline: {forecast['is_exponential'].any()}")
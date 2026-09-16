import os
import pandas as pd
import numpy as np


class VolveDataLoader:

    def __init__(self, file_path: str, cache_dir: str = "data/processed"):
        self.file_path = file_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            self.cache_dir, f"{os.path.basename(file_path).split('.')[0]}.feather"
        )

    def load_raw_data(self) -> pd.DataFrame:
        """
        Reads raw dataset from Excel and caches it in Feather format for faster subsequent loads.
        """
        if os.path.exists(self.cache_path):
            return pd.read_feather(self.cache_path)

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")

        df = pd.read_excel(self.file_path)
        df.columns = [str(col).strip() for col in df.columns]

        # Convert datetimes for Feather compatibility and write cache
        date_col = 'DATEPRD' if 'DATEPRD' in df.columns else 'DATE'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        df.to_feather(self.cache_path)
        return df

    def get_available_wells(self) -> list:
        """Returns unique list of producing wellbores in dataset."""
        df = self.load_raw_data()
        well_col = 'NPD_WELL_BORE_NAME' if 'NPD_WELL_BORE_NAME' in df.columns else 'WELLBORE'
        return df[well_col].dropna().unique().tolist()

    def clean_well_data(self, well_name: str, smoothing_window: int = 3) -> pd.DataFrame:
        """
        Extracts, normalizes, and cleans production data for a specific well, applying rolling median smoothing and monthly aggregation.
        """
        df = self.load_raw_data()

        well_col = 'NPD_WELL_BORE_NAME' if 'NPD_WELL_BORE_NAME' in df.columns else 'WELLBORE'
        date_col = 'DATEPRD' if 'DATEPRD' in df.columns else 'DATE'
        oil_col = 'BORE_OIL_VOL' if 'BORE_OIL_VOL' in df.columns else 'OIL_BBL'
        gas_col = 'BORE_GAS_VOL' if 'BORE_GAS_VOL' in df.columns else 'GAS_MCF'
        water_col = 'BORE_WAT_VOL' if 'BORE_WAT_VOL' in df.columns else 'WATER_BBL'

        well_df = df[df[well_col] == well_name].copy()

        if well_df.empty:
            available_wells = df[well_col].unique().tolist()
            raise ValueError(f"Well '{well_name}' not found. Available wells: {available_wells}")

        well_df[date_col] = pd.to_datetime(well_df[date_col])
        well_df = well_df.sort_values(by=date_col).reset_index(drop=True)

        # Parse numeric volumes safely across streams
        well_df['raw_oil_bbl'] = pd.to_numeric(well_df[oil_col], errors='coerce').fillna(0)
        well_df['raw_gas_mcf'] = pd.to_numeric(well_df.get(gas_col, 0), errors='coerce').fillna(0)
        well_df['raw_water_bbl'] = pd.to_numeric(well_df.get(water_col, 0), errors='coerce').fillna(0)

        # Isolate producing days (where oil > 0) to prevent log-scale distortion during DCA
        producing_df = well_df[well_df['raw_oil_bbl'] > 0].copy()

        # Apply rolling median smoothing to remove operational measurement spikes
        for stream in ['raw_oil_bbl', 'raw_gas_mcf', 'raw_water_bbl']:
            clean_col = stream.replace('raw_', 'cleaned_')
            producing_df[clean_col] = (
                producing_df[stream]
                .rolling(window=smoothing_window, center=True, min_periods=1)
                .median()
            )

        # Monthly Financial Bucketing
        producing_df['year_month'] = producing_df[date_col].dt.to_period('M')

        monthly_df = (
            producing_df.groupby('year_month')
            .agg(
                date=(date_col, 'min'),
                monthly_oil_bbl=('cleaned_oil_bbl', 'sum'),
                monthly_gas_mcf=('cleaned_gas_mcf', 'sum'),
                monthly_water_bbl=('cleaned_water_bbl', 'sum'),
                producing_days=('raw_oil_bbl', 'count')
            )
            .reset_index()
        )

        # Elapsed monthly timeline (t = 1, 2, 3...)
        monthly_df['t_month'] = np.arange(1, len(monthly_df) + 1)

        # Compute dynamic Gor (MCF/BBL) and Water Cut (%) ratios
        monthly_df['gor_mcf_bbl'] = np.where(
            monthly_df['monthly_oil_bbl'] > 0,
            monthly_df['monthly_gas_mcf'] / monthly_df['monthly_oil_bbl'],
            0.0
        )
        monthly_df['water_cut'] = np.where(
            (monthly_df['monthly_oil_bbl'] + monthly_df['monthly_water_bbl']) > 0,
            monthly_df['monthly_water_bbl'] / (monthly_df['monthly_oil_bbl'] + monthly_df['monthly_water_bbl']),
            0.0
        )

        return monthly_df[
            [
                't_month',
                'date',
                'monthly_oil_bbl',
                'monthly_gas_mcf',
                'monthly_water_bbl',
                'gor_mcf_bbl',
                'water_cut',
                'producing_days'
            ]
        ]


# VERIFICATION RUNNER
if __name__ == "__main__":
    dataset_path = os.path.join("data", "raw", "volve_production.xlsx")

    try:
        loader = VolveDataLoader(dataset_path)
        wells = loader.get_available_wells()
        print(f"[Success] Fast Cache Loader Ready. Available Wells: {len(wells)}")

        test_well = wells[0]
        cleaned = loader.clean_well_data(test_well)
        print(f"\nMulti-Stream Cleaned Data Output for Well ({test_well}):")
        print(cleaned.head())

    except Exception as e:
        print(f"[Error] Data loader verification failed: {e}")
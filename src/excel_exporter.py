import os
import sys
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class InstitutionalExcelExporter:
    """
    Tier-1 Investment Banking & Energy Private Equity Excel Exporter.
    Injects dynamic Excel formulas (=SUM, =MAX, arithmetic) and applies 
    Wall Street presentation standards (Navy/Slate theme, double-underlines).
    """

    # Wall Street / Institutional Palette
    COLOR_NAVY = "1B365D"
    COLOR_ACCENT = "E8EEF5"
    COLOR_ZEBRA = "F9FAFB"
    COLOR_MUTED = "595959"

    FILL_NAVY = PatternFill(start_color=COLOR_NAVY, end_color=COLOR_NAVY, fill_type="solid")
    FILL_ACCENT = PatternFill(start_color=COLOR_ACCENT, end_color=COLOR_ACCENT, fill_type="solid")
    FILL_ZEBRA = PatternFill(start_color=COLOR_ZEBRA, end_color=COLOR_ZEBRA, fill_type="solid")

    # Typography
    FONT_TITLE = Font(name="Segoe UI", size=16, bold=True, color=COLOR_NAVY)
    FONT_SUBTITLE = Font(name="Segoe UI", size=10, italic=True, color=COLOR_MUTED)
    FONT_SECTION = Font(name="Segoe UI", size=12, bold=True, color=COLOR_NAVY)
    FONT_HEADER = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    FONT_BOLD = Font(name="Segoe UI", size=10, bold=True, color="000000")
    FONT_REGULAR = Font(name="Segoe UI", size=10, bold=False, color="000000")
    FONT_KPI_VAL = Font(name="Segoe UI", size=14, bold=True, color=COLOR_NAVY)

    # Accounting Borders
    BORDER_THIN_GRAY = Side(border_style="thin", color="D9D9D9")
    BORDER_MEDIUM_NAVY = Side(border_style="medium", color=COLOR_NAVY)
    BORDER_DOUBLE_BLACK = Side(border_style="double", color="000000")

    BORDER_CELL = Border(
        left=BORDER_THIN_GRAY, right=BORDER_THIN_GRAY,
        top=BORDER_THIN_GRAY, bottom=BORDER_THIN_GRAY
    )
    BORDER_TOTAL = Border(top=Side(border_style="thin", color="000000"), bottom=BORDER_DOUBLE_BLACK)

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_deal_model(
        self,
        well_name: str,
        fit_results: dict,
        valuation_results: dict,
        monte_carlo_results: dict = None,
        filename: str = "Volve_Asset_Valuation_Model.xlsx"
    ) -> str:
        """Exports single-well deal valuation model with optional Monte Carlo analytics."""
        file_path = os.path.join(self.output_dir, filename)
        wb = openpyxl.Workbook()

        # Sheet 1: Executive Deal Summary
        ws_summary = wb.active
        ws_summary.title = "Executive Summary"
        self._build_executive_summary(ws_summary, well_name, fit_results, valuation_results["metrics"])

        # Sheet 2: Monthly Cash Flow Waterfall
        ws_cf = wb.create_sheet(title="Monthly Cash Flows")
        self._build_cash_flow_sheet(ws_cf, valuation_results["cash_flow_table"])

        # Optional Sheet 3: Monte Carlo Percentile Risk Audit
        if monte_carlo_results:
            ws_mc = wb.create_sheet(title="Monte Carlo Risk Audit")
            self._build_monte_carlo_sheet(ws_mc, monte_carlo_results)

        wb.save(file_path)
        return file_path

    def export_field_consolidated_model(
        self,
        field_name: str,
        well_models: dict,
        monte_carlo_results: dict = None,
        filename: str = "Volve_Field_Consolidated_Model.xlsx"
    ) -> str:
        """
        Exports field-wide consolidated model across multiple wellbore assets.
        well_models structure:
            {
                "Wellbore_A": {"fit_results": dict, "valuation_results": dict},
                "Wellbore_B": {"fit_results": dict, "valuation_results": dict}, ...
            }
        """
        file_path = os.path.join(self.output_dir, filename)
        wb = openpyxl.Workbook()

        # Sheet 1: Consolidated Field Rollup
        ws_field = wb.active
        ws_field.title = "Consolidated Field Summary"
        self._build_field_summary(ws_field, field_name, well_models)

        # Individual Asset Cash Flow Sheets
        for well_name, data in well_models.items():
            sheet_title = f"CF_{well_name}"[:31]  # Excel max sheet title length = 31
            ws_well = wb.create_sheet(title=sheet_title)
            self._build_cash_flow_sheet(ws_well, data["valuation_results"]["cash_flow_table"])

        # Monte Carlo Risk Audit (Field Level)
        if monte_carlo_results:
            ws_mc = wb.create_sheet(title="Field Monte Carlo Audit")
            self._build_monte_carlo_sheet(ws_mc, monte_carlo_results)

        wb.save(file_path)
        return file_path

    def _build_executive_summary(self, ws, well_name: str, fit_results: dict, metrics: dict):
        ws.views.sheetView[0].showGridLines = True

        # Document Header
        ws["A1"] = "VALUATION & SUBSURFACE AUDIT MEMORANDUM"
        ws["A1"].font = self.FONT_TITLE
        ws["A2"] = f"Asset Target: Volve Field — Wellbore {well_name} | Class: Proved Developed Producing (PDP)"
        ws["A2"].font = self.FONT_SUBTITLE

        # KPI Summary Cards (Rows 4-6)
        kpis = [
            ("POST-TAX NPV10", metrics["npv10_usd"], "$#,##0"),
            ("UNLEVERED IRR", metrics["irr_annual"], "0.0%"),
            ("MOIC", metrics["moic"], "0.00x"),
            ("ECONOMIC LIFE", f"{metrics['economic_limit_month']} Months", "@")
        ]

        cols = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
        for idx, (title, val, fmt) in enumerate(kpis):
            c1, c2 = cols[idx]
            ws.merge_cells(f"{c1}4:{c2}4")
            ws.merge_cells(f"{c1}5:{c2}5")

            title_cell = ws[f"{c1}4"]
            title_cell.value = title
            title_cell.font = Font(name="Segoe UI", size=9, bold=True, color="595959")
            title_cell.alignment = Alignment(horizontal="center", vertical="center")
            title_cell.fill = self.FILL_ACCENT

            val_cell = ws[f"{c1}5"]
            val_cell.value = val
            val_cell.font = self.FONT_KPI_VAL
            val_cell.alignment = Alignment(horizontal="center", vertical="center")
            val_cell.fill = self.FILL_ACCENT
            if fmt != "@":
                val_cell.number_format = fmt

        # Table 1: Arps Decline Curve Fit Parameters
        ws["A8"] = "1. Subsurface DCA Parameters (Arps Engine)"
        ws["A8"].font = self.FONT_SECTION

        dca_headers = ["Parameter Description", "Symbol", "Fitted Value", "Unit / Metric"]
        for col_idx, h in enumerate(dca_headers, 1):
            cell = ws.cell(row=9, column=col_idx, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center")

        dca_data = [
            ("Initial Oil Production Rate", "q_i", fit_results["qi"], "BBL / Month", "#,##0"),
            ("Initial Monthly Decline Rate", "D_i", fit_results["Di"], "% / Month", "0.00%"),
            ("Arps Hyperbolic Exponent", "b", fit_results["b"], "Dimensionless", "0.000"),
            ("Goodness of Fit (R-Squared)", "R²", fit_results["r2"], "Statistical Correlation", "0.0000"),
            ("Root Mean Square Error", "RMSE", fit_results["rmse"], "BBL / Month", "#,##0")
        ]

        for r_idx, (desc, sym, val, unit, fmt) in enumerate(dca_data, 10):
            ws.cell(row=r_idx, column=1, value=desc).font = self.FONT_REGULAR
            ws.cell(row=r_idx, column=2, value=sym).alignment = Alignment(horizontal="center")
            
            val_cell = ws.cell(row=r_idx, column=3, value=val)
            val_cell.font = self.FONT_BOLD
            val_cell.number_format = fmt
            val_cell.alignment = Alignment(horizontal="right")

            ws.cell(row=r_idx, column=4, value=unit).font = self.FONT_REGULAR

        # Table 2: Financial Valuation Metrics
        ws["A17"] = "2. Financial & Underwriting Metrics"
        ws["A17"].font = self.FONT_SECTION

        fin_headers = ["Metric", "Value", "Benchmark Target"]
        for col_idx, h in enumerate(fin_headers, 1):
            cell = ws.cell(row=18, column=col_idx, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center")

        fin_data = [
            ("Post-Tax NPV10", metrics["npv10_usd"], "$#,##0", "Primary Asset Base"),
            ("10-Year Gross Revenues", metrics["gross_revenue_usd"], "$#,##0", "Hydrocarbon Sales"),
            ("10-Year Operating Expenses", metrics["total_opex_usd"], "$#,##0", "LOE + Water Handling"),
            ("10-Year Undiscounted NCF", metrics["total_ncf_usd"], "$#,##0", "Cumulative Cash Flow"),
            ("Economic Cutoff Threshold", metrics["economic_cutoff_bbl_month"], "#,##0.0 BBL/m", "Operating Breakeven")
        ]

        for r_idx, (m_title, val, fmt, b_mark) in enumerate(fin_data, 19):
            ws.cell(row=r_idx, column=1, value=m_title).font = self.FONT_REGULAR
            
            val_cell = ws.cell(row=r_idx, column=2, value=val)
            val_cell.font = self.FONT_BOLD
            val_cell.number_format = fmt
            val_cell.alignment = Alignment(horizontal="right")

            ws.cell(row=r_idx, column=3, value=b_mark).font = self.FONT_REGULAR

        self._auto_fit_columns(ws)

    def _build_field_summary(self, ws, field_name: str, well_models: dict):
        """Constructs field-level consolidated monthly cash flow schedule using dynamic Excel formulas."""
        ws.views.sheetView[0].showGridLines = True

        ws["A1"] = f"FIELD-LEVEL CONSOLIDATED ASSET MODEL — {field_name.upper()}"
        ws["A1"].font = self.FONT_TITLE
        ws["A2"] = f"Consolidated Subsurface & Cash Flow Rollup across {len(well_models)} Asset Wellbores"
        ws["A2"].font = self.FONT_SUBTITLE

        headers = [
            "Month (t)", "Period", "Field Oil Vol (BBL)", "Field Gas Vol (MCF)", "Field Water Vol (BBL)",
            "Field Gross Rev ($)", "Field Net Rev ($)", "Field Var OPEX ($)", "Field Water OPEX ($)",
            "Field Fixed OPEX ($)", "Field Total OPEX ($)", "Field EBITDA ($)", "Field ABEX ($)",
            "Field Tax Paid ($)", "Field Net Cash Flow ($)", "Field Disc NCF ($)"
        ]

        ws.row_dimensions[4].height = 28
        for col_num, h in enumerate(headers, 1):
            cell = ws.cell(row=4, column=col_num, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Retrieve length of months from first wellbore dataframe
        sample_df = list(well_models.values())[0]["valuation_results"]["cash_flow_table"]
        num_months = len(sample_df)
        well_sheets = [f"CF_{w}"[:31] for w in well_models.keys()]

        for idx in range(num_months):
            r = idx + 5
            t_m = idx + 1
            
            ws.cell(row=r, column=1, value=t_m).alignment = Alignment(horizontal="center")
            ws.cell(row=r, column=2, value=f"Month {t_m}").alignment = Alignment(horizontal="center")

            # Dynamic multi-sheet Excel SUM formulas across individual well sheets
            # E.g., =SUM('CF_Well1'!C5,'CF_Well2'!C5,...)
            cols_map = [
                (3, "C", "#,##0"),      # Oil Vol
                (4, "D", "#,##0"),      # Gas Vol
                (5, "E", "#,##0"),      # Water Vol
                (6, "F", "$#,##0"),     # Gross Rev
                (7, "G", "$#,##0"),     # Net Rev
                (8, "H", "$#,##0"),     # Var OPEX
                (9, "I", "$#,##0"),     # Water OPEX
                (10, "J", "$#,##0"),    # Fixed OPEX
                (11, "K", "$#,##0"),    # Total OPEX
                (12, "L", "$#,##0"),    # EBITDA
                (13, "M", "$#,##0"),    # ABEX
                (14, "N", "$#,##0"),    # Tax
                (15, "O", "$#,##0"),    # NCF
                (16, "P", "$#,##0")     # Disc NCF
            ]

            for col_idx, col_let, fmt in cols_map:
                sheet_refs = ",".join([f"'{sheet}'!{col_let}{r}" for sheet in well_sheets])
                cell = ws.cell(row=r, column=col_idx, value=f"=SUM({sheet_refs})")
                cell.number_format = fmt
                if col_idx in (15, 16):
                    cell.font = self.FONT_BOLD

            if idx % 2 == 1:
                for c in range(1, 17):
                    ws.cell(row=r, column=c).fill = self.FILL_ZEBRA

        # Totals Row
        tot_r = num_months + 5
        tot_label = ws.cell(row=tot_r, column=1, value="FIELD TOTAL")
        tot_label.font = self.FONT_HEADER
        tot_label.alignment = Alignment(horizontal="center")

        for c in range(1, 17):
            cell = ws.cell(row=tot_r, column=c)
            cell.fill = self.FILL_NAVY
            cell.border = self.BORDER_TOTAL

        for c_idx, col_let in enumerate(["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"], 3):
            cell = ws.cell(row=tot_r, column=c_idx, value=f"=SUM({col_let}5:{col_let}{tot_r-1})")
            cell.font = self.FONT_HEADER
            cell.number_format = "$#,##0" if c_idx >= 6 else "#,##0"

        self._auto_fit_columns(ws)

    def _build_monte_carlo_sheet(self, ws, mc_results: dict):
        """Builds Monte Carlo simulation percentile summary and iteration audit log."""
        ws.views.sheetView[0].showGridLines = True

        ws["A1"] = "MONTE CARLO PROBABILISTIC RISK AUDIT"
        ws["A1"].font = self.FONT_TITLE
        ws["A2"] = f"Uncertainty Quantification ({mc_results.get('iterations', 1000):,} Stochastic Trials)"
        ws["A2"].font = self.FONT_SUBTITLE

        # Table 1: Percentile Summary (P10, P50, P90 Distribution)
        ws["A4"] = "1. Reserve & Valuation Percentile Metrics"
        ws["A4"].font = self.FONT_SECTION

        perc_headers = ["Metric Description", "P90 (Conservative)", "P50 (Median)", "P10 (Optimistic)", "Mean / Expected"]
        for col_idx, h in enumerate(perc_headers, 1):
            cell = ws.cell(row=5, column=col_idx, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center")

        percentile_data = [
            ("Estimated Ultimate Recovery (MMbbl)", mc_results.get("eur_p90"), mc_results.get("eur_p50"), mc_results.get("eur_p10"), mc_results.get("eur_mean"), "0.00"),
            ("Post-Tax NPV10 ($MM)", mc_results.get("npv_p90"), mc_results.get("npv_p50"), mc_results.get("npv_p10"), mc_results.get("npv_mean"), "$#,##0.00"),
            ("Unlevered IRR (%)", mc_results.get("irr_p90"), mc_results.get("irr_p50"), mc_results.get("irr_p10"), mc_results.get("irr_mean"), "0.0%")
        ]

        for r_idx, (m_label, p90, p50, p10, mean_val, fmt) in enumerate(percentile_data, 6):
            ws.cell(row=r_idx, column=1, value=m_label).font = self.FONT_BOLD
            for c_i, val in enumerate([p90, p50, p10, mean_val], 2):
                cell = ws.cell(row=r_idx, column=c_i, value=val)
                cell.font = self.FONT_REGULAR
                cell.number_format = fmt
                cell.alignment = Alignment(horizontal="right")

        # Table 2: Simulated Trial Audit Log (First 100 Iterations preview)
        ws["A11"] = "2. Stochastic Iteration Audit Log (Sample Runs)"
        ws["A11"].font = self.FONT_SECTION

        trial_headers = ["Trial #", "Oil Price ($/BBL)", "qi (BBL/m)", "Di (%/m)", "b-exponent", "EUR (MMbbl)", "NPV10 ($)", "IRR (%)"]
        for col_idx, h in enumerate(trial_headers, 1):
            cell = ws.cell(row=12, column=col_idx, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center")

        trials_df = mc_results.get("trials_df", pd.DataFrame())
        if not trials_df.empty:
            for idx, row in trials_df.head(100).iterrows():
                r = idx + 13
                ws.cell(row=r, column=1, value=int(row.get("trial", idx+1))).alignment = Alignment(horizontal="center")
                ws.cell(row=r, column=2, value=row.get("oil_price", 0)).number_format = "$#,##0.00"
                ws.cell(row=r, column=3, value=row.get("qi", 0)).number_format = "#,##0"
                ws.cell(row=r, column=4, value=row.get("Di", 0)).number_format = "0.00%"
                ws.cell(row=r, column=5, value=row.get("b", 0)).number_format = "0.000"
                ws.cell(row=r, column=6, value=row.get("eur_mmbbl", 0)).number_format = "0.00"
                ws.cell(row=r, column=7, value=row.get("npv10", 0)).number_format = "$#,##0"
                ws.cell(row=r, column=8, value=row.get("irr", 0)).number_format = "0.0%"

                if idx % 2 == 1:
                    for c in range(1, 9):
                        ws.cell(row=r, column=c).fill = self.FILL_ZEBRA

        self._auto_fit_columns(ws)

    def _build_cash_flow_sheet(self, ws, cf_df: pd.DataFrame):
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Month (t)", "Period", "Oil Vol (BBL)", "Gas Vol (MCF)", "Water Vol (BBL)",
            "Gross Rev ($)", "Net Rev ($)", "Var OPEX ($)", "Water OPEX ($)", "Fixed OPEX ($)",
            "Total OPEX ($)", "EBITDA ($)", "ABEX ($)", "Tax Paid ($)", "Net Cash Flow ($)", "Discounted NCF ($)"
        ]

        ws.row_dimensions[1].height = 28
        for col_num, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num, value=h)
            cell.fill = self.FILL_NAVY
            cell.font = self.FONT_HEADER
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Write Data & Inject Dynamic Formulas
        for idx, row in cf_df.iterrows():
            r = idx + 2
            
            ws.cell(row=r, column=1, value=int(row["t_month"])).alignment = Alignment(horizontal="center")
            
            date_val = str(row["date"])[:10] if "date" in row else f"Month {int(row['t_month'])}"
            ws.cell(row=r, column=2, value=date_val).alignment = Alignment(horizontal="center")

            # Volumes
            ws.cell(row=r, column=3, value=row["oil_prod_bbl"]).number_format = "#,##0"
            ws.cell(row=r, column=4, value=row["gas_prod_mcf"]).number_format = "#,##0"
            ws.cell(row=r, column=5, value=row["water_prod_bbl"]).number_format = "#,##0"

            # Dynamic Formulas for Revenues and Expenses
            ws.cell(row=r, column=6, value=row["total_gross_revenue"]).number_format = "$#,##0"
            ws.cell(row=r, column=7, value=f"=F{r}").number_format = "$#,##0"  # Net Rev = Gross Rev (0% royalty)
            
            ws.cell(row=r, column=8, value=row["var_opex"]).number_format = "$#,##0"
            ws.cell(row=r, column=9, value=row["water_opex"]).number_format = "$#,##0"
            ws.cell(row=r, column=10, value=row["fixed_opex"]).number_format = "$#,##0"
            
            # Dynamic Total OPEX: =SUM(H{r}:J{r})
            ws.cell(row=r, column=11, value=f"=SUM(H{r}:J{r})").number_format = "$#,##0"
            
            # Dynamic EBITDA: =G{r}-K{r}
            ws.cell(row=r, column=12, value=f"=G{r}-K{r}").number_format = "$#,##0"
            
            ws.cell(row=r, column=13, value=row["abex"]).number_format = "$#,##0"
            ws.cell(row=r, column=14, value=row["tax_liability"]).number_format = "$#,##0"
            
            # Dynamic Net Cash Flow: =L{r}-M{r}-N{r}
            ncf_cell = ws.cell(row=r, column=15, value=f"=L{r}-M{r}-N{r}")
            ncf_cell.number_format = "$#,##0"
            ncf_cell.font = self.FONT_BOLD

            # Discounted NCF: =O{r}/((1+0.10)^(A{r}/12))
            disc_cell = ws.cell(row=r, column=16, value=f"=O{r}/((1+0.10)^(A{r}/12))")
            disc_cell.number_format = "$#,##0"

            if idx % 2 == 1:
                for c in range(1, 17):
                    ws.cell(row=r, column=c).fill = self.FILL_ZEBRA

        # Totals Row with Dynamic Formulas
        last_row = len(cf_df) + 1
        tot_r = last_row + 1

        tot_label = ws.cell(row=tot_r, column=1, value="TOTAL")
        tot_label.font = self.FONT_HEADER
        tot_label.alignment = Alignment(horizontal="center")

        for c in range(1, 17):
            cell = ws.cell(row=tot_r, column=c)
            cell.fill = self.FILL_NAVY
            cell.border = self.BORDER_TOTAL

        # Formula SUMs across columns
        for c_idx, col_let in enumerate(["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"], 3):
            cell = ws.cell(row=tot_r, column=c_idx, value=f"=SUM({col_let}2:{col_let}{last_row})")
            cell.font = self.FONT_HEADER
            cell.number_format = "$#,##0" if c_idx >= 6 else "#,##0"

        self._auto_fit_columns(ws)

    @staticmethod
    def _auto_fit_columns(ws):
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 14)
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import VolveDataLoader
from src.dca_engine import ArpsDCAEngine
from src.dcf_engine import UpstreamDCFEngine
from src.excel_exporter import InstitutionalExcelExporter
from src.field_aggregator import FieldAggregatorEngine
from src.monte_carlo import MonteCarloEngine

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Volve Field Asset Valuation & Reserve Engineering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
    <style>
    .main-header { font-size: 26px; font-weight: 700; color: #1B365D; margin-bottom: 0px; }
    .sub-header { font-size: 14px; color: #595959; margin-bottom: 20px; }
    .metric-card { background-color: #F8F9FA; border-left: 5px solid #1B365D; padding: 15px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CACHING
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "raw", "volve_production.xlsx")

@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at `{DATA_PATH}`. Please check the file path.")
        st.stop()
    loader = VolveDataLoader(DATA_PATH)
    return loader

@st.cache_data
def get_producing_wells(_loader):
    all_wells = _loader.get_available_wells()
    producing_wells = []
    for well in all_wells:
        df_temp = _loader.clean_well_data(well)
        if len(df_temp) >= 3:
            producing_wells.append(well)
    return producing_wells

loader = load_and_prep_data()
available_wells = get_producing_wells(loader)

# Safety check in case no wells pass the threshold
if not available_wells:
    st.error("No wellbores found with sufficient historical production data (≥3 records).")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & ASSUMPTIONS
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ Upstream Underwriting")
st.sidebar.markdown("---")

# Asset Scope & Analysis Mode
st.sidebar.subheader("Asset Scope")
analysis_mode = st.sidebar.radio(
    "Analysis Mode", ["Single Wellbore", "Field-Level Portfolio"]
)

if analysis_mode == "Single Wellbore":
    selected_well = st.sidebar.selectbox("Select Target Wellbore", available_wells)
    selected_wells = [selected_well]
else:
    selected_wells = st.sidebar.multiselect(
        "Select Field Wellbores", available_wells, default=available_wells
    )
    if not selected_wells:
        st.error("Please select at least one wellbore for field-level evaluation.")
        st.stop()
    selected_well = selected_wells[0]  # Primary well reference for single fit displays

st.sidebar.subheader("Subsurface Controls")
forecast_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=12, max_value=240, value=120, step=12)
d_lim_annual = st.sidebar.slider("Terminal Decline Rate (D_lim)", min_value=0.03, max_value=0.15, value=0.06, step=0.01)

st.sidebar.subheader("Commodity & Fiscal Pricing")
oil_price = st.sidebar.number_input("Realized Oil Price ($/BBL)", value=75.0, step=2.5)
gas_price = st.sidebar.number_input("Realized Gas Price ($/MCF)", value=3.50, step=0.25)
disc_rate = st.sidebar.slider("Annual Discount Rate (NPV)", min_value=0.05, max_value=0.25, value=0.10, step=0.01)

st.sidebar.subheader("OPEX & Fiscal Terms")
var_opex = st.sidebar.number_input("Variable LOE ($/BBL)", value=12.0, step=1.0)
water_cost = st.sidebar.number_input("Water Handling ($/BBL)", value=1.50, step=0.25)
fixed_opex = st.sidebar.number_input("Fixed LOE ($/Month)", value=25000.0, step=2500.0)

if analysis_mode == "Single Wellbore":
    initial_capex = st.sidebar.number_input("Initial Acquisition CAPEX ($)", value=5000000.0, step=250000.0)
else:
    field_capex = st.sidebar.number_input("Field Infrastructure CAPEX ($)", value=15000000.0, step=500000.0)
    initial_capex = field_capex

apply_tax = st.sidebar.checkbox("Apply Petroleum Tax Model (78%)", value=True)

# Combined variable operating cost per BBL
effective_var_opex = var_opex + water_cost

dcf_params = {
    "oil_price_usd": oil_price,
    "gas_price_mcf": gas_price,
    "variable_opex_bbl": effective_var_opex,
    "fixed_opex_month": fixed_opex,
    "discount_rate": disc_rate,
    "apply_ncs_tax": apply_tax,
}

# -----------------------------------------------------------------------------
# 4. SUBSURFACE DCA & DCF CALCULATIONS
# -----------------------------------------------------------------------------
dca = ArpsDCAEngine()
dcf = UpstreamDCFEngine(**dcf_params)

if analysis_mode == "Single Wellbore":
    df_clean = loader.clean_well_data(selected_well)

    if len(df_clean) < 3:
        st.error(f"⚠️ **Insufficient Historical Data**: Wellbore **{selected_well}** has only {len(df_clean)} active production record(s). Arps non-linear regression requires a minimum of 3 historical data points.")
        st.info("Select another wellbore from the sidebar to continue analysis.")
        st.stop()

    try:
        fit_results = dca.fit_decline_curve(df_clean['t_month'].values, df_clean['monthly_oil_bbl'].values)
    except ValueError as e:
        st.error(f"⚠️ **Decline Curve Fitting Error**: {e}")
        st.stop()

    forecast_df = dca.forecast_production(
        qi=fit_results['qi'],
        Di=fit_results['Di'],
        b=fit_results['b'],
        forecast_months=forecast_months,
        d_lim_annual=d_lim_annual
    )

    possible_oil_cols = ['oil_prod_bbl', 'forecast_monthly_bbl', 'monthly_oil_bbl', 'q_oil', 'oil_bbl', 'q']
    oil_col = next((col for col in possible_oil_cols if col in forecast_df.columns), forecast_df.columns[1])

    forecast_df['oil_prod_bbl'] = forecast_df[oil_col]
    forecast_df['forecast_monthly_bbl'] = forecast_df[oil_col]

    valuation = dcf.run_cash_flow_model(forecast_df, initial_capex=initial_capex)
    metrics = valuation["metrics"]
    cf_df = valuation["cash_flow_table"]

else:
    # Field-Level Aggregation Mode Execution
    aggregator = FieldAggregatorEngine(loader, dcf_params)
    field_res = aggregator.aggregate_field(
        well_list=selected_wells,
        forecast_months=forecast_months,
        d_lim_annual=d_lim_annual,
        total_field_capex=initial_capex
    )

    cf_df = field_res["field_df"]
    f_metrics = field_res["field_metrics"]
    metrics = {
        "npv10_usd": f_metrics["field_npv"],
        "irr_annual": f_metrics["field_irr"],
        "moic": (f_metrics["total_ncf_usd"] + initial_capex) / initial_capex if initial_capex > 0 else 0.0,
        "total_ncf_usd": f_metrics["total_ncf_usd"],
        "economic_limit_month": forecast_months
    }

    # Reference single well DCA fit for primary baseline simulation runs
    df_clean = loader.clean_well_data(selected_well)
    fit_results = dca.fit_decline_curve(df_clean['t_month'].values, df_clean['monthly_oil_bbl'].values)
    forecast_df = cf_df

# -----------------------------------------------------------------------------
# 5. DASHBOARD HEADER & KPI CARDS
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>VOLVE ASSET VALUATION & SUBSURFACE AUDIT</div>", unsafe_allow_html=True)
if analysis_mode == "Single Wellbore":
    st.markdown(f"<div class='sub-header'>Target Asset: <b>{selected_well}</b> | Reserve Classification: Proved Developed Producing (PDP)</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='sub-header'>Scope: <b>Field-Level Portfolio ({len(selected_wells)} Active Wells)</b> | Reserve Classification: Proved Developed Producing (PDP)</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Post-Tax NPV10", f"${metrics['npv10_usd']:,.0f}")
col2.metric("Unlevered IRR", f"{metrics['irr_annual']*100:.1f}%" if isinstance(metrics['irr_annual'], float) else "N/A")
col3.metric("MOIC", f"{metrics['moic']:.2f}x")
col4.metric("Total Net Cash Flow", f"${metrics['total_ncf_usd']:,.0f}")
col5.metric("Forecast Horizon", f"{forecast_months} Months")

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. VISUALIZATION TABS
# -----------------------------------------------------------------------------
tab_dca, tab_cf, tab_sens, tab_mc, tab_export = st.tabs([
    "📈 Subsurface Production Profile", 
    "💰 Monthly Cash Flow Waterfall", 
    "🎯 Valuation Sensitivity Matrix", 
    "🎲 Monte Carlo Risk Simulation",
    "📥 Institutional Export"
])

# TAB 1: Production Profile Plot
with tab_dca:
    if analysis_mode == "Single Wellbore":
        st.subheader("Arps Decline Curve Fit (Semi-Log Production Profile)")
        fig_dca = go.Figure()

        fig_dca.add_trace(go.Scatter(
            x=df_clean['t_month'], y=df_clean['monthly_oil_bbl'],
            mode='markers', name='Historical Oil Production',
            marker=dict(color='#1B365D', size=6, symbol='circle')
        ))

        fig_dca.add_trace(go.Scatter(
            x=forecast_df['t_month'], y=forecast_df['oil_prod_bbl'],
            mode='lines', name=f"Arps Forecast (b={fit_results['b']:.2f}, R²={fit_results['r2']:.3f})",
            line=dict(color='#D9534F', width=2.5)
        ))

        fig_dca.update_layout(
            xaxis_title="Time (Months)",
            yaxis_title="Monthly Oil Production (BBL)",
            yaxis_type="log",
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_dca, use_container_width=True)

        st.markdown("**Fitted Hyperbolic Decline Parameters**")
        dca_col1, dca_col2, dca_col3, dca_col4 = st.columns(4)
        dca_col1.write(f"**q_i:** {fit_results['qi']:,.0f} BBL/m")
        dca_col2.write(f"**D_i:** {fit_results['Di']*100:.2f}% / month")
        dca_col3.write(f"**b-factor:** {fit_results['b']:.3f}")
        dca_col4.write(f"**Fit Quality (R²):** {fit_results['r2']:.4f}")
    else:
        st.subheader("Consolidated Monthly Field Production Profile (BBL)")
        fig_field = go.Figure()
        fig_field.add_trace(go.Scatter(
            x=cf_df["t_month"],
            y=cf_df["oil_prod_bbl"],
            mode="lines",
            fill="tozeroy",
            name="Field Consolidated Production",
            line=dict(color="#1B365D", width=2.5)
        ))
        fig_field.update_layout(
            xaxis_title="Time (Months)",
            yaxis_title="Monthly Oil Production (BBL)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_field, use_container_width=True)

        st.subheader("Individual Wellbore Breakdown")
        st.dataframe(field_res["well_summaries"].style.format({
            "qi": "{:,.0f}",
            "b": "{:.3f}",
            "eur_bbl": "{:,.0f} BBL",
            "well_npv10": "${:,.0f}"
        }), use_container_width=True)

# TAB 2: Cash Flow Waterfall Chart
with tab_cf:
    st.subheader("Monthly Net Cash Flow Profile ($)")
    
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=cf_df['t_month'], y=cf_df['net_cash_flow'],
        name='Net Cash Flow ($)',
        marker_color=np.where(cf_df['net_cash_flow'] < 0, '#D9534F', '#2E7D32')
    ))

    fig_cf.update_layout(
        xaxis_title="Time (Months)",
        yaxis_title="Net Cash Flow ($USD)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    fmt_dict = {col: "${:,.0f}" for col in cf_df.columns if col not in ['t_month', 'oil_prod_bbl', 'forecast_monthly_bbl']}
    if 'oil_prod_bbl' in cf_df.columns:
        fmt_dict['oil_prod_bbl'] = "{:,.0f}"

    with st.expander("Inspect Raw Monthly Cash Flow Table"):
        st.dataframe(cf_df.style.format(fmt_dict), height=300)

# TAB 3: Sensitivity Matrix (NPV vs Oil Price & Discount Rate)
with tab_sens:
    st.subheader("NPV Sensitivity Matrix ($MM)")
    st.markdown("Post-Tax Net Present Value evaluated across varying Crude Oil Prices and Discount Rates.")

    oil_prices_sens = [50.0, 60.0, 70.0, 75.0, 85.0, 95.0, 105.0]
    disc_rates_sens = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    matrix_data = []
    for r in disc_rates_sens:
        row = []
        for p in oil_prices_sens:
            s_dcf = UpstreamDCFEngine(
                oil_price_usd=p,
                gas_price_mcf=gas_price,
                variable_opex_bbl=effective_var_opex,
                fixed_opex_month=fixed_opex,
                discount_rate=r,
                apply_ncs_tax=apply_tax
            )
            res = s_dcf.run_cash_flow_model(forecast_df, initial_capex=initial_capex)
            row.append(res["metrics"]["npv10_usd"] / 1e6)
        matrix_data.append(row)

    sens_df = pd.DataFrame(
        matrix_data,
        index=[f"{int(r*100)}%" for r in disc_rates_sens],
        columns=[f"${p:.0f}" for p in oil_prices_sens]
    )

    st.dataframe(
        sens_df.style.background_gradient(cmap="Blues", axis=None).format("${:.2f}M"),
        use_container_width=True
    )

# TAB 4: Monte Carlo Simulation
with tab_mc:
    st.subheader("Probabilistic NPV & IRR Distribution (Monte Carlo)")
    st.markdown("Evaluate risk profiles across stochastic iterations of Crude Oil Prices, OPEX, and Hyperbolic $b$-factors.")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    iterations = mc_col1.slider("Simulation Runs", min_value=250, max_value=2000, value=1000, step=250)
    oil_std = mc_col2.slider("Oil Price Volatility ($\sigma$)", min_value=2.0, max_value=25.0, value=10.0, step=1.0)
    opex_std = mc_col3.slider("OPEX Variance ($\pm\%$)", min_value=0.05, max_value=0.35, value=0.15, step=0.05)

    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        mc_engine = MonteCarloEngine(fit_results, dcf_params)
        
        with st.spinner("Running stochastic iterations..."):
            sim_results = mc_engine.run_simulation(
                iterations=iterations,
                oil_price_std=oil_std,
                opex_std_pct=opex_std,
                forecast_months=forecast_months,
                d_lim_annual=d_lim_annual,
                initial_capex=initial_capex
            )

        npv_p90 = np.percentile(sim_results['npv10_mm'], 10)
        npv_p50 = np.percentile(sim_results['npv10_mm'], 50)
        npv_p10 = np.percentile(sim_results['npv10_mm'], 90)

        irr_p90 = np.percentile(sim_results['irr_annual'], 10) * 100
        irr_p50 = np.percentile(sim_results['irr_annual'], 50) * 100
        irr_p10 = np.percentile(sim_results['irr_annual'], 90) * 100

        st.markdown("### **Reserve Probabilities (P-Values)**")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("P90 (Downside)", f"${npv_p90:.2f} MM", f"IRR: {irr_p90:.1f}%")
        p_col2.metric("P50 (Median Base)", f"${npv_p50:.2f} MM", f"IRR: {irr_p50:.1f}%")
        p_col3.metric("P10 (Upside)", f"${npv_p10:.2f} MM", f"IRR: {irr_p10:.1f}%")

        st.markdown("### **NPV10 Outcome Distribution**")
        fig_mc = go.Figure()
        
        fig_mc.add_trace(go.Histogram(
            x=sim_results['npv10_mm'],
            nbinsx=40,
            name='NPV Distribution',
            marker_color='#1B365D',
            opacity=0.75
        ))

        fig_mc.add_vline(x=npv_p90, line_dash="dash", line_color="#D9534F", annotation_text="P90 (Downside)")
        fig_mc.add_vline(x=npv_p50, line_dash="solid", line_color="#2E7D32", annotation_text="P50 (Median)")
        fig_mc.add_vline(x=npv_p10, line_dash="dash", line_color="#0275d8", annotation_text="P10 (Upside)")

        fig_mc.update_layout(
            xaxis_title="Post-Tax NPV10 ($MM)",
            yaxis_title="Frequency / Iterations",
            template="plotly_white",
            height=450,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        with st.expander("Inspect Raw Simulation Sample Data"):
            st.dataframe(sim_results.style.format({
                'npv10_mm': "${:.2f}M",
                'irr_annual': "{:.2%}",
                'eur_mmbbl': "{:.2f} MMbbl",
                'oil_price': "${:.2f}",
                'sampled_qi': "{:,.0f}",
                'sampled_b': "{:.2f}"
            }), height=250)

# TAB 5: Excel Report Exporter
with tab_export:
    st.subheader("Generate Audit-Ready Financial Model (.xlsx)")
    st.markdown(
        "Export the fitted subsurface model, production forecast, and dynamic cash flow waterfall "
        "into a Tier-1 investment banking workbook matching Wall Street presentation standards."
    )

    if st.button("🚀 Build & Export Excel Model", type="primary"):
        exporter = InstitutionalExcelExporter()
        
        safe_well_name = str(selected_well).replace("/", "_").replace("\\", "_").replace(" ", "_")
        output_filename = f"Volve_{safe_well_name}_Valuation_Model.xlsx"
        
        saved_file = exporter.export_deal_model(selected_well, fit_results, valuation, filename=output_filename)
        
        with open(saved_file, "rb") as f:
            st.download_button(
                label="📥 Download Excel File",
                data=f,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.success(f"Successfully generated `{output_filename}`!")
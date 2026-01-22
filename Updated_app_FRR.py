
import time
from io import BytesIO
from collections import OrderedDict
import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ----------------------------- Page Config -----------------------------
st.set_page_config(page_title="AFRR Bids Explorer", layout="wide")
st.title("AFRR Auction Results Explorer")

# ----------------------------- Constants ------------------------------
PERIODS_4H = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']  # six 4h periods
DATASET = "ods125"
BASE_URL = "https://opendata.elia.be/api/records/1.0/search/"

# ----------------------------- Helpers -------------------------------

def to_float(series, default=0.0):
    """Safe numeric conversion to float with fallback default."""
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

def coerce_selected(s: pd.Series) -> pd.Series:
    """Coerce selectedbyoptimizer to boolean robustly."""
    return s.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))

def initial_ymax(v):
    """Initial y-max = rounded(max * 1.1), with safe lower bound."""
    if v <= 0:
        return 500
    return int(round(v * 1.1))

def _save_fig_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

def normalize_period(val: object) -> str:
    """
    Normalize 'capacitybiddeliveryperiod' values to a canonical form:
    - Unify dash characters, collapse spaces around dash.
    - Ensure 0-24 variants map to '0 - 24'.
    - Return clean string (not categorical).
    """
    s = str(val).strip()
    s = s.replace('–', '-').replace('—', '-')  # en/em dashes -> hyphen
    s = re.sub(r'\s*-\s*', ' - ', s)          # spaces around hyphen
    s = re.sub(r'\s+', ' ', s)                # collapse multiple spaces
    # Canonicalize common 0-24 variants
    if s in {'0-24', '0 -24', '0- 24', '0 – 24', '0 — 24', '0 - 24'}:
        return '0 - 24'
    return s

# ----------------------------- Data Fetch (Pagination + Retries) -----------------------------

@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch(dataset: str, d: str, rows_per_page: int = 10000, max_pages: int = 20,
          retries: int = 3, backoff: float = 0.7) -> pd.DataFrame:
    """
    Fetches all records for a given delivery date using pagination and simple retries.
    """
    all_records = []
    start = 0
    headers = {"User-Agent": "AFRR-Explorer/1.0"}

    for page in range(max_pages):
        params = {
            "dataset": dataset,
            "rows": rows_per_page,
            "start": start,
            "refine.deliverydate": d
        }

        last_exc = None
        for attempt in range(retries):
            try:
                r = requests.get(BASE_URL, params=params, timeout=30, headers=headers)
                r.raise_for_status()
                js = r.json()
                recs = js.get("records", [])
                all_records.extend([rec.get('fields', {}) for rec in recs])

                # pagination: stop if fewer than requested rows
                if len(recs) < rows_per_page:
                    return pd.DataFrame(all_records)

                start += rows_per_page
                break  # next page
            except Exception as e:
                last_exc = e
                time.sleep(backoff * (2 ** attempt))
        else:
            # All retries failed for this page
            st.warning(f"Partial data fetched. Error during fetch at start={start}: {last_exc}")
            break

    return pd.DataFrame(all_records)

# ----------------------------- Computations (Vectorized) -----------------------------

@st.cache_data(show_spinner=False)
def process_data(df_raw: pd.DataFrame):
    """
    Cleans raw DataFrame, ensures required columns, types, and returns:
    - df: cleaned with 'period_str' (string) and 'period_4h' (categorical)  # <<<
    - allcctu: all rows where period_str == '0 - 24'                        # <<<
    - allcctu_sel: same but selected
    - dynamic_ymax_up, dynamic_ymax_down
    """
    df = df_raw.copy()

    # Ensure expected columns
    df = ensure_cols(df, [
        'afrrofferedvolumeupmw', 'afrrofferedvolumedownmw',
        'afrrawardedvolumeupmw', 'afrrawardedvolumedownmw',
        'priceupmwh', 'pricedownmwh', 'selectedbyoptimizer',
        'capacitybiddeliveryperiod', 'index'
    ])

    # Coerce types
    for c in ['priceupmwh', 'pricedownmwh', 'afrrofferedvolumeupmw', 'afrrofferedvolumedownmw',
              'afrrawardedvolumeupmw', 'afrrawardedvolumedownmw']:
        df[c] = to_float(df[c])

    # selected flag robustly to boolean
    df['selectedbyoptimizer'] = coerce_selected(df['selectedbyoptimizer'])

    # --- Period handling: keep a clean string and a separate 4h categorical  --- # <<<
    df['period_str'] = df['capacitybiddeliveryperiod'].map(normalize_period)

    # Only 4h periods get a categorical; others (e.g., '0 - 24') become <NA> in period_4h
    df['period_4h'] = pd.Categorical(
        df['period_str'].where(df['period_str'].isin(PERIODS_4H), pd.NA),
        categories=PERIODS_4H,
        ordered=True
    )

    # Dynamic ymax per date (consider only 6x 4h periods)
    df_4h = df[df['period_4h'].notna()]
    max_up_price = float(df_4h['priceupmwh'].max()) if not df_4h.empty else 0.0
    max_down_price = float(df_4h['pricedownmwh'].max()) if not df_4h.empty else 0.0
    dynamic_ymax_up = initial_ymax(max_up_price)
    dynamic_ymax_down = initial_ymax(max_down_price)

    # All-CCTU 0-24 (use normalized string)
    allcctu = df[df['period_str'] == '0 - 24'].copy()       # <<<
    allcctu_sel = allcctu[allcctu['selectedbyoptimizer']].copy()

    return df, allcctu, allcctu_sel, dynamic_ymax_up, dynamic_ymax_down

@st.cache_data(show_spinner=False)
def build_summary_and_costs(df: pd.DataFrame, allcctu_sel: pd.DataFrame):
    """
    Builds the summary table and cost breakdown.
    Returns summary_df, cost_up, cost_down, cost_allcctu, total_cost
    """
    # Summary by period/direction for the 6x 4-hour periods (use period_4h)  # <<<
    d4 = df[df['period_4h'].notna()].copy()

    # Total submitted per period/direction (using offered volumes)
    agg_submitted = d4.groupby('period_4h').agg(
        tot_off_up=('afrrofferedvolumeupmw', 'sum'),
        tot_off_down=('afrrofferedvolumedownmw', 'sum')
    )

    # Selected-only subset for averages/marginals
    sel = d4[d4['selectedbyoptimizer']].copy()

    # UP aggregates
    grp_up = sel.groupby('period_4h').agg(
        tot_awarded=('afrrawardedvolumeupmw', 'sum'),
        w_price_sum=('priceupmwh', lambda s: (s * sel.loc[s.index, 'afrrawardedvolumeupmw']).sum()),
        marginal=('priceupmwh', 'max')
    )
    grp_up['avg_price'] = grp_up.apply(
        lambda r: r['w_price_sum'] / r['tot_awarded'] if r['tot_awarded'] > 0 else 0.0, axis=1
    )

    # DOWN aggregates
    grp_dn = sel.groupby('period_4h').agg(
        tot_awarded=('afrrawardedvolumedownmw', 'sum'),
        w_price_sum=('pricedownmwh', lambda s: (s * sel.loc[s.index, 'afrrawardedvolumedownmw']).sum()),
        marginal=('pricedownmwh', 'max')
    )
    grp_dn['avg_price'] = grp_dn.apply(
        lambda r: r['w_price_sum'] / r['tot_awarded'] if r['tot_awarded'] > 0 else 0.0, axis=1
    )

    # Build tidy summary
    sum_up = pd.DataFrame({
        "Period": grp_up.index.astype(str),
        "Direction": "UP",
        "Total Awarded Volume (MW)": grp_up['tot_awarded'].round(2),
        "Average Price (€/MWh)": grp_up['avg_price'].round(2),
        "Marginal Price (€/MWh)": grp_up['marginal'].fillna(0.0).round(2),
        "Total Submitted Volume (MW)": agg_submitted['tot_off_up'].reindex(grp_up.index).fillna(0.0).round(2)
    })

    sum_dn = pd.DataFrame({
        "Period": grp_dn.index.astype(str),
        "Direction": "DOWN",
        "Total Awarded Volume (MW)": grp_dn['tot_awarded'].round(2),
        "Average Price (€/MWh)": grp_dn['avg_price'].round(2),
        "Marginal Price (€/MWh)": grp_dn['marginal'].fillna(0.0).round(2),
        "Total Submitted Volume (MW)": agg_submitted['tot_off_down'].reindex(grp_dn.index).fillna(0.0).round(2)
    })

    summary = pd.concat([sum_up, sum_dn], ignore_index=True)
    summary['Period'] = pd.Categorical(summary['Period'], categories=PERIODS_4H, ordered=True)
    summary = summary.sort_values(['Direction', 'Period'])

    # Costs (4h periods use 4 hours)
    def comp_direction(direction: str) -> float:
        sub = summary[summary["Direction"] == direction]
        return float((sub["Average Price (€/MWh)"] * sub["Total Awarded Volume (MW)"] * 4.0).sum())

    comp_up = comp_direction("UP")
    comp_down = comp_direction("DOWN")

    # All-CCTU cost component (24 hours)
    v_up_24 = to_float(allcctu_sel['afrrawardedvolumeupmw'])
    v_dn_24 = to_float(allcctu_sel['afrrawardedvolumedownmw'])
    p_up_24 = to_float(allcctu_sel['priceupmwh'])
    p_dn_24 = to_float(allcctu_sel['pricedownmwh'])
    comp_allcctu = float(((p_dn_24 * v_dn_24) + (p_up_24 * v_up_24)).sum() * 24.0)

    total_cost = comp_allcctu + comp_up + comp_down
    return summary, comp_up, comp_down, comp_allcctu, total_cost

# ----------------------------- Sidebar Controls -----------------------------

with st.sidebar:
    date = st.date_input("Select delivery date", help="Pick the date you want to analyze.")
    st.markdown("---")
    st.subheader("Chart Options")

    selected_metric = st.selectbox(
        "Bar charts metric",
        ["Average Price (€/MWh)", "Marginal Price (€/MWh)", "Total Awarded Volume (MW)", "Total Submitted Volume (MW)"],
        index=0,
        key=f"metric_select_{date}"  # date-scoped key to avoid session state conflicts
    )

    st.markdown("**Merit Order Options**")
    opt_show_step = st.checkbox("Show step (cumulative curve)", value=True, key="opt_show_step")
    opt_show_points = st.checkbox("Show scatter points", value=True, key="opt_show_points")
    opt_only_selected = st.checkbox("Show only selected points", value=False, key="opt_only_selected")
    opt_show_marginal_line = st.checkbox("Show marginal price line", value=True, key="opt_show_marginal_line")
    opt_sync_x_limits = st.checkbox("Sync X (volume) limits UP & DOWN", value=True, key="opt_sync_x_limits")
    opt_grid = st.checkbox("Show grid on charts", value=True, key="opt_grid")

# Reset per-date computed defaults when date changes
if 'last_date' not in st.session_state or st.session_state['last_date'] != str(date):
    st.session_state['last_date'] = str(date)
    st.session_state['ymax_up_input'] = None
    st.session_state['ymax_down_input'] = None

if not date:
    st.info("Please select a delivery date from the sidebar to begin.")
    st.stop()

# ----------------------------- Data Load -----------------------------
df_raw = fetch(DATASET, str(date))
if df_raw.empty:
    st.warning("No data found for that date.")
    st.stop()

df, allcctu, allcctu_sel, dynamic_ymax_up, dynamic_ymax_down = process_data(df_raw)
summary, comp_up, comp_down, comp_allcctu, total_cost = build_summary_and_costs(df, allcctu_sel)

# Initialize y-max inputs using dynamic values (once per date)
if st.session_state['ymax_up_input'] is None:
    st.session_state['ymax_up_input'] = dynamic_ymax_up
if st.session_state['ymax_down_input'] is None:
    st.session_state['ymax_down_input'] = dynamic_ymax_down

# ----------------------------- Metrics -----------------------------
st.subheader("Auction Cost Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Auction Cost (€)", f"{total_cost:,.0f}")
c2.metric("All-CCTU (0–24) Cost (€)", f"{comp_allcctu:,.0f}")
c3.metric("UP (periods) Cost (€)", f"{comp_up:,.0f}")
c4.metric("DOWN (periods) Cost (€)", f"{comp_down:,.0f}")

# ----------------------------- Centered Layout -----------------------------
col_left, col_main, col_right = st.columns([0.08, 0.84, 0.08])
with col_main:
    # ---------------- All-CCTU Table ----------------
    st.markdown("### All-CCTU (0–24): Selected Bids")
    st.markdown("Selected aFRR bids for the All-CCTU (0–24h) period.")
    if not allcctu_sel.empty:
        t = allcctu_sel[[
            'index', 'pricedownmwh', 'priceupmwh',
            'afrrofferedvolumedownmw', 'afrrawardedvolumedownmw', 'afrrawardedvolumeupmw'
        ]].copy()
        for c in ['pricedownmwh', 'priceupmwh', 'afrrofferedvolumedownmw', 'afrrawardedvolumedownmw', 'afrrawardedvolumeupmw']:
            t[c] = to_float(t[c]).round(2)
        t.columns = [
            "Index", "Price Down (€/MWh)", "Price Up (€/MWh)",
            "Offered Down (MW)", "Awarded Down (MW)", "Awarded Up (MW)"
        ]
        st.dataframe(t, use_container_width=True, height=240)
    else:
        st.info("No All-CCTU selected bids for this date.")

    # ---------------- Summary Table ----------------
    st.markdown("### Period Results Summary")
    st.markdown("This summary table shows total volumes and prices by direction and period.")
    st.dataframe(summary, use_container_width=True, height=320)

    # ---------------- Dual Bar Charts (UP vs DOWN) ----------------
    st.markdown("### Bar Charts by Period (UP vs DOWN)")
    st.write(
        "Choose a metric to visualize for the 6 periods. Left chart shows **UP**, right shows **DOWN**. "
        "Y-axes start at 0 and share a common max with small headroom."
    )

    metric_col = selected_metric
    # Prepare data for bars
    sum_up = summary[summary["Direction"] == "UP"].set_index("Period").reindex(PERIODS_4H)
    sum_dn = summary[summary["Direction"] == "DOWN"].set_index("Period").reindex(PERIODS_4H)

    y_up = to_float(sum_up[metric_col]).fillna(0.0)
    y_dn = to_float(sum_dn[metric_col]).fillna(0.0)

    combined_max = float(max(y_up.max(), y_dn.max()))
    y_top = 1.0 if combined_max <= 0 else combined_max * 1.05

    try:
        fig_bar, (ax_up, ax_dn) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    except Exception:
        fig_bar, (ax_up, ax_dn) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        fig_bar.subplots_adjust(top=0.86)

    # UP bars
    ax_up.bar(PERIODS_4H, y_up, color="#2E86AB", edgecolor="black")
    ax_up.set_title(f"UP – {selected_metric}")
    ax_up.set_ylabel(selected_metric)
    ax_up.set_ylim(bottom=0, top=y_top)
    ax_up.tick_params(axis='x', rotation=45)
    if opt_grid:
        ax_up.grid(True, axis='y', linestyle='--', alpha=0.5)

    # DOWN bars
    ax_dn.bar(PERIODS_4H, y_dn, color="#F39C12", edgecolor="black")
    ax_dn.set_title(f"DOWN – {selected_metric}")
    ax_dn.set_ylim(bottom=0, top=y_top)
    ax_dn.tick_params(axis='x', rotation=45)
    if opt_grid:
        ax_dn.grid(True, axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig_bar)
    bar_png = _save_fig_png(fig_bar)

    # ---------------- Scatter Plot (All-CCTU) ----------------
    st.markdown("---")
    st.markdown("## Scatter Plot: Offered UP vs Upward Price (All-CCTU 0–24)")
    st.write(
        "Scatter of offered upward volumes vs upward prices for the All-CCTU (0–24h) period. "
        "Use the dropdown to filter by a specific **Offered Volume Down** value."
    )

    vals = allcctu['afrrofferedvolumedownmw'].dropna().unique()
    try:
        vals = sorted(list({float(v) for v in vals}))
    except Exception:
        vals = []

    if len(vals) == 0:
        st.info("No data to select for 'Offered Volume Down'.")
    else:
        selected_value = st.selectbox(
            "Select the 'Offered Volume Down' for filtering:",
            vals, index=0, key=f"ovd_select_{date}"
        )
        mask0 = (to_float(allcctu['afrrofferedvolumedownmw']) == float(selected_value))
        p0 = allcctu[mask0]
        if not p0.empty:
            x = to_float(p0['afrrofferedvolumeupmw'])
            y = to_float(p0['priceupmwh'])
            sel_mask = p0['selectedbyoptimizer'].astype(bool)

            try:
                fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
            except Exception:
                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                fig.subplots_adjust(top=0.9)

            if opt_only_selected:
                ax.scatter(x[sel_mask], y[sel_mask], color='red', edgecolor='black', label='Selected', s=40, zorder=3)
            else:
                ax.scatter(x[~sel_mask], y[~sel_mask], color='yellow', edgecolor='black', label='Not selected', s=40, zorder=2)
                ax.scatter(x[sel_mask], y[sel_mask], color='red', edgecolor='black', label='Selected', s=40, zorder=3)

            ax.set_ylabel('Price Up (€/MWh)')
            ax.set_xlabel("Offered Volume Up (MW)")
            ax.set_title("Offered Upward Volume vs Price Up", fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
            if opt_grid:
                ax.grid(True, linestyle='--', alpha=0.4)

            y_top_scatter = max(1.0, (float(y.max()) if len(y) else 0.0) * 1.05)
            ax.set_ylim(bottom=0, top=y_top_scatter)

            # De-duplicated legend (deterministic)
            h, l = ax.get_legend_handles_labels()
            if l:
                uniq = OrderedDict(zip(l, h))
                ax.legend(uniq.values(), uniq.keys(), fontsize=8)

            st.pyplot(fig)
        else:
            st.info("No data to plot for this combination.")

    # ---------------- aFRR Up Merit Order ----------------
    st.markdown("---")
    st.markdown("## aFRR Up Merit Order by Delivery Period")
    st.write(
        "Cumulative upward offered volumes sorted by price for each 4-hour delivery period. "
        "Selected bids highlighted. Set the max Y (price) below."
    )

    ymax_up_input = st.number_input(
        "Set y-max (€/MWh) for aFRR Up Merit Order plots:",
        min_value=10, max_value=2000, value=st.session_state['ymax_up_input'], step=10, key=f'ymax_up_{date}'
    )
    st.session_state['ymax_up_input'] = ymax_up_input

    # Prepare up merit order plots (use period_4h)
    xmax_up_all = 0.0
    data_up_plot = []
    marginals_up = {}
    for p in PERIODS_4H:
        sub = df[df['period_4h'].astype(object) == p].copy()  # <<<
        sub = sub[['afrrofferedvolumeupmw', 'priceupmwh', 'selectedbyoptimizer']].dropna()
        if sub.empty:
            data_up_plot.append(([], [], [], [], [], [], p))
            marginals_up[p] = 0.0
            continue
        sub = sub.sort_values(by='priceupmwh')
        cum = sub['afrrofferedvolumeupmw'].cumsum()
        prc = sub['priceupmwh']
        sel_mask = sub['selectedbyoptimizer']
        data_up_plot.append((cum, prc, cum[sel_mask], prc[sel_mask], cum[~sel_mask], prc[~sel_mask], p))
        if not cum.empty:
            xmax_up_all = max(xmax_up_all, float(cum.max()))
        marginals_up[p] = float(prc[sel_mask].max()) if sel_mask.any() else 0.0

    try:
        fig1, axs1 = plt.subplots(3, 2, figsize=(9.2, 9.2), constrained_layout=True)
    except Exception:
        fig1, axs1 = plt.subplots(3, 2, figsize=(9.2, 9.2))
        fig1.subplots_adjust(top=0.92)
    axs1 = axs1.flatten()

    for i, (cum, prc, cs, ps, cnc, pnc, period_lbl) in enumerate(data_up_plot):
        ax = axs1[i]
        if len(cum) and opt_show_step:
            ax.step(cum, prc, where='post', color='gray', linewidth=1, zorder=1)
        if opt_show_points:
            if not opt_only_selected and len(cnc):
                ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=18, label='Not selected', zorder=2)
            if len(cs):
                ax.scatter(cs, ps, color='red', edgecolor='black', s=18, label='Selected', zorder=3)
        if opt_show_marginal_line and marginals_up.get(period_lbl, 0) > 0:
            ax.axhline(marginals_up[period_lbl], color='#555', linestyle='--', linewidth=0.9, alpha=0.7, label='Marginal')

        ax.set_title(f"Period {period_lbl}", fontsize=10)
        ax.set_ylabel("Price up (€/MWh)")
        ax.set_xlabel("")
        xlim_max = max(1.0, xmax_up_all) if opt_sync_x_limits else max(1.0, float(cum.max()) if len(cum) else 1.0)
        ax.set(xlim=(0, xlim_max), ylim=(0, ymax_up_input))
        if opt_grid:
            ax.grid(True, linestyle='--', alpha=0.4)

        h, l = ax.get_legend_handles_labels()
        if l:
            uniq = OrderedDict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)

        ax.tick_params(axis='both', labelsize=8)

    for j in range(len(PERIODS_4H), len(axs1)):
        fig1.delaxes(axs1[j])

    fig1.suptitle("aFRR Up Merit Order by Delivery Period", fontsize=14, y=1.02)
    st.pyplot(fig1)
    up_png = _save_fig_png(fig1)

    # ---------------- aFRR Down Merit Order ----------------
    st.markdown("---")
    st.markdown("## aFRR Down Merit Order by Delivery Period")
    st.write(
        "Cumulative downward offered volumes sorted by price for each 4-hour delivery period. "
        "Selected bids highlighted. Set the max Y (price) below."
    )

    ymax_down_input = st.number_input(
        "Set y-max (€/MWh) for aFRR Down Merit Order plots:",
        min_value=10, max_value=2000, value=st.session_state['ymax_down_input'], step=10, key=f'ymax_down_{date}'
    )
    st.session_state['ymax_down_input'] = ymax_down_input

    xmax_down_all = 0.0
    data_down_plot = []
    marginals_down = {}
    for p in PERIODS_4H:
        sub = df[df['period_4h'].astype(object) == p].copy()  # <<<
        sub = sub[['afrrofferedvolumedownmw', 'pricedownmwh', 'selectedbyoptimizer']].dropna()
        if sub.empty:
            data_down_plot.append(([], [], [], [], [], [], p))
            marginals_down[p] = 0.0
            continue
        sub = sub.sort_values(by='pricedownmwh')
        cum = sub['afrrofferedvolumedownmw'].cumsum()
        prc = sub['pricedownmwh']
        sel_mask = sub['selectedbyoptimizer']
        data_down_plot.append((cum, prc, cum[sel_mask], prc[sel_mask], cum[~sel_mask], prc[~sel_mask], p))
        if not cum.empty:
            xmax_down_all = max(xmax_down_all, float(cum.max()))
        marginals_down[p] = float(prc[sel_mask].max()) if sel_mask.any() else 0.0

    try:
        fig2, axs2 = plt.subplots(3, 2, figsize=(9.2, 9.2), constrained_layout=True)
    except Exception:
        fig2, axs2 = plt.subplots(3, 2, figsize=(9.2, 9.2))
        fig2.subplots_adjust(top=0.92)
    axs2 = axs2.flatten()

    for i, (cum, prc, cs, ps, cnc, pnc, period_lbl) in enumerate(data_down_plot):
        ax = axs2[i]
        if len(cum) and opt_show_step:
            ax.step(cum, prc, where='post', color='gray', linewidth=1, zorder=1)
        if opt_show_points:
            if not opt_only_selected and len(cnc):
                ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=18, label='Not selected', zorder=2)
            if len(cs):
                ax.scatter(cs, ps, color='red', edgecolor='black', s=18, label='Selected', zorder=3)
        if opt_show_marginal_line and marginals_down.get(period_lbl, 0) > 0:
            ax.axhline(marginals_down[period_lbl], color='#555', linestyle='--', linewidth=0.9, alpha=0.7, label='Marginal')

        ax.set_title(f"Period {period_lbl}", fontsize=10)
        ax.set_ylabel("Price down (€/MWh)")
        ax.set_xlabel("")
        xlim_max = max(1.0, xmax_down_all) if opt_sync_x_limits else max(1.0, float(cum.max()) if len(cum) else 1.0)
        ax.set(xlim=(0, xlim_max), ylim=(0, ymax_down_input))
        if opt_grid:
            ax.grid(True, linestyle='--', alpha=0.4)

        h, l = ax.get_legend_handles_labels()
        if l:
            uniq = OrderedDict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)

        ax.tick_params(axis='both', labelsize=8)

    for j in range(len(PERIODS_4H), len(axs2)):
        fig2.delaxes(axs2[j])

    fig2.suptitle("aFRR Down Merit Order by Delivery Period", fontsize=14, y=1.02)
    st.pyplot(fig2)
    down_png = _save_fig_png(fig2)

# ----------------------------- Downloads -----------------------------
st.markdown("---")
st.subheader("Downloads")

colA, colB, colC, colD = st.columns(4)

with colA:
    st.download_button(
        label="Download Raw (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"afrr_raw_{date}.csv",
        mime="text/csv"
    )

with colB:
    st.download_button(
        label="Download Summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"afrr_summary_{date}.csv",
        mime="text/csv"
    )

with colC:
    cost_df = pd.DataFrame({
        "Component": ["All-CCTU", "UP (periods)", "DOWN (periods)", "TOTAL"],
        "Cost (€)": [comp_allcctu, comp_up, comp_down, (comp_allcctu + comp_up + comp_down)]
    })
    st.download_button(
        label="Download Costs (CSV)",
        data=cost_df.to_csv(index=False).encode("utf-8"),
        file_name=f"afrr_costs_{date}.csv",
        mime="text/csv"
    )

with colD:
    st.download_button(
        label="Download Bar Chart (PNG)",
        data=bar_png,
        file_name=f"afrr_bars_{date}.png",
        mime="image/png"
    )

colE, colF = st.columns(2)
with colE:
    st.download_button(
        label="Download Merit UP (PNG)",
        data=up_png,
        file_name=f"afrr_merit_up_{date}.png",
        mime="image/png"
    )
with colF:
    st.download_button(
        label="Download Merit DOWN (PNG)",
        data=down_png,
        file_name=f"afrr_merit_down_{date}.png",
        mime="image/png"
    )

# Optional XLSX download if available
try:
    import io
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Raw")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        cost_df.to_excel(writer, index=False, sheet_name="Costs")
    xbuf.seek(0)
    st.download_button(
        label="Download XLSX (Raw+Summary+Costs)",
        data=xbuf,
        file_name=f"afrr_export_{date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception:
    pass

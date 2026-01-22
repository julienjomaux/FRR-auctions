
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="AFRR Bids Explorer", layout="wide")
st.title("AFRR Auction Results Explorer")
date = st.date_input("Select delivery date:", help="Pick the date you want to analyze.")

@st.cache_data(show_spinner=True)
def fetch(dataset, d):
    url = "https://opendata.elia.be/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 10000, "refine.deliverydate": d}
    r = requests.get(url, params=params)
    return pd.DataFrame([rec['fields'] for rec in r.json().get("records", [])])

def to_float(series, default=0.0):
    """Safe numeric conversion to float with fallback default."""
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

if date:
    df = fetch("ods125", str(date))
    if df.empty:
        st.warning("No data found for that date.")
        st.stop()

    # Ensure expected columns exist (fallback to zero if absent)
    for col in [
        'afrrofferedvolumeupmw', 'afrrofferedvolumedownmw',
        'afrrawardedvolumeupmw', 'afrrawardedvolumedownmw',
        'priceupmwh', 'pricedownmwh', 'selectedbyoptimizer',
        'capacitybiddeliveryperiod', 'index'
    ]:
        if col not in df.columns:
            df[col] = 0

    # ---------------- Total Cost Calculation (immediately after date) ----------------
    # All-CCTU 0-24 selected bids component (sum[(Pdown*Vdown_awarded + Pup*Vup_awarded)*24])
    allcctu = df[df['capacitybiddeliveryperiod'].astype(str) == '0 - 24'].copy()
    allcctu['selectedbyoptimizer'] = allcctu['selectedbyoptimizer'].astype(str).str.lower() == "true"
    allcctu_sel = allcctu[allcctu['selectedbyoptimizer']].copy()

    v_up_24 = to_float(allcctu_sel['afrrawardedvolumeupmw'])
    v_dn_24 = to_float(allcctu_sel['afrrawardedvolumedownmw'])
    p_up_24 = to_float(allcctu_sel['priceupmwh'])
    p_dn_24 = to_float(allcctu_sel['pricedownmwh'])

    comp_allcctu = ((p_dn_24 * v_dn_24) + (p_up_24 * v_up_24)).sum() * 24.0  # €/MWh * MW * h = €

    # Build summary by period/direction to reuse for both the table and cost components
    periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
    rows = []
    for p in periods:
        dper = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
        for dir_, vol_col, awd_col, prc_col in [
            ("UP", "afrrofferedvolumeupmw", "afrrawardedvolumeupmw", "priceupmwh"),
            ("DOWN", "afrrofferedvolumedownmw", "afrrawardedvolumedownmw", "pricedownmwh"),
        ]:
            # Convert to numeric safely
            dper[vol_col] = to_float(dper[vol_col])
            dper[awd_col] = to_float(dper[awd_col])
            dper[prc_col] = to_float(dper[prc_col])
            dper['selectedbyoptimizer'] = dper['selectedbyoptimizer'].astype(str).str.lower() == "true"

            tot_vol_sub = dper[vol_col].sum()
            selected = dper[dper['selectedbyoptimizer']]

            tot_awarded = selected[awd_col].sum()
            if tot_awarded > 0:
                # Weighted average by awarded volume
                avg_price = (selected[prc_col] * selected[awd_col]).sum() / tot_awarded
            else:
                avg_price = 0.0
            marginal_price = selected[prc_col].max() if not selected.empty else 0.0

            rows.append([p, dir_, round(tot_awarded, 2), round(avg_price, 2), round(marginal_price, 2), round(tot_vol_sub, 2)])

    summary = pd.DataFrame(rows, columns=[
        "Period", "Direction", "Total Awarded Volume (MW)", "Average Price (€/MWh)",
        "Marginal Price (€/MWh)", "Total Submitted Volume (MW)"
    ])

    # Cost components for UP and DOWN directions across periods:
    # component = sum_over_periods( avg_price_period * total_awarded_volume_period * 4h )
    def comp_direction(direction: str) -> float:
        sub = summary[summary["Direction"] == direction]
        return float((sub["Average Price (€/MWh)"] * sub["Total Awarded Volume (MW)"] * 4.0).sum())

    comp_up = comp_direction("UP")
    comp_down = comp_direction("DOWN")
    total_cost = comp_allcctu + comp_up + comp_down

    # Show cost summary metrics
    st.subheader("Auction Cost Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Auction Cost (€)", f"{total_cost:,.0f}")
    c2.metric("All-CCTU (0–24) Cost (€)", f"{comp_allcctu:,.0f}")
    c3.metric("UP (periods) Cost (€)", f"{comp_up:,.0f}")
    c4.metric("DOWN (periods) Cost (€)", f"{comp_down:,.0f}")

    # ---------------- Layout: Center content: empty–main–empty columns ----------------
    col_left, col_main, col_right = st.columns([0.15, 0.7, 0.15])
    with col_main:

        # ---------------- All-CCTU Table ----------------
        sel = allcctu[allcctu['selectedbyoptimizer']]
        st.markdown("### All-CCTU (0-24): Selected Bids")
        st.markdown("Selected aFRR bids for the All-CCTU (0-24h) period.")
        if not sel.empty:
            # Prepare table with both up/down awarded volumes, rounded to 2 decimals
            t = sel[[
                'index', 'pricedownmwh', 'priceupmwh',
                'afrrofferedvolumedownmw', 'afrrawardedvolumedownmw', 'afrrawardedvolumeupmw'
            ]].copy()

            # Convert numeric columns and round to 2 decimals
            for c in ['pricedownmwh', 'priceupmwh', 'afrrofferedvolumedownmw', 'afrrawardedvolumedownmw', 'afrrawardedvolumeupmw']:
                t[c] = to_float(t[c]).round(2)

            t.columns = [
                "Index", "Price Down (€/MWh)", "Price Up (€/MWh)",
                "Offered Down (MW)", "Awarded Down (MW)", "Awarded Up (MW)"
            ]
            st.table(t)
        else:
            st.info("No All-CCTU selected bids for this date.")

        # ---------------- Summary Table ----------------
        st.markdown("### Period Results Summary")
        st.markdown("This summary table shows total volumes and prices by direction and period.")
        st.dataframe(summary, use_container_width=True)

        # ---------------- Dual Bar Charts (UP vs DOWN) ----------------
        st.markdown("### Bar Charts by Period (UP vs DOWN)")
        st.write(
            "Choose a metric to visualize for the 6 periods. Left chart shows **UP**, right shows **DOWN**. "
            "All y-axes start at 0 and include the maximum values displayed."
        )

        metric_options = {
            "Average Price (€/MWh)": "Average Price (€/MWh)",
            "Marginal Price (€/MWh)": "Marginal Price (€/MWh)",
            "Total Awarded Volume (MW)": "Total Awarded Volume (MW)",
            "Total Submitted Volume (MW)": "Total Submitted Volume (MW)"
        }
        selected_metric = st.selectbox(
            "Select metric for bar charts:",
            list(metric_options.keys()),
            index=0
        )
        metric_col = metric_options[selected_metric]

        # Prepare data for bars
        sum_up = summary[summary["Direction"] == "UP"].set_index("Period").reindex(periods)
        sum_dn = summary[summary["Direction"] == "DOWN"].set_index("Period").reindex(periods)

        y_up = to_float(sum_up[metric_col]).fillna(0.0)
        y_dn = to_float(sum_dn[metric_col]).fillna(0.0)

        # Compute a common y max across both charts so max is included, with a small headroom
        combined_max = float(max(y_up.max(), y_dn.max()))
        if combined_max <= 0:
            y_top = 1.0  # minimal visible top for empty/zero data
        else:
            y_top = combined_max * 1.05  # small margin to avoid clipping the tallest bar

        fig_bar, (ax_up, ax_dn) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # UP bars
        ax_up.bar(periods, y_up, color="#2E86AB", edgecolor="black")
        ax_up.set_title(f"UP – {selected_metric}")
        ax_up.set_ylabel(selected_metric)
        ax_up.set_ylim(bottom=0, top=y_top)  # y starts at 0 and includes max
        ax_up.tick_params(axis='x', rotation=45)
        ax_up.set_xlabel("")  # remove x-axis title

        # DOWN bars
        ax_dn.bar(periods, y_dn, color="#F39C12", edgecolor="black")
        ax_dn.set_title(f"DOWN – {selected_metric}")
        ax_dn.set_ylim(bottom=0, top=y_top)  # y starts at 0 and includes max
        ax_dn.tick_params(axis='x', rotation=45)
        ax_dn.set_xlabel("")  # remove x-axis title

        plt.tight_layout()
        st.pyplot(fig_bar)

        # ---------------- Scatter Plot ----------------
        st.markdown("---")
        st.markdown("## Scatter Plot: Offered UP vs Upward Price (All-CCTU 0-24)")
        st.write(
            "This scatter plot displays the offered upward volumes versus their respective upward prices "
            "for the All-CCTU (0-24h) period. Use the droplist to filter bids by a specific 'Offered Volume Down' value."
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
                "Select the 'Offered Volume Down' for filtering:", vals, index=0)
            mask0 = (to_float(allcctu['afrrofferedvolumedownmw']) == float(selected_value))
            p0 = allcctu[mask0]
            if not p0.empty:
                x = to_float(p0['afrrofferedvolumeupmw'])
                y = to_float(p0['priceupmwh'])
                sel_mask = p0['selectedbyoptimizer']

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(x[~sel_mask], y[~sel_mask], color='yellow', edgecolor='black', label='Not selected', s=40)
                ax.scatter(x[sel_mask], y[sel_mask], color='red', edgecolor='black', label='Selected', s=40)
                # Remove x-axis title; keep y label
                ax.set_ylabel('Price Up (€/MWh)')
                ax.set_xlabel("")
                ax.set_title("Offered Upward Volume vs Price Up", fontsize=12, fontweight='bold')
                ax.tick_params(axis='both', labelsize=8)
                ax.legend(fontsize=8)
                ax.grid(True)
                # y-axis starts at 0 and includes max
                y_top_scatter = max(1.0, (float(y.max()) if len(y) else 0.0) * 1.05)
                ax.set_ylim(bottom=0, top=y_top_scatter)
                st.pyplot(fig)
            else:
                st.info("No data to plot for this combination.")

        # ---------------- aFRR Up Merit Order Section ----------------
        st.markdown("---")
        st.markdown("## AFRR Up Merit Order by Delivery Period")
        st.write(
            "These charts show the cumulative upward offered volumes sorted by price for each 4-hour delivery period. "
            "Selected bids are highlighted. You can set the maximum Y (price) value for all plots below."
        )
        default_ymax_up = 500  # adjust as needed
        ymax_up = st.number_input(
            "Set y-max (€/MWh) for AFRR Up Merit Order plots:",
            min_value=10, max_value=2000, value=default_ymax_up, step=10
        )

        # Prepare up merit order plots for all periods
        xmax = 0.0
        data_up = []
        for p in periods:
            sub = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
            # Valid rows have numeric offered volume and price
            sub['afrrofferedvolumeupmw'] = to_float(sub['afrrofferedvolumeupmw'])
            sub['priceupmwh'] = to_float(sub['priceupmwh'])
            valid = sub['afrrofferedvolumeupmw'].notnull() & sub['priceupmwh'].notnull()
            sub = sub[valid]
            if sub.empty:
                data_up.append(([], [], [], [], [], [], p))
                continue
            sub['selectedbyoptimizer'] = sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
            sub = sub.sort_values(by='priceupmwh')
            cum = sub['afrrofferedvolumeupmw'].cumsum()
            prc = sub['priceupmwh']
            sel_mask = sub['selectedbyoptimizer']
            data_up.append((cum, prc, cum[sel_mask], prc[sel_mask], cum[~sel_mask], prc[~sel_mask], p))
            if not cum.empty:
                xmax = max(xmax, float(cum.max()))

        fig1, axs1 = plt.subplots(3, 2, figsize=(9, 9))
        axs1 = axs1.flatten()
        for i, (cum, prc, cs, ps, cnc, pnc, p) in enumerate(data_up):
            ax = axs1[i]
            if len(cum): ax.step(cum, prc, where='post', color='gray', linewidth=1)
            if len(cnc): ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=20, label='Not selected')
            if len(cs): ax.scatter(cs, ps, color='red', edgecolor='black', s=20, label='Selected')
            ax.set_title(f"Period {p}", fontsize=10)
            # Remove x-axis title; keep y label
            ax.set_ylabel("Price up (€/MWh)")
            ax.set_xlabel("")
            ax.set(
                xlim=(0, max(1.0, xmax)),
                ylim=(0, ymax_up)  # y starts at 0; top controlled by user
            )
            ax.grid(True)
            h, l = ax.get_legend_handles_labels()
            if l: ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
        for j in range(len(periods), len(axs1)):
            fig1.delaxes(axs1[j])
        plt.tight_layout()
        plt.suptitle("AFRR Up Merit Order by Delivery Period", fontsize=14, y=1.05)
        st.pyplot(fig1)

        # ---------------- aFRR Down Merit Order Section ----------------
        st.markdown("---")
        st.markdown("## AFRR Down Merit Order by Delivery Period")
        st.write(
            "These charts show the cumulative downward offered volumes sorted by price for each 4-hour delivery period. "
            "Selected bids are highlighted. You can set the maximum Y (price) value for all plots below."
        )
        default_ymax_down = 500  # adjust as needed
        ymax_down = st.number_input(
            "Set y-max (€/MWh) for AFRR Down Merit Order plots:",
            min_value=10, max_value=2000, value=default_ymax_down, step=10, key='ymax_down'
        )

        # Prepare down merit order plots for all periods
        xmaxd = 0.0
        data_down = []
        for p in periods:
            sub = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
            # Valid rows have numeric offered volume and price
            sub['afrrofferedvolumedownmw'] = to_float(sub['afrrofferedvolumedownmw'])
            sub['pricedownmwh'] = to_float(sub['pricedownmwh'])
            valid = sub['afrrofferedvolumedownmw'].notnull() & sub['pricedownmwh'].notnull()
            sub = sub[valid]
            if sub.empty:
                data_down.append(([], [], [], [], [], [], p))
                continue
            sub['selectedbyoptimizer'] = sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
            sub = sub.sort_values(by='pricedownmwh')
            cum = sub['afrrofferedvolumedownmw'].cumsum()
            prc = sub['pricedownmwh']
            sel_mask = sub['selectedbyoptimizer']
            data_down.append((cum, prc, cum[sel_mask], prc[sel_mask], cum[~sel_mask], prc[~sel_mask], p))
            if not cum.empty:
                xmaxd = max(xmaxd, float(cum.max()))

        fig2, axs2 = plt.subplots(3, 2, figsize=(9, 9))
        axs2 = axs2.flatten()
        for i, (cum, prc, cs, ps, cnc, pnc, p) in enumerate(data_down):
            ax = axs2[i]
            if len(cum): ax.step(cum, prc, where='post', color='gray', linewidth=1)
            if len(cnc): ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=20, label='Not selected')
            if len(cs): ax.scatter(cs, ps, color='red', edgecolor='black', s=20, label='Selected')
            ax.set_title(f"Period {p}", fontsize=10)
            # Remove x-axis title; keep y label
            ax.set_ylabel("Price down (€/MWh)")
            ax.set_xlabel("")
            ax.set(
                xlim=(0, max(1.0, xmaxd)),
                ylim=(0, ymax_down)  # y starts at 0; top controlled by user
            )
            ax.grid(True)
            h, l = ax.get_legend_handles_labels()
            if l: ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
        for j in range(len(periods), len(axs2)):
            fig2.delaxes(axs2[j])
        plt.tight_layout()
        plt.suptitle("AFRR Down Merit Order by Delivery Period", fontsize=14, y=1.05)
        st.pyplot(fig2)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import io
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Performance Analytics Pro", layout="wide")

def force_numeric_series(series):
    if series.dtype == 'object':
        def time_to_minutes(val):
            if isinstance(val, datetime.time):
                return val.hour * 60 + val.minute + val.second / 60
            if isinstance(val, str) and ':' in val:
                try:
                    parts = val.split(':')
                    if len(parts) == 3: return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                    elif len(parts) == 2: return int(parts[0]) + int(parts[1]) / 60
                except: return 0
            return val
        series = series.apply(time_to_minutes)
    return pd.to_numeric(series, errors='coerce').fillna(0)

st.title("⚽ Sport Science Analytics Hub")

# --- 1. DATA LOADING ---
uploaded_file = st.sidebar.file_uploader("Upload XLSX", type=["xlsx"])

if uploaded_file:
    xl = pd.ExcelFile(uploaded_file)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    
    # 1. PRIMARY FILTER: X-Axis Choice
    x_col = st.sidebar.selectbox("Display on X-Axis:", ['Sheet_Segment', 'Match_Label'])

    # 2. DYNAMIC SHEET SELECTION
    if x_col == 'Sheet_Segment':
        selected_sheets = st.sidebar.multiselect(
            "Select Sheets (X-Axis Segments):", 
            xl.sheet_names, 
            default=[xl.sheet_names[0]]
        )
    else:
        selected_sheets = [st.sidebar.selectbox("Select Data Sheet:", xl.sheet_names)]

    if not selected_sheets:
        st.warning("Please select at least one sheet.")
        st.stop()

    # LOAD AND PROCESS DATA
    all_data_list = []
    for sheet in selected_sheets:
        df = pd.read_excel(uploaded_file, sheet_name=sheet).copy()
        df.columns = [str(col).strip() for col in df.columns]
        df['Sheet_Segment'] = sheet
        
        # Clean naming columns to prevent duplicates due to spaces
        if 'adversaire' in df.columns:
            df['adversaire'] = df['adversaire'].astype(str).str.strip()
        if 'Journnée' in df.columns:
            df['Journnée'] = df['Journnée'].astype(str).str.strip()
            
        for col in df.columns:
            if col not in ['Name', 'adversaire', 'Journnée', 'compétition', 'Sheet_Segment']:
                df[col] = force_numeric_series(df[col])
        all_data_list.append(df)

    # Merge all selected sheets
    merged_df = pd.concat(all_data_list, ignore_index=True)

    # CALCULATE METRICS
    calc_dict = {}
    
    hit_src = ['Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)']
    if all(c in merged_df.columns for c in hit_src):
        calc_dict['HIT'] = merged_df[hit_src].sum(axis=1)
    
    acc_cols = ['Acceleration B1 Efforts (Gen 2)', 'Acceleration B2 Efforts (Gen 2)', 'Acceleration B3 Efforts (Gen 2)']
    if all(c in merged_df.columns for c in acc_cols):
        calc_dict['Total Acceleration'] = merged_df[acc_cols].sum(axis=1)

    dec_cols = ['decceleration B1 Efforts (Gen 2)', 'decceleration B2 Efforts (Gen 2)', 'decceleration B3 Efforts (Gen 2)']
    if all(c in merged_df.columns for c in dec_cols):
        calc_dict['Total Decceleration'] = merged_df[dec_cols].sum(axis=1)

    # REFINED MATCH LABEL: Strictly distinct by stripping inputs
    j_str = merged_df['Journnée'].astype(str).str.strip()
    adv_str = merged_df['adversaire'].fillna('').astype(str).str.strip()
    calc_dict['Match_Label'] = "J" + j_str + "\n" + adv_str
    
    merged_df = merged_df.assign(**calc_dict).copy()

    # 3. FILTER BY MATCHES (Ensuring distinct values)
    st.sidebar.markdown("---")
    # Using set/unique to ensure even if sheets overlap, the labels are distinct in the filter
    distinct_matches = sorted(list(merged_df['Match_Label'].unique()))
    
    selected_filter_matches = st.sidebar.multiselect(
        "Filter by Matches:", 
        distinct_matches, 
        default=distinct_matches
    )
    
    # Filter original DF
    filtered_df = merged_df[merged_df['Match_Label'].isin(selected_filter_matches)]

    # 4. SELECT PLAYERS
    distinct_players = sorted(filtered_df['Name'].dropna().unique())
    selected_players = st.sidebar.multiselect("Select Players:", distinct_players)
    
    # 5. SELECT METRICS
    metrics_list = ['Maximum Velocity (km/h)', 'HIT', 'Total Player Load', 'Meterage Per Minute', 
                    'Total Acceleration', 'Total Decceleration', 'Total Distance (m)', 'Sprint']
    available_metrics = [m for m in metrics_list if m in filtered_df.columns]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", available_metrics, default=available_metrics)
    
    chart_color = st.sidebar.color_picker("Line Color", "#0077b6")
    show_labels = st.sidebar.checkbox("Show Values on Chart", value=True)

    # --- 3. MAIN AREA: RENDERING ---
    if st.sidebar.button("🚀 GENERATE ALL CHARTS", type="primary"):
        if not selected_players or not selected_metrics:
            st.error("Please select at least one Player and one Metric.")
        else:
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                for player in selected_players:
                    st.header(f"👤 Player: {player}")
                    
                    p_data = filtered_df[filtered_df['Name'] == player].copy()
                    
                    if x_col == 'Sheet_Segment':
                        # AVERAGE player performance per segment across all filtered matches
                        display_data = p_data.groupby('Sheet_Segment', sort=False)[selected_metrics].mean().reset_index()
                        # Sort based on the selection order in sidebar
                        display_data['Sheet_Segment'] = pd.Categorical(display_data['Sheet_Segment'], categories=selected_sheets, ordered=True)
                        display_data = display_data.sort_values('Sheet_Segment')
                    else:
                        # PROGRESSION across matches
                        # Sort by original Journee index if possible, otherwise by string
                        display_data = p_data.sort_values(by='Match_Label').reset_index(drop=True)
                    
                    if display_data.empty:
                        st.info(f"No data for {player} in selection.")
                        continue

                    # Charting
                    fig, axes = plt.subplots(len(selected_metrics), 1, figsize=(15, 6 * len(selected_metrics)))
                    if len(selected_metrics) == 1: axes = [axes]
                    fig.suptitle(f"Performance Analysis: {player}", fontsize=22, fontweight='bold', y=1.0)

                    for i, metric in enumerate(selected_metrics):
                        ax = axes[i]
                        y = display_data[metric]
                        x = display_data[x_col].astype(str)
                        
                        y_mean, y_max, y_min = y.mean(), y.max(), y.min()

                        ax.plot(x, y, color=chart_color, linewidth=2.5, marker='o')
                        ax.axhline(y_mean, color='gray', linestyle='--', alpha=0.6)
                        
                        # Markers for Max/Min
                        for idx, val in enumerate(y):
                            if val == y_max: ax.scatter(x[idx], val, color='green', s=150, zorder=5, edgecolors='black')
                            elif val == y_min: ax.scatter(x[idx], val, color='red', s=150, zorder=5, edgecolors='black')

                        if show_labels:
                            for xi, yi in zip(x, y):
                                ax.text(xi, yi, f'{yi:.1f}', fontweight='bold', ha='center', va='bottom', fontsize=10)

                        # External Legend
                        handles = [
                            Line2D([0], [0], color=chart_color, lw=2, label='Performance Trend'),
                            Line2D([0], [0], color='gray', ls='--', label=f'Avg: {y_mean:.1f}'),
                            Line2D([0], [0], marker='o', color='w', mfc='green', label=f'Max: {y_max:.1f}', mec='k'),
                            Line2D([0], [0], marker='o', color='w', mfc='red', label=f'Min: {y_min:.1f}', mec='k')
                        ]
                        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), title="Legend")
                        
                        title_text = f"AVG {metric}" if x_col == 'Sheet_Segment' else metric
                        ax.set_title(title_text.upper(), loc='left', fontweight='bold', fontsize=14)
                        ax.grid(True, axis='y', linestyle=':', alpha=0.5)

                    plt.subplots_adjust(right=0.8, hspace=0.4, top=0.92)
                    st.pyplot(fig)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            st.sidebar.download_button("📥 Download PDF Report", pdf_buffer.getvalue(), "Report.pdf", use_container_width=True)
else:
    st.info("Please upload an Excel file to start.")
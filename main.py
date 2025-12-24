import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import io
import datetime
import textwrap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Performance Analytics Pro", layout="wide")

# Optimized Numeric Conversion logic
def fast_to_minutes(val):
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, datetime.time):
        return val.hour * 60 + val.minute + val.second / 60
    if isinstance(val, str):
        if ':' in val:
            parts = val.split(':')
            try:
                if len(parts) == 3: # HH:MM:SS
                    return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                elif len(parts) == 2: # MM:SS
                    return int(parts[0]) + int(parts[1]) / 60
            except:
                return 0
    return 0

# --- CORE OPTIMIZATION: CACHED DATA PROCESSING ---
@st.cache_data
def load_and_preprocess(file_bytes, sheets_to_load):
    """Reads Excel and processes metrics once. Result is cached in memory."""
    all_data = []
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xl:
        for sheet in sheets_to_load:
            df = xl.parse(sheet)
            df.columns = [str(col).strip() for col in df.columns]
            df['Sheet_Segment'] = sheet
            
            target_cols = [
                'Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)',
                'Acceleration B1 Efforts (Gen 2)', 'Acceleration B2 Efforts (Gen 2)', 'Acceleration B3 Efforts (Gen 2)',
                'decceleration B1 Efforts (Gen 2)', 'decceleration B2 Efforts (Gen 2)', 'decceleration B3 Efforts (Gen 2)',
                'Maximum Velocity (km/h)', 'Total Player Load', 'Meterage Per Minute', 'Total Distance (m)', 'Sprint'
            ]
            
            if 'adversaire' in df.columns: df['adversaire'] = df['adversaire'].astype(str).str.strip()
            if 'Journnée' in df.columns: df['Journnée'] = df['Journnée'].astype(str).str.strip()
            
            for col in [c for c in target_cols if c in df.columns]:
                df[col] = df[col].apply(fast_to_minutes)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            all_data.append(df)

    if not all_data: return pd.DataFrame()
    merged_df = pd.concat(all_data, ignore_index=True)
    
    calc_dict = {}
    current_cols = merged_df.columns

    # HIT Calculation
    hit_src = ['Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)']
    if all(c in current_cols for c in hit_src):
        calc_dict['HIT'] = merged_df[hit_src].sum(axis=1)
    
    # Accel/Decel Calculation
    acc_cols = ['Acceleration B1 Efforts (Gen 2)', 'Acceleration B2 Efforts (Gen 2)', 'Acceleration B3 Efforts (Gen 2)']
    if all(c in current_cols for c in acc_cols):
        calc_dict['Total Acceleration'] = merged_df[acc_cols].sum(axis=1)

    dec_cols = ['decceleration B1 Efforts (Gen 2)', 'decceleration B2 Efforts (Gen 2)', 'decceleration B3 Efforts (Gen 2)']
    if all(c in current_cols for c in dec_cols):
        calc_dict['Total Decceleration'] = merged_df[dec_cols].sum(axis=1)

    # Label Formatting
    j_str = merged_df['Journnée'].astype(str).str.strip()
    adv_str = merged_df['adversaire'].fillna('').astype(str).str.strip()
    calc_dict['Match_Label'] = "J" + j_str + "\n" + adv_str
    
    return merged_df.assign(**calc_dict)

st.title("⚽ Sport Science Analytics Hub")

# --- 1. DATA LOADING ---
uploaded_file = st.sidebar.file_uploader("Upload XLSX", type=["xlsx"])

if uploaded_file:
    xl_summary = pd.ExcelFile(uploaded_file)
    sheet_names = xl_summary.sheet_names
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    
    x_col = st.sidebar.selectbox("Display on X-Axis:", ['Sheet_Segment', 'Match_Label'])

    # --- UPDATED SELECTION LOGIC ---
    if x_col == 'Sheet_Segment':
        # MULTI CHOICE for Sheet Segment mode
        selected_sheets = st.sidebar.multiselect("Select Sheets (X-Axis Segments):", sheet_names, default=[sheet_names[0]])
    else:
        # SINGLE CHOICE for Match Label mode
        single_sheet = st.sidebar.selectbox("Select Data Source (Sheet):", sheet_names)
        selected_sheets = [single_sheet] # Wrap in list for processing

    if not selected_sheets:
        st.warning("Please select a data source.")
        st.stop()

    merged_df = load_and_preprocess(uploaded_file.getvalue(), tuple(selected_sheets))

    # --- 2. FILTERS ---
    st.sidebar.markdown("---")
    distinct_matches = sorted(list(merged_df['Match_Label'].unique()))
    selected_filter_matches = st.sidebar.multiselect("Filter by Matches:", distinct_matches, default=distinct_matches)
    filtered_df = merged_df[merged_df['Match_Label'].isin(selected_filter_matches)]

    distinct_players = sorted(filtered_df['Name'].dropna().unique())
    selected_players = st.sidebar.multiselect("Select Players:", distinct_players)
    
    metrics_list = ['Maximum Velocity (km/h)', 'HIT', 'Total Player Load', 'Meterage Per Minute', 
                    'Total Acceleration', 'Total Decceleration', 'Total Distance (m)', 'Sprint']
    available_metrics = [m for m in metrics_list if m in filtered_df.columns]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", available_metrics, default=available_metrics)
    
    chart_color = st.sidebar.color_picker("Line Color", "#0077b6")
    show_labels = st.sidebar.checkbox("Show Values on Chart", value=True)

    # --- 3. RENDERING ---
    if st.sidebar.button("🚀 GENERATE ALL CHARTS", type="primary"):
        if not selected_players or not selected_metrics:
            st.error("Please select at least one Player and one Metric.")
        else:
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                # Create labels based on mode
                segments_label = "Segments" if x_col == 'Sheet_Segment' else "Sheet Source"
                segments_str = ", ".join(selected_sheets)

                for player in selected_players:
                    st.header(f"👤 Player: {player}")
                    st.caption(f"{segments_label}: {segments_str}")
                    
                    p_data = filtered_df[filtered_df['Name'] == player].copy()
                    
                    if x_col == 'Sheet_Segment':
                        display_data = p_data.groupby('Sheet_Segment', sort=False)[selected_metrics].mean().reset_index()
                        display_data['Sheet_Segment'] = pd.Categorical(display_data['Sheet_Segment'], categories=selected_sheets, ordered=True)
                        display_data = display_data.sort_values('Sheet_Segment')
                    else:
                        display_data = p_data.sort_values(by='Match_Label').reset_index(drop=True)
                    
                    if display_data.empty:
                        st.info(f"No data found for {player}")
                        continue

                    # Charting
                    fig, axes = plt.subplots(len(selected_metrics), 1, figsize=(14, 5 * len(selected_metrics)))
                    if len(selected_metrics) == 1: axes = [axes]
                    
                    # Title including Player and Selection
                    main_title = f"Performance Analysis: {player}\nPeriod: {segments_str}"
                    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.99)

                    for i, metric in enumerate(selected_metrics):
                        ax = axes[i]
                        y = display_data[metric]
                        raw_x = display_data[x_col].astype(str)
                        wrapped_x = [textwrap.fill(lx, width=12) for lx in raw_x]
                        
                        y_mean, y_max, y_min = y.mean(), y.max(), y.min()

                        ax.plot(wrapped_x, y, color=chart_color, linewidth=2.5, marker='o', markersize=6, alpha=0.8)
                        ax.axhline(y_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
                        
                        for idx, val in enumerate(y):
                            if val == y_max: ax.scatter(wrapped_x[idx], val, color='#2ecc71', s=120, zorder=5, edgecolors='black')
                            elif val == y_min: ax.scatter(wrapped_x[idx], val, color='#e74c3c', s=120, zorder=5, edgecolors='black')

                        if show_labels:
                            for xi, yi in zip(wrapped_x, y):
                                ax.text(xi, yi, f'{yi:.1f}', fontweight='bold', ha='center', va='bottom', fontsize=9, color='#333')

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
                        ax.set_facecolor('#fafafa')
                        
                        handles = [
                            Line2D([0], [0], color=chart_color, lw=2, marker='o', label='Trend'),
                            Line2D([0], [0], color='gray', ls='--', label=f'Avg: {y_mean:.1f}'),
                            Line2D([0], [0], marker='o', color='w', mfc='#2ecc71', label=f'Max: {y_max:.1f}', mec='k'),
                            Line2D([0], [0], marker='o', color='w', mfc='#e74c3c', label=f'Min: {y_min:.1f}', mec='k')
                        ]
                        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
                        
                        title_label = f"AVG {metric}" if x_col == 'Sheet_Segment' else metric
                        ax.set_title(title_label.upper(), loc='left', fontweight='bold', fontsize=12, color='#444')

                    plt.subplots_adjust(right=0.82, hspace=0.35, top=0.93)
                    st.pyplot(fig)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            st.sidebar.download_button("📥 Download PDF Report", pdf_buffer.getvalue(), "Report.pdf", use_container_width=True)
else:
    st.info("Please upload an Excel file to start.")
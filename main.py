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

# --- CORE OPTIMIZATION: LOAD & REFINE ONCE ---
@st.cache_data(show_spinner="Refining and loading data...")
def load_all_and_refine(file_bytes):
    """
    Reads all sheets once, filters for only target columns immediately, 
    and caches the result to make filtering instant.
    """
    target_cols = [
        'Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)',
        'Acceleration B1 Efforts (Gen 2)', 'Acceleration B2 Efforts (Gen 2)', 'Acceleration B3 Efforts (Gen 2)',
        'decceleration B1 Efforts (Gen 2)', 'decceleration B2 Efforts (Gen 2)', 'decceleration B3 Efforts (Gen 2)',
        'Maximum Velocity (km/h)', 'Total Player Load', 'Meterage Per Minute', 'Total Distance (m)', 'Sprint'
    ]
    id_cols = ['Name', 'Journnée', 'adversaire']
    
    all_data = []
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xl:
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df.columns = [str(col).strip() for col in df.columns]
            
            # REFINE: Keep only necessary columns that exist
            existing_targets = [c for c in target_cols if c in df.columns]
            existing_ids = [c for c in id_cols if c in df.columns]
            df = df[existing_ids + existing_targets].copy()
            
            df['Sheet_Segment'] = sheet
            
            # Clean text columns
            if 'adversaire' in df.columns: df['adversaire'] = df['adversaire'].fillna('').astype(str).str.strip()
            if 'Journnée' in df.columns: df['Journnée'] = df['Journnée'].fillna('').astype(str).str.strip()
            if 'Name' in df.columns: df['Name'] = df['Name'].fillna('Unknown')

            # Numeric Conversion
            for col in existing_targets:
                df[col] = df[col].apply(fast_to_minutes)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            all_data.append(df)

    if not all_data: return pd.DataFrame()
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Pre-calculate Metrics
    if all(c in merged_df.columns for c in ['Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)']):
        merged_df['HIT'] = merged_df[['Velocity Band 3 Total Distance (m)', 'Velocity Band 4 Total Distance (m)', 'Velocity Band 5 Total Distance (m)']].sum(axis=1)
    
    acc_cols = [c for c in ['Acceleration B1 Efforts (Gen 2)', 'Acceleration B2 Efforts (Gen 2)', 'Acceleration B3 Efforts (Gen 2)'] if c in merged_df.columns]
    if acc_cols: merged_df['Total Acceleration'] = merged_df[acc_cols].sum(axis=1)

    dec_cols = [c for c in ['decceleration B1 Efforts (Gen 2)', 'decceleration B2 Efforts (Gen 2)', 'decceleration B3 Efforts (Gen 2)'] if c in merged_df.columns]
    if dec_cols: merged_df['Total Decceleration'] = merged_df[dec_cols].sum(axis=1)

    merged_df['Match_Label'] = "J" + merged_df['Journnée'].astype(str) + "\n" + merged_df['adversaire'].astype(str)
    
    return merged_df

st.title("⚽ Sport Science Analytics Hub")

# --- 1. DATA LOADING ---
uploaded_file = st.sidebar.file_uploader("Upload XLSX", type=["xlsx"])

if uploaded_file:
    # This happens once and is cached
    full_df = load_all_and_refine(uploaded_file.getvalue())
    sheet_names = list(full_df['Sheet_Segment'].unique())
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    
    x_col = st.sidebar.selectbox("Display on X-Axis:", ['Sheet_Segment', 'Match_Label'])

    if x_col == 'Sheet_Segment':
        selected_sheets = st.sidebar.multiselect("Select Sheets:", sheet_names, default=sheet_names)
        working_df = full_df[full_df['Sheet_Segment'].isin(selected_sheets)]
    else:
        single_sheet = st.sidebar.selectbox("Select Data Source:", sheet_names)
        working_df = full_df[full_df['Sheet_Segment'] == single_sheet]

    # --- 2. FILTERS ---
    st.sidebar.markdown("---")
    distinct_matches = sorted(list(working_df['Match_Label'].unique()))
    selected_filter_matches = st.sidebar.multiselect("Filter by Matches:", distinct_matches, default=distinct_matches)
    filtered_df = working_df[working_df['Match_Label'].isin(selected_filter_matches)]

    distinct_players = sorted(filtered_df['Name'].unique())
    selected_players = st.sidebar.multiselect("Select Players:", distinct_players)
    
    metrics_list = ['Maximum Velocity (km/h)', 'HIT', 'Total Player Load', 'Meterage Per Minute', 
                    'Total Acceleration', 'Total Decceleration', 'Total Distance (m)', 'Sprint']
    available_metrics = [m for m in metrics_list if m in filtered_df.columns]
    selected_metrics = st.sidebar.multiselect("Select Metrics:", available_metrics, default=available_metrics[:4])
    
    chart_color = st.sidebar.color_picker("Line Color", "#0077b6")
    show_labels = st.sidebar.checkbox("Show Values on Chart", value=True)

    # --- 3. RENDERING ---
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
                        display_data = p_data.groupby('Sheet_Segment', sort=False)[selected_metrics].mean().reset_index()
                        display_data['Sheet_Segment'] = pd.Categorical(display_data['Sheet_Segment'], categories=selected_sheets, ordered=True)
                        display_data = display_data.sort_values('Sheet_Segment')
                        main_title = f"Performance Analysis: {player}"
                    else:
                        display_data = p_data.sort_values(by='Match_Label').reset_index(drop=True)
                        main_title = f"Performance Analysis: {player}\nSheet: {single_sheet}"
                    
                    if display_data.empty:
                        st.info(f"No data found for {player}")
                        continue

                    fig, axes = plt.subplots(len(selected_metrics), 1, figsize=(14, 5 * len(selected_metrics)))
                    if len(selected_metrics) == 1: axes = [axes]
                    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.99)

                    for i, metric in enumerate(selected_metrics):
                        ax = axes[i]
                        y = display_data[metric]
                        raw_x = display_data[x_col].astype(str)
                        wrapped_x = [textwrap.fill(lx, width=12) for lx in raw_x]
                        
                        y_mean, y_max, y_min = y.mean(), y.max(), y.min()

                        # Plot line
                        ax.plot(wrapped_x, y, color=chart_color, linewidth=2.5, marker='o', markersize=6, alpha=0.8)
                        ax.axhline(y_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
                        
                        # Highlighting Max/Min
                        for idx, val in enumerate(y):
                            if val == y_max: ax.scatter(wrapped_x[idx], val, color='#2ecc71', s=120, zorder=5, edgecolors='black')
                            elif val == y_min: ax.scatter(wrapped_x[idx], val, color='#e74c3c', s=120, zorder=5, edgecolors='black')

                        if show_labels:
                            for xi, yi in zip(wrapped_x, y):
                                ax.text(xi, yi, f'{yi:.1f}', fontweight='bold', ha='center', va='bottom', fontsize=9, color='#333')

                        # Styling
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
                        ax.set_facecolor('#fafafa')
                        
                        # RESTORED LEGEND
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
#!/usr/bin/env python3
"""
Interactive Factor Explorer for Community Notes

Run with: streamlit run explore_factors.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
import hdbscan

# Page config
st.set_page_config(
    page_title="CN Factor Explorer",
    page_icon="📊",
    layout="wide"
)


def detect_factor_columns(df):
    """Detect factor columns dynamically from the dataframe."""
    factor_cols = []
    for col in df.columns:
        # Only include coreNoteFactor1 and internalNoteFactor columns
        if col == 'coreNoteFactor1' or col.startswith('internalNoteFactor'):
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                factor_cols.append(col)
    # Sort to ensure consistent ordering
    factor_cols = sorted(factor_cols)
    return factor_cols


@st.cache_data(show_spinner=False)
def load_data(_version=3):
    """Load and merge scored notes with content. Returns (df, factor_cols)."""
    base_path = '/Users/riad/Documents/GitHub/communitynotes/scoring/src/data'

    scored = pd.read_csv(base_path + '/scored_notes.tsv', sep='\t', low_memory=False)
    notes = pd.read_csv(base_path + '/notes-00000.tsv', sep='\t', low_memory=False)

    df = scored.merge(notes[['noteId', 'summary']], on='noteId', how='left')

    # Detect factor columns dynamically
    factor_cols = detect_factor_columns(df)

    # Filter to notes with all factors
    df = df.dropna(subset=factor_cols)

    # Add length column
    df['length'] = df['summary'].fillna('').str.len()

    # Detect language
    def detect_lang(text):
        if not isinstance(text, str):
            return 'Unknown'
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text):
            return 'CJK'
        if any(w in text.lower() for w in [' pour ', ' dans ', ' cette ', ' avec ']):
            return 'French'
        if any(w in text.lower() for w in [' para ', ' está ', ' como ', ' pero ']):
            return 'Spanish'
        return 'English'

    df['language'] = df['summary'].apply(detect_lang)

    return df, factor_cols

def main():
    st.title("🔍 Community Notes Factor Explorer")

    # Load data
    with st.spinner("Loading data..."):
        df, factor_cols = load_data()

    st.markdown(f"Explore the {len(factor_cols)}-dimensional embedding space of Community Notes")

    st.sidebar.header("🎛️ Controls")

    # --- Filters ---
    st.sidebar.subheader("Filters")

    # Status filter
    statuses = ['All'] + list(df['finalRatingStatus'].unique())
    selected_status = st.sidebar.selectbox("Note Status", statuses)

    # Language filter
    languages = ['All'] + list(df['language'].unique())
    selected_lang = st.sidebar.selectbox("Language", languages)

    # Highlight specific notes
    st.sidebar.subheader("Highlight Notes")
    highlight_input = st.sidebar.text_area(
        "Note IDs to highlight",
        help="Paste noteIds (one per line, comma-separated, or space-separated)",
        height=100
    )

    # Parse highlight IDs
    highlight_ids = set()
    if highlight_input.strip():
        # Split by newlines, commas, or spaces
        import re as re_module
        parts = re_module.split(r'[\n,\s]+', highlight_input.strip())
        for part in parts:
            part = part.strip()
            if part:
                try:
                    highlight_ids.add(int(part))
                except ValueError:
                    pass  # Skip invalid IDs

    if highlight_ids:
        st.sidebar.success(f"Highlighting {len(highlight_ids)} notes")
        # Debug: show parsed IDs and check if they exist in data
        st.sidebar.caption(f"Parsed IDs: {list(highlight_ids)[:5]}...")
        found_count = df['noteId'].isin(highlight_ids).sum()
        st.sidebar.caption(f"Found in data: {found_count} | noteId dtype: {df['noteId'].dtype}")

    # Extreme filter
    st.sidebar.subheader("Extreme Values")
    extreme_factor = st.sidebar.selectbox(
        "Filter by extreme values of:",
        ['None'] + factor_cols
    )

    extreme_percentile = st.sidebar.slider(
        "Percentile threshold",
        min_value=1, max_value=25, value=10,
        help="Show only top/bottom X% of values"
    )

    extreme_direction = st.sidebar.radio(
        "Direction",
        ['Both', 'Low only', 'High only'],
        horizontal=True
    )

    # HDBSCAN Clustering
    st.sidebar.subheader("HDBSCAN Clustering")
    enable_clustering = st.sidebar.checkbox("Enable clustering", value=False)

    cluster_factors = []
    min_cluster_size = 100
    min_samples = 10

    if enable_clustering:
        cluster_factors = st.sidebar.multiselect(
            "Cluster on factors",
            factor_cols,
            default=factor_cols[:2] if len(factor_cols) >= 2 else factor_cols
        )
        min_cluster_size = st.sidebar.slider(
            "Min cluster size",
            min_value=10, max_value=1000, value=100,
            help="Minimum number of points to form a cluster"
        )
        min_samples = st.sidebar.slider(
            "Min samples",
            min_value=1, max_value=100, value=10,
            help="Points in neighborhood for core points"
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['finalRatingStatus'] == selected_status]

    if selected_lang != 'All':
        filtered_df = filtered_df[filtered_df['language'] == selected_lang]

    if extreme_factor != 'None':
        low_thresh = filtered_df[extreme_factor].quantile(extreme_percentile / 100)
        high_thresh = filtered_df[extreme_factor].quantile(1 - extreme_percentile / 100)

        if extreme_direction == 'Low only':
            filtered_df = filtered_df[filtered_df[extreme_factor] <= low_thresh]
        elif extreme_direction == 'High only':
            filtered_df = filtered_df[filtered_df[extreme_factor] >= high_thresh]
        else:  # Both
            filtered_df = filtered_df[
                (filtered_df[extreme_factor] <= low_thresh) |
                (filtered_df[extreme_factor] >= high_thresh)
            ]

    # Run HDBSCAN clustering if enabled
    if enable_clustering and len(cluster_factors) >= 2:
        cluster_data = filtered_df[cluster_factors].values
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True
        )
        filtered_df = filtered_df.copy()
        filtered_df['cluster'] = clusterer.fit_predict(cluster_data)
        n_clusters = len(set(filtered_df['cluster'])) - (1 if -1 in filtered_df['cluster'].values else 0)
        n_noise = (filtered_df['cluster'] == -1).sum()
        st.sidebar.success(f"Found {n_clusters} clusters, {n_noise:,} noise points")

        # Cluster filter
        cluster_ids = sorted([c for c in filtered_df['cluster'].unique() if c >= 0])
        selected_cluster = st.sidebar.selectbox(
            "Filter to cluster",
            ['All'] + cluster_ids,
            format_func=lambda x: f"Cluster {x}" if x != 'All' else 'All'
        )
        if selected_cluster != 'All':
            filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]
    else:
        filtered_df['cluster'] = -1

    # Sample for performance
    st.sidebar.subheader("Sampling")
    max_points = st.sidebar.slider("Max points to display", 10, 50000, 10000)

    sample_strategy = st.sidebar.radio(
        "Sampling strategy",
        ['Random', 'Most extreme (from origin)', 'Most extreme (from centroid)'],
        help="How to select points when there are more than max"
    )

    if len(filtered_df) > max_points:
        # Always include highlighted notes first
        highlighted_in_filtered = filtered_df[filtered_df['noteId'].isin(highlight_ids)]
        non_highlighted = filtered_df[~filtered_df['noteId'].isin(highlight_ids)]
        n_highlighted = len(highlighted_in_filtered)
        remaining_slots = max(0, max_points - n_highlighted)

        if sample_strategy == 'Random':
            if remaining_slots > 0 and len(non_highlighted) > 0:
                sampled_non_highlighted = non_highlighted.sample(min(remaining_slots, len(non_highlighted)), random_state=42)
            else:
                sampled_non_highlighted = non_highlighted.iloc[:0]  # empty
            plot_df = pd.concat([highlighted_in_filtered, sampled_non_highlighted])
        elif sample_strategy == 'Most extreme (from origin)':
            # Calculate distance from origin using all factors
            non_highlighted = non_highlighted.copy()
            non_highlighted['_distance'] = np.sqrt(
                sum(non_highlighted[col]**2 for col in factor_cols)
            )
            sampled_non_highlighted = non_highlighted.nlargest(remaining_slots, '_distance')
            sampled_non_highlighted = sampled_non_highlighted.drop(columns=['_distance'])
            plot_df = pd.concat([highlighted_in_filtered, sampled_non_highlighted])
        else:  # Most extreme from centroid
            # Calculate distance from centroid
            non_highlighted = non_highlighted.copy()
            centroid = filtered_df[factor_cols].mean()
            non_highlighted['_distance'] = np.sqrt(
                sum((non_highlighted[col] - centroid[col])**2 for col in factor_cols)
            )
            sampled_non_highlighted = non_highlighted.nlargest(remaining_slots, '_distance')
            sampled_non_highlighted = sampled_non_highlighted.drop(columns=['_distance'])
            plot_df = pd.concat([highlighted_in_filtered, sampled_non_highlighted])

        info_msg = f"Showing {len(plot_df):,} of {len(filtered_df):,} notes ({sample_strategy})"
        if n_highlighted > 0:
            info_msg += f" + {n_highlighted} highlighted"
        st.sidebar.info(info_msg)
    else:
        plot_df = filtered_df

    # Add highlighted column
    plot_df = plot_df.copy()
    plot_df['highlighted'] = plot_df['noteId'].isin(highlight_ids)
    n_highlighted_visible = plot_df['highlighted'].sum()
    if highlight_ids:
        if n_highlighted_visible > 0:
            st.sidebar.info(f"✅ {n_highlighted_visible} of {len(highlight_ids)} highlighted notes visible in plot")
        else:
            st.sidebar.warning(f"⚠️ 0 of {len(highlight_ids)} highlighted notes visible (not in filtered data or sample)")

    st.sidebar.markdown(f"**{len(filtered_df):,}** notes match filters")

    # --- Visualization Mode ---
    st.sidebar.subheader("Visualization")
    viz_mode = st.sidebar.radio(
        "Mode",
        ['1D Distribution', '2D Scatter', '3D Scatter', 'Data Table']
    )

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        if viz_mode == '1D Distribution':
            factor_1d = st.selectbox(
                "Select Factor",
                factor_cols
            )

            fig = px.histogram(
                plot_df,
                x=factor_1d,
                color='finalRatingStatus',
                nbins=50,
                title=f"Distribution of {factor_1d}",
                color_discrete_map={
                    'CURRENTLY_RATED_HELPFUL': 'green',
                    'CURRENTLY_RATED_NOT_HELPFUL': 'red',
                    'NEEDS_MORE_RATINGS': 'gray'
                }
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            st.subheader("Statistics")
            stats = plot_df.groupby('finalRatingStatus')[factor_1d].agg(['mean', 'std', 'count'])
            st.dataframe(stats.round(4))

        elif viz_mode == '2D Scatter':
            c1, c2 = st.columns(2)
            with c1:
                factor_x = st.selectbox(
                    "X Axis",
                    factor_cols,
                    index=0
                )
            with c2:
                factor_y = st.selectbox(
                    "Y Axis",
                    factor_cols,
                    index=min(1, len(factor_cols) - 1)
                )

            color_options = ['finalRatingStatus', 'language', 'coreNoteIntercept'] + factor_cols
            if enable_clustering and len(cluster_factors) >= 2:
                color_options = ['cluster'] + color_options

            color_by = st.selectbox(
                "Color by",
                color_options,
                format_func=lambda x: x.replace('finalRatingStatus', 'Status').replace('coreNoteIntercept', 'Intercept').replace('cluster', 'Cluster')
            )

            # Create hover text - wrap text for readability
            def wrap_text(text, width=60):
                if not isinstance(text, str):
                    return ''
                text = text[:400]  # Limit length
                words = text.split()
                lines = []
                current_line = []
                current_len = 0
                for word in words:
                    if current_len + len(word) + 1 <= width:
                        current_line.append(word)
                        current_len += len(word) + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_len = len(word)
                if current_line:
                    lines.append(' '.join(current_line))
                return '<br>'.join(lines[:8])  # Max 8 lines

            plot_df = plot_df.copy()
            plot_df['hover_text'] = plot_df['summary'].apply(wrap_text)

            # Create custom hover template
            hover_template = (
                "<b>Note:</b> %{customdata[0]}<br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<b>Ratings:</b> %{customdata[2]}<br>"
                "<b>Intercept:</b> %{customdata[3]:.3f}<br>"
                "<br><b>Summary:</b><br>%{customdata[4]}<extra></extra>"
            )

            if color_by == 'cluster':
                # Convert cluster to string for categorical coloring
                plot_df = plot_df.copy()
                plot_df['cluster_str'] = plot_df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
                fig = px.scatter(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    color='cluster_str',
                    custom_data=['noteId', 'finalRatingStatus', 'numRatings', 'coreNoteIntercept', 'hover_text'],
                    opacity=0.6,
                    title=f"{factor_x} vs {factor_y} (HDBSCAN)"
                )
            elif color_by in ['finalRatingStatus', 'language']:
                fig = px.scatter(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    color=color_by,
                    custom_data=['noteId', 'finalRatingStatus', 'numRatings', 'coreNoteIntercept', 'hover_text'],
                    opacity=0.5,
                    title=f"{factor_x} vs {factor_y}",
                    color_discrete_map={
                        'CURRENTLY_RATED_HELPFUL': 'green',
                        'CURRENTLY_RATED_NOT_HELPFUL': 'red',
                        'NEEDS_MORE_RATINGS': 'gray'
                    } if color_by == 'finalRatingStatus' else None
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    color=color_by,
                    custom_data=['noteId', 'finalRatingStatus', 'numRatings', 'coreNoteIntercept', 'hover_text'],
                    opacity=0.5,
                    title=f"{factor_x} vs {factor_y}",
                    color_continuous_scale='RdBu_r'
                )

            fig.update_traces(hovertemplate=hover_template)

            # Add crosshairs at origin
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig.update_layout(height=700)
            fig.update_traces(marker=dict(size=5))

            # Add highlighted notes overlay
            if highlight_ids:
                highlighted_df = plot_df[plot_df['highlighted']]
                if len(highlighted_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=highlighted_df[factor_x],
                        y=highlighted_df[factor_y],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='yellow',
                            line=dict(color='black', width=2),
                            symbol='star'
                        ),
                        name='Highlighted',
                        customdata=highlighted_df[['noteId', 'finalRatingStatus', 'numRatings', 'coreNoteIntercept', 'hover_text']].values,
                        hovertemplate=hover_template
                    ))

            st.plotly_chart(fig, use_container_width=True)

            # Correlation
            corr = plot_df[[factor_x, factor_y]].corr().iloc[0, 1]
            st.metric("Correlation", f"{corr:.4f}")

        elif viz_mode == '3D Scatter':
            c1, c2, c3 = st.columns(3)
            with c1:
                factor_x = st.selectbox(
                    "X Axis",
                    factor_cols,
                    index=0,
                    key='3d_x'
                )
            with c2:
                factor_y = st.selectbox(
                    "Y Axis",
                    factor_cols,
                    index=min(1, len(factor_cols) - 1),
                    key='3d_y'
                )
            with c3:
                factor_z = st.selectbox(
                    "Z Axis",
                    factor_cols,
                    index=min(2, len(factor_cols) - 1),
                    key='3d_z'
                )

            color_options_3d = ['finalRatingStatus', 'language'] + [f for f in factor_cols if f not in [factor_x, factor_y, factor_z]]
            if enable_clustering and len(cluster_factors) >= 2:
                color_options_3d = ['cluster'] + color_options_3d

            color_by_3d = st.selectbox(
                "Color by",
                color_options_3d,
                format_func=lambda x: x.replace('finalRatingStatus', 'Status').replace('cluster', 'Cluster'),
                key='3d_color'
            )

            # Create hover text for 3D
            def wrap_text_3d(text, width=50):
                if not isinstance(text, str):
                    return ''
                text = text[:300]
                words = text.split()
                lines = []
                current_line = []
                current_len = 0
                for word in words:
                    if current_len + len(word) + 1 <= width:
                        current_line.append(word)
                        current_len += len(word) + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_len = len(word)
                if current_line:
                    lines.append(' '.join(current_line))
                return '<br>'.join(lines[:6])

            plot_df = plot_df.copy()
            plot_df['hover_text'] = plot_df['summary'].apply(wrap_text_3d)

            hover_template_3d = (
                "<b>Note:</b> %{customdata[0]}<br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<br><b>Summary:</b><br>%{customdata[2]}<extra></extra>"
            )

            if color_by_3d == 'cluster':
                plot_df = plot_df.copy()
                plot_df['cluster_str'] = plot_df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
                fig = px.scatter_3d(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    z=factor_z,
                    color='cluster_str',
                    custom_data=['noteId', 'finalRatingStatus', 'hover_text'],
                    opacity=0.6,
                    title="3D Factor Space (HDBSCAN)"
                )
            elif color_by_3d in ['finalRatingStatus', 'language']:
                fig = px.scatter_3d(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    z=factor_z,
                    color=color_by_3d,
                    custom_data=['noteId', 'finalRatingStatus', 'hover_text'],
                    opacity=0.6,
                    title="3D Factor Space",
                    color_discrete_map={
                        'CURRENTLY_RATED_HELPFUL': 'green',
                        'CURRENTLY_RATED_NOT_HELPFUL': 'red',
                        'NEEDS_MORE_RATINGS': 'gray'
                    } if color_by_3d == 'finalRatingStatus' else None
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x=factor_x,
                    y=factor_y,
                    z=factor_z,
                    color=color_by_3d,
                    custom_data=['noteId', 'finalRatingStatus', 'hover_text'],
                    opacity=0.6,
                    title="3D Factor Space",
                    color_continuous_scale='RdBu_r'
                )

            fig.update_traces(hovertemplate=hover_template_3d)

            fig.update_layout(height=800)
            fig.update_traces(marker=dict(size=3))

            # Add highlighted notes overlay for 3D
            if highlight_ids:
                highlighted_df = plot_df[plot_df['highlighted']]
                if len(highlighted_df) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=highlighted_df[factor_x],
                        y=highlighted_df[factor_y],
                        z=highlighted_df[factor_z],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='yellow',
                            line=dict(color='black', width=2),
                            symbol='diamond'
                        ),
                        name='Highlighted',
                        customdata=highlighted_df[['noteId', 'finalRatingStatus', 'hover_text']].values,
                        hovertemplate=hover_template_3d
                    ))

            st.plotly_chart(fig, use_container_width=True)

        else:  # Data Table
            st.subheader("Filtered Notes")

            # Sort options
            sort_col = st.selectbox(
                "Sort by",
                factor_cols + ['coreNoteIntercept', 'numRatings', 'length'],
                format_func=lambda x: x.replace('coreNoteIntercept', 'Intercept')
            )
            sort_asc = st.checkbox("Ascending", value=False)

            display_cols = ['noteId', 'finalRatingStatus', 'coreNoteIntercept', 'numRatings'] + factor_cols + ['summary']
            display_df = filtered_df[display_cols].sort_values(sort_col, ascending=sort_asc)

            # Pagination
            page_size = 50
            total_pages = max(1, len(display_df) // page_size + (1 if len(display_df) % page_size else 0))
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size

            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                height=600,
                use_container_width=True
            )

            st.caption(f"Showing {start_idx+1}-{min(end_idx, len(display_df))} of {len(display_df)} notes")

    with col2:
        st.subheader("📋 Quick Stats")

        st.metric("Total Notes", f"{len(filtered_df):,}")

        status_counts = filtered_df['finalRatingStatus'].value_counts()
        for status, count in status_counts.items():
            label = status.replace('CURRENTLY_RATED_', '').replace('_', ' ').title()
            st.metric(label, f"{count:,}")

        st.divider()

        st.subheader("📤 Export")

        export_format = st.radio("Format", ['CSV', 'TSV'], horizontal=True)

        # Export scope option
        export_scope = st.radio(
            "Export scope",
            ['Displayed points only', 'All filtered notes'],
            help=f"Displayed: {len(plot_df):,} points | All filtered: {len(filtered_df):,} notes"
        )

        # Cluster export options
        if enable_clustering and len(cluster_factors) >= 2 and 'cluster' in filtered_df.columns:
            base_df = plot_df if export_scope == 'Displayed points only' else filtered_df
            available_clusters = sorted([c for c in base_df['cluster'].unique()])
            export_cluster = st.selectbox(
                "Export cluster",
                ['All (including noise)', 'All clusters (no noise)'] + [f'Cluster {c}' if c >= 0 else 'Noise only' for c in available_clusters],
                key='export_cluster'
            )
        else:
            export_cluster = 'All (including noise)'

        if st.button("Export Filtered Notes", type="primary"):
            export_cols = ['noteId', 'finalRatingStatus', 'coreNoteIntercept', 'numRatings'] + factor_cols
            if enable_clustering and len(cluster_factors) >= 2:
                export_cols.append('cluster')
            export_cols.append('summary')

            # Use plot_df (sampled) or filtered_df (all) based on scope
            export_df = plot_df.copy() if export_scope == 'Displayed points only' else filtered_df.copy()

            # Apply cluster filter for export
            if export_cluster == 'All clusters (no noise)':
                export_df = export_df[export_df['cluster'] >= 0]
            elif export_cluster == 'Noise only':
                export_df = export_df[export_df['cluster'] == -1]
            elif export_cluster.startswith('Cluster '):
                cluster_num = int(export_cluster.replace('Cluster ', ''))
                export_df = export_df[export_df['cluster'] == cluster_num]

            export_df = export_df[export_cols]

            file_suffix = export_cluster.lower().replace(' ', '_').replace('(', '').replace(')', '')
            if export_format == 'CSV':
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download CSV ({len(export_df):,} notes)",
                    data=csv,
                    file_name=f"notes_{file_suffix}.csv",
                    mime="text/csv"
                )
            else:
                tsv = export_df.to_csv(index=False, sep='\t')
                st.download_button(
                    label=f"📥 Download TSV ({len(export_df):,} notes)",
                    data=tsv,
                    file_name=f"notes_{file_suffix}.tsv",
                    mime="text/tab-separated-values"
                )

        st.divider()

        # Quick view of sample notes
        st.subheader("📝 Sample Notes")

        if len(filtered_df) > 0:
            sample_notes = filtered_df.sample(min(5, len(filtered_df)))
            for _, row in sample_notes.iterrows():
                note_id_str = str(row['noteId'])[:12]
                status_short = row['finalRatingStatus'].replace('CURRENTLY_RATED_', '')
                with st.expander(f"Note {note_id_str}... ({status_short})"):
                    summary_text = str(row['summary'])[:500] if pd.notna(row['summary']) else 'N/A'
                    st.write(summary_text)
                    factor_str = " | ".join(f"{col}: {row[col]:.2f}" for col in factor_cols)
                    st.caption(factor_str)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.utils import initialize_session_state
from modules.visualization import DataVisualizer
import io
import base64

initialize_session_state()

st.title("📊 Data Quality Insights & Visualization")

st.markdown("""
Visualize data quality metrics and cleaning progress to identify priorities and track improvements. 
This dashboard provides actionable insights to guide your data cleaning workflow.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

df = st.session_state.dataset
original_df = st.session_state.original_dataset
visualizer = DataVisualizer()

if 'saved_visualizations' not in st.session_state:
    st.session_state.saved_visualizations = []

# ===== 1. DATA QUALITY OVERVIEW =====
st.subheader("1. 📈 Data Quality Overview")

overview_cols = st.columns(4)

with overview_cols[0]:
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    st.metric("Missing Data", f"{missing_pct:.1f}%", 
             delta=f"{total_missing:,} cells",
             delta_color="inverse")

with overview_cols[1]:
    analyzed_count = len(st.session_state.column_analysis)
    total_cols = len(df.columns)
    analyzed_pct = (analyzed_count / total_cols * 100) if total_cols > 0 else 0
    st.metric("Analyzed Columns", f"{analyzed_count}/{total_cols}",
             delta=f"{analyzed_pct:.0f}%",
             delta_color="normal")

with overview_cols[2]:
    cleaned_count = len(st.session_state.cleaning_history)
    cleaned_pct = (cleaned_count / total_cols * 100) if total_cols > 0 else 0
    st.metric("Cleaned Columns", f"{cleaned_count}/{total_cols}",
             delta=f"{cleaned_pct:.0f}%",
             delta_color="normal")

with overview_cols[3]:
    if st.session_state.column_analysis:
        avg_quality = sum(
            analysis.get('data_quality', {}).get('score', 0) 
            for analysis in st.session_state.column_analysis.values()
        ) / len(st.session_state.column_analysis)
        quality_color = "green" if avg_quality >= 80 else "orange" if avg_quality >= 60 else "red"
        st.metric("Avg Quality Score", f"{avg_quality:.0f}/100",
                 delta="Good" if avg_quality >= 80 else "Needs Work",
                 delta_color="normal" if avg_quality >= 80 else "inverse")
    else:
        st.metric("Avg Quality Score", "N/A")

st.divider()

# ===== 2. CLEANING PRIORITIES =====
st.subheader("2. 🎯 Cleaning Priorities")

if st.session_state.column_analysis:
    # Identify columns that need attention
    priority_data = []
    
    for col, analysis in st.session_state.column_analysis.items():
        quality_score = analysis.get('data_quality', {}).get('score', 0)
        missing_pct = analysis.get('basic_info', {}).get('missing_percentage', 0)
        outlier_count = analysis.get('outlier_analysis', {}).get('summary', {}).get('consensus_outlier_count', 0)
        is_cleaned = col in st.session_state.cleaning_history
        
        # Calculate priority score (lower quality = higher priority)
        priority_score = (100 - quality_score) + missing_pct + (outlier_count / len(df) * 10)
        
        priority_data.append({
            'Column': col,
            'Quality Score': quality_score,
            'Missing %': round(missing_pct, 1),
            'Outliers': outlier_count,
            'Status': '✅ Cleaned' if is_cleaned else '⏳ Pending',
            'Priority': 'High' if priority_score > 50 else 'Medium' if priority_score > 20 else 'Low',
            'priority_score': priority_score
        })
    
    priority_df = pd.DataFrame(priority_data)
    priority_df = priority_df.sort_values('priority_score', ascending=False).drop('priority_score', axis=1)
    
    # Show high priority columns
    high_priority = priority_df[priority_df['Priority'] == 'High']
    
    if len(high_priority) > 0:
        st.warning(f"🔴 {len(high_priority)} high-priority columns need attention")
        st.dataframe(high_priority, use_container_width=True, hide_index=True)
        
        st.markdown("**Recommendation:** Focus on these columns first for maximum data quality improvement.")
    else:
        st.success("✅ No high-priority issues detected!")
    
    # Priority visualization
    fig = go.Figure()
    
    for priority_level, color in [('High', 'red'), ('Medium', 'orange'), ('Low', 'green')]:
        level_data = priority_df[priority_df['Priority'] == priority_level]
        if len(level_data) > 0:
            fig.add_trace(go.Bar(
                name=f'{priority_level} Priority',
                x=level_data['Column'],
                y=level_data['Quality Score'],
                marker_color=color,
                text=level_data['Quality Score'],
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Data Quality by Column (Sorted by Priority)",
        xaxis_title="Columns",
        yaxis_title="Quality Score",
        height=400,
        showlegend=True,
        barmode='group',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📊 Analyze columns first to see cleaning priorities")

st.divider()

# ===== 3. MISSING DATA PATTERNS =====
st.subheader("3. ❌ Missing Data Patterns")

missing_summary = df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(missing_cols) > 0:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Missing data bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=missing_cols.index,
            y=missing_cols.values,
            marker_color='red',
            text=missing_cols.values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Columns",
            yaxis_title="Count",
            height=350,
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Missing Data Summary")
        missing_pct = (missing_cols / len(df) * 100).round(1)
        
        for col in missing_cols.head(10).index:
            pct = missing_pct[col]
            count = missing_cols[col]
            color = "🔴" if pct > 20 else "🟡" if pct > 5 else "🟢"
            st.write(f"{color} **{col}**: {count:,} ({pct}%)")
        
        if len(missing_cols) > 10:
            st.caption(f"... and {len(missing_cols) - 10} more columns")
        
        st.markdown("---")
        st.markdown("**Action Items:**")
        critical = sum(missing_pct > 20)
        if critical > 0:
            st.write(f"• {critical} columns with >20% missing")
            st.write(f"• Consider removal or imputation")
        else:
            st.write(f"• {len(missing_cols)} columns need attention")
            st.write(f"• Review imputation strategies")
    
    # Missing pattern heatmap (top 20 columns)
    if len(missing_cols) > 0:
        st.markdown("### Missing Data Pattern Matrix")
        top_missing_cols = missing_cols.head(20).index.tolist()
        pattern_fig = visualizer.plot_missing_patterns(df[top_missing_cols])
        st.plotly_chart(pattern_fig, use_container_width=True)
else:
    st.success("✅ No missing data in the dataset!")

st.divider()

# ===== 4. BEFORE/AFTER COMPARISON =====
st.subheader("4. 🔄 Cleaning Impact")

if original_df is not None and len(st.session_state.cleaning_history) > 0:
    st.markdown("Compare data quality before and after cleaning operations")
    
    comparison_cols = st.columns(3)
    
    with comparison_cols[0]:
        orig_missing = original_df.isnull().sum().sum()
        curr_missing = df.isnull().sum().sum()
        reduction = orig_missing - curr_missing
        reduction_pct = (reduction / orig_missing * 100) if orig_missing > 0 else 0
        
        st.metric("Missing Values Reduced", 
                 f"{reduction:,}",
                 delta=f"-{reduction_pct:.1f}%",
                 delta_color="normal")
    
    with comparison_cols[1]:
        st.metric("Columns Cleaned", 
                 f"{len(st.session_state.cleaning_history)}",
                 delta=f"{len(st.session_state.cleaning_history)} operations")
    
    with comparison_cols[2]:
        orig_quality = original_df.shape[0] * original_df.shape[1] - orig_missing
        curr_quality = df.shape[0] * df.shape[1] - curr_missing
        quality_improvement = ((curr_quality - orig_quality) / (original_df.shape[0] * original_df.shape[1]) * 100)
        
        st.metric("Data Completeness Improvement",
                 f"+{quality_improvement:.1f}%",
                 delta="Better quality")
    
    # Before/After comparison chart
    comparison_data = []
    for col in st.session_state.cleaning_history.keys():
        if col in original_df.columns:
            orig_missing_col = original_df[col].isnull().sum()
            curr_missing_col = df[col].isnull().sum()
            
            comparison_data.append({
                'Column': col,
                'Before': orig_missing_col,
                'After': curr_missing_col,
                'Improvement': orig_missing_col - curr_missing_col
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Before Cleaning',
            x=comp_df['Column'],
            y=comp_df['Before'],
            marker_color='lightcoral'
        ))
        fig.add_trace(go.Bar(
            name='After Cleaning',
            x=comp_df['Column'],
            y=comp_df['After'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title="Missing Values: Before vs After Cleaning",
            xaxis_title="Cleaned Columns",
            yaxis_title="Missing Value Count",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
else:
    if original_df is None:
        st.info("📊 Original dataset not available for comparison")
    else:
        st.info("📊 No cleaning operations performed yet. Clean some columns to see the impact!")

st.divider()

# ===== 5. CUSTOM VISUALIZATIONS (OPTIONAL) =====
with st.expander("🎨 Create Custom Visualizations (Advanced)", expanded=False):
    st.markdown("""
    Create additional custom visualizations for specific analysis needs. 
    These can be saved and included in your final report.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_columns = st.multiselect(
            "Select column(s):",
            options=list(df.columns),
            help="Choose columns to visualize"
        )
    
    with col2:
        viz_type = st.selectbox(
            "Chart Type:",
            options=['bar', 'line', 'scatter', 'box', 'histogram', 'correlation'],
            format_func=lambda x: {
                'bar': '📊 Bar Chart',
                'line': '📈 Line Chart',
                'scatter': '🔵 Scatter Plot',
                'box': '📦 Box Plot',
                'histogram': '📊 Histogram',
                'correlation': '🔗 Correlation Matrix'
            }[x]
        )
    
    if selected_columns:
        chart_title = st.text_input("Chart Title:", value=f"{viz_type.title()} - {', '.join(selected_columns[:2])}")
        
        if st.button("🎨 Generate Visualization", type="primary"):
            try:
                fig = None
                
                if viz_type == 'bar':
                    col = selected_columns[0]
                    value_counts = df[col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               labels={'x': col, 'y': 'Count'}, title=chart_title)
                
                elif viz_type == 'line':
                    col = selected_columns[0]
                    fig = px.line(df, y=col, title=chart_title)
                
                elif viz_type == 'scatter' and len(selected_columns) >= 2:
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], title=chart_title)
                
                elif viz_type == 'box':
                    fig = go.Figure()
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig.add_trace(go.Box(y=df[col].dropna(), name=col))
                    fig.update_layout(title=chart_title)
                
                elif viz_type == 'histogram':
                    fig = go.Figure()
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, opacity=0.7))
                    fig.update_layout(title=chart_title, barmode='overlay')
                
                elif viz_type == 'correlation':
                    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if len(numeric_cols) >= 2:
                        fig = visualizer.plot_correlation_matrix(df[numeric_cols])
                        fig.update_layout(title=chart_title)
                    else:
                        st.error("Need at least 2 numeric columns for correlation")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("💾 Save to Report"):
                        img_bytes = fig.to_image(format="png", width=1200, height=600)
                        img_b64 = base64.b64encode(img_bytes).decode()
                        
                        st.session_state.saved_visualizations.append({
                            'name': chart_title,
                            'type': viz_type,
                            'columns': selected_columns,
                            'title': chart_title,
                            'fig': fig,
                            'img_b64': img_b64
                        })
                        
                        st.success(f"✅ Visualization saved!")
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")

# ===== 6. SAVED VISUALIZATIONS =====
st.divider()

st.subheader("6. 💾 Saved Visualizations for Reports")

if st.session_state.saved_visualizations:
    st.success(f"📊 {len(st.session_state.saved_visualizations)} visualization(s) saved")
    
    for idx, viz in enumerate(st.session_state.saved_visualizations):
        with st.expander(f"📈 {viz['name']}"):
            st.plotly_chart(viz['fig'], use_container_width=True)
            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                st.session_state.saved_visualizations.pop(idx)
                st.rerun()
    
    if st.button("🗑️ Clear All"):
        st.session_state.saved_visualizations = []
        st.rerun()
else:
    st.info("No custom visualizations saved yet.")

# ===== 7. RECOMMENDATIONS =====
st.divider()

st.subheader("7. 💡 Recommended Next Steps")

recommendations = []

# Check for high priority issues
if st.session_state.column_analysis:
    high_priority_cols = [
        col for col, analysis in st.session_state.column_analysis.items()
        if analysis.get('data_quality', {}).get('score', 100) < 70
    ]
    
    if high_priority_cols:
        recommendations.append(f"🔴 **Address {len(high_priority_cols)} low-quality columns**: " + 
                              ", ".join(high_priority_cols[:3]) + 
                              ("..." if len(high_priority_cols) > 3 else ""))

# Check for unanalyzed columns
unanalyzed = set(df.columns) - set(st.session_state.column_analysis.keys())
if unanalyzed:
    recommendations.append(f"📊 **Analyze {len(unanalyzed)} remaining columns** for complete coverage")

# Check for uncleaned analyzed columns
if st.session_state.column_analysis:
    analyzed_uncleaned = set(st.session_state.column_analysis.keys()) - set(st.session_state.cleaning_history.keys())
    if analyzed_uncleaned:
        recommendations.append(f"🧹 **Consider cleaning {len(analyzed_uncleaned)} analyzed columns** with issues")

# Check missing data
if len(missing_cols) > 0:
    critical_missing = sum((missing_cols / len(df) * 100) > 20)
    if critical_missing > 0:
        recommendations.append(f"⚠️ **{critical_missing} columns have >20% missing data** - review imputation strategies")

if recommendations:
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    st.markdown("---")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🔍 Analyze Columns", use_container_width=True):
            st.switch_page("pages/2_Column_Analysis.py")
    with action_cols[1]:
        if st.button("🧹 Clean Data", use_container_width=True):
            st.switch_page("pages/3_Cleaning_Wizard.py")
    with action_cols[2]:
        if st.button("📊 Generate Report", use_container_width=True):
            st.switch_page("pages/6_Reports.py")
else:
    st.success("✅ Great work! Your dataset quality looks good. Consider generating a final report.")

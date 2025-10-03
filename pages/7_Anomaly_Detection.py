import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state
from modules.data_analyzer import ColumnAnalyzer
from modules.visualization import DataVisualizer
import plotly.graph_objects as go

initialize_session_state()

st.title("🔍 Anomaly Detection")

st.markdown("""
Detect and analyze anomalies across all columns in your dataset. This section uses multiple detection methods 
to identify unusual patterns, outliers, and data quality issues.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

df = st.session_state.dataset
analyzer = ColumnAnalyzer()
visualizer = DataVisualizer()

if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = {}

st.subheader("1. Run Anomaly Detection")

detect_col1, detect_col2 = st.columns([3, 1])

with detect_col1:
    st.markdown("Detect anomalies across all columns or specific columns")

with detect_col2:
    if st.button("🔍 Detect All Anomalies", type="primary"):
        with st.spinner("Analyzing dataset for anomalies..."):
            anomaly_results = {}
            
            for col in df.columns:
                series = df[col]
                
                if pd.api.types.is_numeric_dtype(series):
                    analysis = analyzer.analyze_column(df, col)
                    outlier_info = analysis.get('outlier_analysis', {})
                    
                    if outlier_info.get('summary', {}).get('consensus_outlier_count', 0) > 0:
                        anomaly_results[col] = {
                            'type': 'numeric_outliers',
                            'count': outlier_info['summary']['consensus_outlier_count'],
                            'percentage': outlier_info['summary']['consensus_outlier_percentage'],
                            'severity': outlier_info['summary']['severity'],
                            'methods': outlier_info,
                            'indices': []
                        }
                        
                        for method, result in outlier_info.items():
                            if isinstance(result, dict) and 'outlier_values' in result:
                                outlier_vals = result['outlier_values']
                                outlier_indices = series[series.isin(outlier_vals)].index.tolist()
                                anomaly_results[col]['indices'].extend(outlier_indices)
                        
                        anomaly_results[col]['indices'] = list(set(anomaly_results[col]['indices']))[:100]
                
                else:
                    missing_count = series.isnull().sum()
                    if missing_count > 0:
                        unusual_values = []
                        value_counts = series.value_counts()
                        
                        if len(value_counts) > 0:
                            rare_threshold = len(series) * 0.01
                            unusual_values = value_counts[value_counts < rare_threshold].index.tolist()[:20]
                        
                        if unusual_values or missing_count > len(series) * 0.1:
                            anomaly_results[col] = {
                                'type': 'categorical_issues',
                                'missing_count': int(missing_count),
                                'unusual_values': unusual_values,
                                'unusual_count': len(unusual_values)
                            }
            
            st.session_state.anomaly_results = anomaly_results
            
            if anomaly_results:
                st.success(f"✅ Anomaly detection complete! Found issues in {len(anomaly_results)} columns.")
            else:
                st.info("✅ No significant anomalies detected in the dataset.")
            st.rerun()

st.divider()

st.subheader("2. Anomaly Detection Results")

if st.session_state.anomaly_results:
    total_anomalies = sum(
        result.get('count', result.get('missing_count', 0)) 
        for result in st.session_state.anomaly_results.values()
    )
    
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Columns with Anomalies", len(st.session_state.anomaly_results))
    with metrics_cols[1]:
        st.metric("Total Anomalies", total_anomalies)
    with metrics_cols[2]:
        high_severity = sum(1 for r in st.session_state.anomaly_results.values() 
                          if r.get('severity') == 'high')
        st.metric("High Severity", high_severity)
    with metrics_cols[3]:
        avg_pct = np.mean([r.get('percentage', 0) for r in st.session_state.anomaly_results.values()])
        st.metric("Avg Anomaly %", f"{avg_pct:.1f}%")
    
    st.divider()
    
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        severity_filter = st.multiselect(
            "Filter by Severity:",
            options=['low', 'moderate', 'high'],
            default=['low', 'moderate', 'high']
        )
    
    with filter_col2:
        type_filter = st.multiselect(
            "Filter by Type:",
            options=['numeric_outliers', 'categorical_issues'],
            default=['numeric_outliers', 'categorical_issues'],
            format_func=lambda x: 'Numeric Outliers' if x == 'numeric_outliers' else 'Categorical Issues'
        )
    
    st.markdown("### Detailed Anomaly Reports")
    
    for col, result in st.session_state.anomaly_results.items():
        if result['type'] in type_filter:
            if result['type'] == 'numeric_outliers' and result.get('severity') not in severity_filter:
                continue
            
            severity_emoji = {'low': '🟢', 'moderate': '🟡', 'high': '🔴'}.get(result.get('severity', 'low'), '⚪')
            
            with st.expander(f"{severity_emoji} **{col}** - {result['type'].replace('_', ' ').title()} ({result.get('count', result.get('missing_count', 0))} anomalies)"):
                
                if result['type'] == 'numeric_outliers':
                    st.markdown(f"**Severity:** {result['severity'].title()}")
                    st.markdown(f"**Anomaly Count:** {result['count']}")
                    st.markdown(f"**Percentage:** {result['percentage']:.2f}%")
                    
                    st.markdown("#### Detection Methods:")
                    methods_df_data = []
                    for method, method_result in result['methods'].items():
                        if isinstance(method_result, dict) and 'method' in method_result:
                            methods_df_data.append({
                                'Method': method_result['method'],
                                'Count': method_result.get('outlier_count', 0),
                                'Percentage': f"{method_result.get('outlier_percentage', 0):.2f}%"
                            })
                    
                    if methods_df_data:
                        st.dataframe(pd.DataFrame(methods_df_data), use_container_width=True, hide_index=True)
                    
                    if result.get('indices'):
                        st.markdown("#### Anomalous Data Points:")
                        anomaly_df = df.loc[result['indices'][:50], [col]]
                        anomaly_df['Row_Index'] = anomaly_df.index
                        anomaly_df = anomaly_df[['Row_Index', col]]
                        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
                        
                        if len(result['indices']) > 50:
                            st.caption(f"Showing first 50 of {len(result['indices'])} anomalies")
                    
                    try:
                        analysis = analyzer.analyze_column(df, col)
                        fig = visualizer.plot_outliers(df[col], col, analysis['outlier_analysis'])
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                
                elif result['type'] == 'categorical_issues':
                    st.markdown(f"**Missing Values:** {result['missing_count']}")
                    
                    if result.get('unusual_values'):
                        st.markdown(f"**Unusual/Rare Values:** {result['unusual_count']}")
                        st.write("Rare values (< 1% frequency):")
                        st.write(", ".join([str(v) for v in result['unusual_values'][:10]]))
                    
                    value_counts = df[col].value_counts().head(20)
                    fig = go.Figure(data=[
                        go.Bar(x=value_counts.index.astype(str), y=value_counts.values)
                    ])
                    fig.update_layout(
                        title=f"Value Distribution - {col}",
                        xaxis_title="Values",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("3. Export Anomalies")
    
    export_cols = st.columns([2, 1, 1])
    
    with export_cols[0]:
        st.markdown("Export anomaly detection results for further analysis")
    
    with export_cols[1]:
        if st.button("📄 Export Summary"):
            summary_data = []
            for col, result in st.session_state.anomaly_results.items():
                summary_data.append({
                    'Column': col,
                    'Type': result['type'],
                    'Anomaly_Count': result.get('count', result.get('missing_count', 0)),
                    'Severity': result.get('severity', 'N/A'),
                    'Percentage': result.get('percentage', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download Anomaly Summary CSV",
                data=csv,
                file_name="anomaly_summary.csv",
                mime="text/csv"
            )
    
    with export_cols[2]:
        if 'saved_visualizations' not in st.session_state:
            st.session_state.saved_visualizations = []
        
        if st.button("💾 Save to Report"):
            st.session_state.anomaly_results_for_report = st.session_state.anomaly_results.copy()
            st.success("✅ Anomaly results saved to report!")

else:
    st.info("👆 Click 'Detect All Anomalies' to start the analysis")
    
    st.markdown("""
    ### 🔍 What We Detect:
    
    **For Numeric Columns:**
    - **IQR Method**: Detects values outside 1.5 × IQR range
    - **Z-Score Method**: Identifies values with |z-score| > 3
    - **Modified Z-Score**: Uses median absolute deviation for robust detection
    - **Consensus Outliers**: Values flagged by multiple methods
    
    **For Categorical Columns:**
    - **Missing Values**: High percentage of null values
    - **Rare Values**: Categories appearing < 1% of the time
    - **Data Quality Issues**: Inconsistent formats or unusual patterns
    
    ### 📊 Severity Levels:
    - 🟢 **Low**: < 1% anomalies
    - 🟡 **Moderate**: 1-5% anomalies  
    - 🔴 **High**: > 5% anomalies
    """)

st.divider()

st.markdown("""
### 💡 Best Practices:
- Review high severity anomalies first
- Investigate the context before removing outliers
- Consider domain knowledge when assessing anomalies
- Use the Cleaning Wizard to handle confirmed anomalies
- All anomaly detection results can be included in your final report
""")

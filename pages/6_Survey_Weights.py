import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state
from modules.survey_weights import SurveyWeightsManager
from modules.visualization import DataVisualizer
import plotly.express as px
import plotly.graph_objects as go

# Initialize session state
initialize_session_state()

st.title("⚖️ Survey Weights Analysis")

# Check if dataset is loaded
if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

df = st.session_state.dataset

# Initialize weights manager
if 'weights_manager' not in st.session_state:
    st.session_state.weights_manager = SurveyWeightsManager()

weights_manager = st.session_state.weights_manager

st.markdown("""
Apply and analyze survey design weights to account for sampling design effects. 
Compare weighted vs unweighted estimates and calculate margins of error.
""")

# Weights Configuration
st.subheader("🔧 Weights Configuration")

# Select weights column
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) == 0:
    st.error("❌ No numeric columns found. Weights must be numeric.")
    st.stop()

weights_col = st.selectbox(
    "Select weights column:",
    options=['None'] + numeric_columns,
    help="Choose the column containing survey design weights"
)

if weights_col != 'None':
    # Set weights column
    if st.button("Set Weights Column", type="primary"):
        try:
            validation_result = weights_manager.set_weights_column(df, weights_col)
            
            if validation_result['valid']:
                st.success(f"✅ Weights column '{weights_col}' set successfully!")
                
                if validation_result['warnings']:
                    st.warning("⚠️ Warnings:")
                    for warning in validation_result['warnings']:
                        st.write(f"• {warning}")
                        
            else:
                st.error("❌ Invalid weights column:")
                for error in validation_result['errors']:
                    st.write(f"• {error}")
                    
        except Exception as e:
            st.error(f"❌ Error setting weights column: {str(e)}")

# Display weights diagnostics if weights are set
if weights_manager.weights_column:
    st.subheader("📊 Weights Diagnostics")
    
    diagnostics = weights_manager.get_weights_diagnostics(df)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Weight", f"{diagnostics['basic_stats']['mean']:.3f}")
    with col2:
        st.metric("Weight Range", 
                 f"{diagnostics['basic_stats']['min']:.2f} - {diagnostics['basic_stats']['max']:.2f}")
    with col3:
        st.metric("Design Effect", f"{diagnostics['design_effect_estimate']['design_effect']:.2f}")
    with col4:
        st.metric("Effective N", f"{diagnostics['design_effect_estimate']['effective_sample_size']:.0f}")
    
    # Weights distribution visualization
    st.subheader("📈 Weights Distribution")
    
    weights_data = df[weights_manager.weights_column].dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            x=weights_data,
            nbins=30,
            title="Distribution of Weights",
            labels={'x': 'Weight Value', 'y': 'Frequency'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = go.Figure(data=go.Box(y=weights_data, name="Weights"))
        fig.update_layout(title="Weights Box Plot", yaxis_title="Weight Value")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    if diagnostics['recommendations']:
        st.subheader("💡 Recommendations")
        for rec in diagnostics['recommendations']:
            st.info(f"• {rec}")

# Weighted Analysis
if weights_manager.weights_column:
    st.subheader("📊 Weighted Analysis")
    
    # Select columns for analysis
    analysis_columns = st.multiselect(
        "Select columns to analyze:",
        options=[col for col in df.columns if col != weights_manager.weights_column],
        help="Choose columns to compare weighted vs unweighted statistics"
    )
    
    confidence_level = st.slider(
        "Confidence Level for Margins of Error:",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f"
    )
    
    if analysis_columns:
        # Perform analysis
        if st.button("Run Weighted Analysis", type="primary"):
            with st.spinner("Calculating weighted statistics..."):
                
                # Compare weighted vs unweighted
                comparison = weights_manager.compare_weighted_unweighted(df, analysis_columns)
                
                st.subheader("🔍 Results Summary")
                
                for column in analysis_columns:
                    if column in comparison['results']:
                        result = comparison['results'][column]
                        
                        st.write(f"### {column}")
                        
                        if 'weighted' in result and 'unweighted' in result:
                            weighted = result['weighted']
                            unweighted = result['unweighted']
                            
                            # Display comparison table
                            if 'mean' in weighted and 'mean' in unweighted:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Unweighted Mean", f"{unweighted['mean']:.4f}")
                                with col2:
                                    st.metric("Weighted Mean", f"{weighted['mean']:.4f}")
                                with col3:
                                    if 'differences' in result and 'mean_percent_change' in result['differences']:
                                        change = result['differences']['mean_percent_change']
                                        st.metric("% Change", f"{change:+.2f}%")
                                
                                # Additional statistics
                                stats_df = pd.DataFrame({
                                    'Unweighted': {
                                        'Count': unweighted.get('count', 'N/A'),
                                        'Mean': f"{unweighted.get('mean', 'N/A'):.4f}" if 'mean' in unweighted else 'N/A',
                                        'Std Dev': f"{unweighted.get('std', 'N/A'):.4f}" if 'std' in unweighted else 'N/A',
                                        'Median': f"{unweighted.get('q50', 'N/A'):.4f}" if 'q50' in unweighted else 'N/A'
                                    },
                                    'Weighted': {
                                        'Count': weighted.get('count', 'N/A'),
                                        'Mean': f"{weighted.get('mean', 'N/A'):.4f}" if 'mean' in weighted else 'N/A',
                                        'Std Dev': f"{weighted.get('std', 'N/A'):.4f}" if 'std' in weighted else 'N/A',
                                        'Median': f"{weighted.get('q50', 'N/A'):.4f}" if 'q50' in weighted else 'N/A'
                                    }
                                })
                                
                                st.dataframe(stats_df, use_container_width=True)
                            
                            elif 'mode' in weighted and 'mode' in unweighted:
                                # For categorical variables
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Unweighted Mode:**", unweighted['mode'])
                                    if 'top_values' in unweighted:
                                        st.write("**Top Values:**")
                                        for val, count in unweighted['top_values']:
                                            st.write(f"• {val}: {count}")
                                
                                with col2:
                                    st.write("**Weighted Mode:**", weighted['mode'])
                                    if 'top_values' in weighted:
                                        st.write("**Top Values (Weighted):**")
                                        for val, weight in weighted['top_values']:
                                            st.write(f"• {val}: {weight:.2f}")
                        
                        # Calculate and display margin of error
                        st.write("#### Margin of Error Analysis")
                        
                        margin_result = weights_manager.calculate_margin_of_error(
                            df, column, confidence_level
                        )
                        
                        if 'error' not in margin_result:
                            if 'margin_of_error' in margin_result:
                                # For numeric variables
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Standard Error", f"{margin_result['standard_error']:.4f}")
                                with col2:
                                    st.metric("Margin of Error", f"±{margin_result['margin_of_error']:.4f}")
                                with col3:
                                    ci = margin_result['confidence_interval']
                                    st.metric("Confidence Interval", 
                                            f"[{ci[0]:.4f}, {ci[1]:.4f}]")
                                
                            elif 'proportions' in margin_result:
                                # For categorical variables
                                st.write("**Proportions with Margins of Error:**")
                                
                                prop_data = []
                                for category in margin_result['proportions']:
                                    prop = margin_result['proportions'][category]
                                    margin = margin_result['margins_of_error'][category]
                                    ci = margin_result['confidence_intervals'][category]
                                    
                                    prop_data.append({
                                        'Category': category,
                                        'Proportion': f"{prop:.4f}",
                                        'Margin of Error': f"±{margin:.4f}",
                                        'Confidence Interval': f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                                    })
                                
                                st.dataframe(pd.DataFrame(prop_data), use_container_width=True)
                        
                        st.divider()

# Export weighted results
if weights_manager.weights_column:
    st.subheader("📥 Export Results")
    
    if st.button("Generate Weighted Analysis Report"):
        # Create comprehensive report
        report_data = {
            'weights_column': weights_manager.weights_column,
            'weights_diagnostics': weights_manager.get_weights_diagnostics(df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if analysis_columns:
            report_data['analysis_results'] = weights_manager.compare_weighted_unweighted(df, analysis_columns)
        
        # Convert to JSON for download
        import json
        report_json = json.dumps(report_data, indent=2, default=str)
        
        st.download_button(
            label="💾 Download Weights Analysis Report",
            data=report_json,
            file_name=f"weights_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Help section
with st.expander("❓ About Survey Weights"):
    st.markdown("""
    ### What are Survey Weights?
    
    Survey weights (design weights) adjust for:
    - **Unequal probability of selection** - Some respondents were more/less likely to be selected
    - **Non-response bias** - Adjusting for systematic differences in who responds
    - **Population representation** - Ensuring sample matches known population characteristics
    
    ### Key Concepts:
    
    - **Design Effect**: How much the sampling design increases variance compared to simple random sampling
    - **Effective Sample Size**: The equivalent simple random sample size accounting for weights
    - **Margin of Error**: Confidence interval width for estimates
    
    ### When to Use Weights:
    
    ✅ **Use weights when:**
    - Your survey used complex sampling (stratified, clustered, etc.)
    - You have post-stratification weights
    - Making population inferences
    
    ❌ **Don't use weights for:**
    - Simple random samples (unless post-stratified)
    - Internal data analysis where population inference isn't needed
    - Exploratory data analysis
    
    ### Interpreting Results:
    
    - Large differences between weighted/unweighted means suggest sampling bias
    - High design effects (>2) indicate weights substantially affect variance
    - Always report confidence intervals for weighted estimates
    """)
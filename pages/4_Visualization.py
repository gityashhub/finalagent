import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.utils import initialize_session_state
from modules.visualization import DataVisualizer
import io
import base64

initialize_session_state()

st.title("📊 Data Visualization")

st.markdown("""
Create custom visualizations of your cleaned data. Select multiple columns and choose from various graph types. 
All generated graphs can be saved as static images and included in your final report.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

df = st.session_state.dataset
visualizer = DataVisualizer()

if 'saved_visualizations' not in st.session_state:
    st.session_state.saved_visualizations = []

st.subheader("1. Select Data and Visualization Type")

col1, col2 = st.columns([2, 1])

with col1:
    selected_columns = st.multiselect(
        "Select column(s) to visualize:",
        options=list(df.columns),
        help="Choose one or more columns for visualization"
    )

with col2:
    viz_type = st.selectbox(
        "Graph Type:",
        options=[
            'bar', 'line', 'scatter', 'box', 'violin', 
            'histogram', 'pie', 'heatmap', 'correlation'
        ],
        format_func=lambda x: {
            'bar': '📊 Bar Chart',
            'line': '📈 Line Chart',
            'scatter': '🔵 Scatter Plot',
            'box': '📦 Box Plot',
            'violin': '🎻 Violin Plot',
            'histogram': '📊 Histogram',
            'pie': '🥧 Pie Chart',
            'heatmap': '🗺️ Heatmap',
            'correlation': '🔗 Correlation Matrix'
        }[x]
    )

if not selected_columns:
    st.info("👆 Please select at least one column to visualize.")
    st.stop()

st.subheader("2. Configure Visualization")

config_cols = st.columns(3)

with config_cols[0]:
    chart_title = st.text_input("Chart Title:", value=f"{viz_type.title()} Chart")

with config_cols[1]:
    if viz_type in ['bar', 'line', 'scatter']:
        x_axis = st.selectbox("X-axis:", options=selected_columns if len(selected_columns) > 0 else list(df.columns))
    else:
        x_axis = None

with config_cols[2]:
    if viz_type in ['scatter', 'line'] and len(selected_columns) > 1:
        y_axis = st.selectbox("Y-axis:", options=[col for col in selected_columns if col != x_axis])
    else:
        y_axis = None

st.subheader("3. Generate Visualization")

if st.button("🎨 Generate Graph", type="primary"):
    try:
        fig = None
        
        if viz_type == 'bar':
            if pd.api.types.is_numeric_dtype(df[x_axis]):
                value_counts = df[x_axis].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    labels={'x': x_axis, 'y': 'Count'},
                    title=chart_title
                )
            else:
                value_counts = df[x_axis].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    labels={'x': x_axis, 'y': 'Count'},
                    title=chart_title
                )
        
        elif viz_type == 'line':
            if y_axis:
                fig = px.line(
                    df.dropna(subset=[x_axis, y_axis]),
                    x=x_axis,
                    y=y_axis,
                    title=chart_title
                )
            else:
                fig = px.line(
                    df.dropna(subset=[x_axis]),
                    x=df.index,
                    y=x_axis,
                    title=chart_title
                )
        
        elif viz_type == 'scatter':
            if y_axis:
                fig = px.scatter(
                    df.dropna(subset=[x_axis, y_axis]),
                    x=x_axis,
                    y=y_axis,
                    title=chart_title
                )
            else:
                fig = px.scatter(
                    df.dropna(subset=[x_axis]),
                    x=df.index,
                    y=x_axis,
                    title=chart_title
                )
        
        elif viz_type == 'box':
            fig = go.Figure()
            for col in selected_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig.add_trace(go.Box(
                        y=df[col].dropna(),
                        name=col
                    ))
            fig.update_layout(title=chart_title, yaxis_title="Values")
        
        elif viz_type == 'violin':
            fig = go.Figure()
            for col in selected_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig.add_trace(go.Violin(
                        y=df[col].dropna(),
                        name=col,
                        box_visible=True,
                        meanline_visible=True
                    ))
            fig.update_layout(title=chart_title, yaxis_title="Values")
        
        elif viz_type == 'histogram':
            fig = go.Figure()
            for col in selected_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig.add_trace(go.Histogram(
                        x=df[col].dropna(),
                        name=col,
                        opacity=0.7
                    ))
            fig.update_layout(
                title=chart_title,
                xaxis_title="Value",
                yaxis_title="Frequency",
                barmode='overlay'
            )
        
        elif viz_type == 'pie':
            if len(selected_columns) == 1:
                value_counts = df[selected_columns[0]].value_counts().head(10)
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=chart_title
                )
            else:
                st.warning("Pie chart requires exactly one column. Using first selected column.")
                value_counts = df[selected_columns[0]].value_counts().head(10)
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=chart_title
                )
        
        elif viz_type == 'heatmap':
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    title=chart_title,
                    color_continuous_scale='RdBu_r'
                )
            else:
                st.error("Heatmap requires at least 2 numeric columns.")
                st.stop()
        
        elif viz_type == 'correlation':
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                fig = visualizer.plot_correlation_matrix(df[numeric_cols])
                fig.update_layout(title=chart_title)
            else:
                st.error("Correlation matrix requires at least 2 numeric columns.")
                st.stop()
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            save_cols = st.columns([3, 1])
            with save_cols[0]:
                save_name = st.text_input(
                    "Save as:",
                    value=f"{viz_type}_{len(st.session_state.saved_visualizations) + 1}",
                    key="save_viz_name"
                )
            
            with save_cols[1]:
                if st.button("💾 Save to Report", key="save_viz_btn"):
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    img_b64 = base64.b64encode(img_bytes).decode()
                    
                    st.session_state.saved_visualizations.append({
                        'name': save_name,
                        'type': viz_type,
                        'columns': selected_columns,
                        'title': chart_title,
                        'fig': fig,
                        'img_b64': img_b64
                    })
                    
                    st.success(f"✅ Visualization '{save_name}' saved to report!")
                    st.rerun()
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")

st.divider()

st.subheader("4. Saved Visualizations for Report")

if st.session_state.saved_visualizations:
    st.info(f"📊 {len(st.session_state.saved_visualizations)} visualization(s) saved for report")
    
    for idx, viz in enumerate(st.session_state.saved_visualizations):
        with st.expander(f"📈 {viz['name']} - {viz['type'].title()} ({', '.join(viz['columns'])})"):
            st.plotly_chart(viz['fig'], use_container_width=True)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🗑️ Remove", key=f"remove_viz_{idx}"):
                    st.session_state.saved_visualizations.pop(idx)
                    st.success("Visualization removed")
                    st.rerun()
    
    if st.button("🗑️ Clear All Visualizations"):
        st.session_state.saved_visualizations = []
        st.success("All visualizations cleared")
        st.rerun()
else:
    st.info("No visualizations saved yet. Generate and save visualizations to include them in your report.")

st.divider()

st.markdown("""
### 💡 Tips:
- **Bar Charts**: Best for comparing categories or discrete values
- **Line Charts**: Ideal for showing trends over time or continuous data
- **Scatter Plots**: Great for showing relationships between two variables
- **Box/Violin Plots**: Perfect for understanding data distribution and outliers
- **Histograms**: Show frequency distribution of numeric data
- **Pie Charts**: Display proportions (works best with categorical data)
- **Heatmaps/Correlation**: Visualize relationships between multiple numeric variables
- All saved visualizations will be included as static images in your final PDF report
""")

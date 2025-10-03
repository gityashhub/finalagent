# Intelligent Data Cleaning Assistant

## Overview
A Streamlit-based web application designed for statistical agencies to clean and analyze survey data. The application provides AI-powered guidance, comprehensive analysis tools, and detailed reporting capabilities for data quality assessment and cleaning operations.

## Purpose
This application helps data analysts and statisticians:
- Upload and analyze survey datasets
- Detect and handle missing values, outliers, and data quality issues
- Apply various cleaning methods with survey weight support
- Get AI-powered recommendations for cleaning strategies
- Generate comprehensive reports of all cleaning operations

## Tech Stack
- **Framework**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Analysis**: Scikit-learn, SciPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **AI Assistant**: Groq API (llama-3.1-8b-instant model)
- **Reporting**: Jinja2 templates

## Project Structure
```
.
├── app.py                          # Main application entry point
├── pages/                          # Streamlit pages
│   ├── 1_Data_Upload.py           # Dataset upload and configuration
│   ├── 2_Column_Analysis.py       # Individual column analysis
│   ├── 3_Cleaning_Wizard.py       # Data cleaning operations with validation
│   ├── 4_Visualization.py         # Custom visualizations (Phase 2)
│   ├── 5_AI_Assistant.py          # AI-powered guidance
│   ├── 6_Reports.py               # Report generation with exports
│   └── 7_Anomaly_Detection.py     # Anomaly detection (Phase 2)
├── modules/                        # Core functionality modules
│   ├── ai_assistant.py            # AI assistant integration
│   ├── cleaning_engine.py         # Data cleaning methods + validation
│   ├── data_analyzer.py           # Column analysis engine
│   ├── report_generator.py        # Report generation
│   ├── survey_weights.py          # Survey weights management
│   ├── utils.py                   # Utility functions
│   └── visualization.py           # Visualization components
├── pyproject.toml                 # Python dependencies
└── .gitignore                     # Git ignore rules
```

## Key Features

### Phase 1 (Core Features)
1. **Data Upload & Configuration**: Support for CSV and Excel files with automatic column type detection
2. **Column Analysis**: Individual column analysis with missing data patterns, outlier detection, and quality assessment
3. **Cleaning Wizard**: Multiple cleaning methods (imputation, outlier handling, standardization) with survey weight support
4. **AI Assistant**: Context-aware guidance using Groq API for cleaning recommendations
5. **Undo/Redo**: Full operation history with undo/redo functionality
6. **Survey Weights**: Integrated support for survey design weights in all analyses

### Phase 2 (Enhanced Features)
7. **Custom Visualizations**: Create and save custom charts (bar, line, scatter, box, violin, histogram, pie, heatmap, correlation) with multi-column selection
8. **Data Export**: Download cleaned and original datasets in CSV format
9. **Anomaly Detection**: Comprehensive anomaly detection across all columns with multiple detection methods (IQR, Z-score, Modified Z-score)
10. **Data Validation**: Pre-export validation checks with quality assessment and recommendations
11. **Enhanced Reporting**: 
    - Cleaned dataset preview with comparison to original
    - Include saved custom visualizations as static images
    - Include anomaly detection results
    - Export reports in Markdown, HTML, or JSON formats

## Environment Setup
The application is configured to run on Replit with:
- Python 3.11
- All dependencies managed via uv (see pyproject.toml)
- Streamlit server on port 5000 with proper CORS configuration

## Configuration
The Streamlit app is configured to:
- Listen on 0.0.0.0:5000 (required for Replit proxy)
- Disable CORS and XSRF protection for Replit environment
- Run in headless mode for server deployment

## AI Assistant Configuration
The AI assistant requires a GROQ_API_KEY environment variable to be set. Without this key, the AI features will not be available, but all other functionality will work normally.

## Recent Changes
- **2025-10-03 Major Improvements**: Application refinement and UX enhancements
  - ✅ **Groq API Integration**: Configured AI Assistant with secure API key management via Replit Secrets
  - ✅ **Fixed Plotly Warnings**: Replaced all deprecated `width='stretch'` with `use_container_width=True` in st.plotly_chart() calls
  - ✅ **Streamlined Workflow**: Removed redundant Anomaly Detection page (page 7) - anomaly detection already integrated in Column Analysis Outliers tab
  - ✅ **Redesigned Visualization Page**: Now provides actionable data quality insights with:
    * Overview metrics (missing data, analyzed columns, cleaning progress, quality scores)
    * Cleaning priorities sorted by urgency with recommendations
    * Missing data patterns analysis with action items
    * Before/after cleaning impact comparison
    * Optional custom visualizations (advanced users)
    * Clear next steps for workflow
  - ✅ **Professional Report Templates**: Improved executive summary with formal structure, tables, and professional language suitable for statistical agencies
  - ✅ **Improved Navigation**: Clean 6-page structure (Data Upload, Column Analysis, Cleaning Wizard, Visualization, AI Assistant, Reports)

- **2025-10-03**: GitHub Import to Replit Environment
  - Successfully imported project from GitHub
  - Installed Python 3.11 and all dependencies using uv package manager
  - All 60 Python packages installed successfully (streamlit, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib, groq, jinja2, scipy)
  - Verified Streamlit configuration (.streamlit/config.toml) is properly set up for Replit
  - Configured workflow to run on port 5000 with 0.0.0.0 binding
  - Set up autoscale deployment configuration
  - Application running successfully and tested
  - Note: GROQ_API_KEY environment variable is not set - AI Assistant features will not be available until configured

## Deployment
The application is configured for autoscale deployment, suitable for stateless web applications. It automatically scales based on traffic and stops when not in use.

## User Preferences
None configured yet.

## Notes
- The application uses a virtual environment (.pythonlibs/) for dependency isolation
- Data files (CSV, Excel) are gitignored to protect sensitive survey data
- The app supports weighted and unweighted statistical analysis
- All cleaning operations are logged with timestamps for audit trails

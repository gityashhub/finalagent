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
│   ├── 3_Cleaning_Wizard.py       # Data cleaning operations
│   ├── 4_AI_Assistant.py          # AI-powered guidance
│   └── 5_Reports.py               # Report generation
├── modules/                        # Core functionality modules
│   ├── ai_assistant.py            # AI assistant integration
│   ├── cleaning_engine.py         # Data cleaning methods
│   ├── data_analyzer.py           # Column analysis engine
│   ├── report_generator.py        # Report generation
│   ├── survey_weights.py          # Survey weights management
│   ├── utils.py                   # Utility functions
│   └── visualization.py           # Visualization components
├── pyproject.toml                 # Python dependencies
└── .gitignore                     # Git ignore rules
```

## Key Features
1. **Data Upload & Configuration**: Support for CSV and Excel files with automatic column type detection
2. **Column Analysis**: Individual column analysis with missing data patterns, outlier detection, and quality assessment
3. **Cleaning Wizard**: Multiple cleaning methods (imputation, outlier handling, standardization) with survey weight support
4. **AI Assistant**: Context-aware guidance using Groq API for cleaning recommendations
5. **Comprehensive Reporting**: Export analysis and cleaning reports in Markdown, HTML, or JSON formats
6. **Undo/Redo**: Full operation history with undo/redo functionality
7. **Survey Weights**: Integrated support for survey design weights in all analyses

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
- **2025-10-03**: Initial Replit environment setup
  - Installed Python 3.11 and all dependencies
  - Configured Streamlit workflow for port 5000
  - Set up deployment configuration (autoscale)
  - Created .gitignore for Python project

## Deployment
The application is configured for autoscale deployment, suitable for stateless web applications. It automatically scales based on traffic and stops when not in use.

## User Preferences
None configured yet.

## Notes
- The application uses a virtual environment (.pythonlibs/) for dependency isolation
- Data files (CSV, Excel) are gitignored to protect sensitive survey data
- The app supports weighted and unweighted statistical analysis
- All cleaning operations are logged with timestamps for audit trails

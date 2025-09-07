# Survey Data Cleaning Assistant

## Overview
This is an AI-powered Streamlit application designed specifically for statistical agencies to perform intelligent survey data cleaning operations. The application provides column-specific analysis and cleaning recommendations with comprehensive audit trails.

## Project Architecture
- **Frontend**: Streamlit web application (Python)
- **AI Integration**: Groq API for intelligent assistance
- **Data Processing**: Pandas, NumPy, Matplotlib, Plotly, Seaborn, Scikit-learn
- **Package Management**: UV for Python dependency management

## Key Features
- Individual column analysis with tailored recommendations
- AI-powered assistance using advanced language models
- Multiple cleaning strategies for missing values, outliers, and inconsistencies
- Comprehensive audit trail with undo/redo functionality
- Statistical rigor for survey data methodology
- Survey weights management
- Comprehensive reporting system

## Project Structure
```
├── app.py                    # Main Streamlit application
├── modules/                  # Core application modules
│   ├── ai_assistant.py      # AI-powered assistant using Groq
│   ├── cleaning_engine.py   # Data cleaning operations
│   ├── data_analyzer.py     # Column analysis and statistics
│   ├── report_generator.py  # Report generation
│   ├── survey_weights.py    # Survey weights management
│   ├── utils.py            # Utility functions
│   └── visualization.py    # Data visualization components
├── pages/                   # Streamlit pages
│   ├── 1_Data_Upload.py    # Dataset upload and configuration
│   ├── 2_Column_Analysis.py # Individual column analysis
│   ├── 3_Cleaning_Wizard.py # Guided cleaning operations
│   ├── 4_AI_Assistant.py   # AI assistant interface
│   ├── 5_Reports.py        # Report generation and export
│   └── 6_Survey_Weights.py # Survey weights management
├── pyproject.toml          # Python dependencies
└── uv.lock                 # Locked dependencies
```

## Setup and Configuration

### Environment Setup
The application is configured to run with:
- Python 3.11+ via UV package manager
- Streamlit server on port 5000 (0.0.0.0:5000)
- All required dependencies automatically managed

### AI Assistant (Optional)
For full AI functionality, set the `GROQ_API_KEY` secret:
- The AI assistant provides intelligent guidance for data cleaning
- Works with column-specific context and statistical methodology
- Uses Llama-3.1-8b-instant model via Groq API
- Application works without API key but AI features will be disabled

### Development Workflow
- **Start Application**: Workflow automatically starts Streamlit on port 5000
- **Development**: All changes auto-reload via Streamlit
- **Dependencies**: Managed via `uv sync` from pyproject.toml

### Deployment
- **Target**: Autoscale deployment for stateless web application
- **Production Command**: `uv run streamlit run app.py --server.address 0.0.0.0 --server.port 5000`
- **Environment**: Optimized for statistical agency workflows

## Recent Changes
- **2025-09-07**: Initial project import and Replit environment setup
  - Configured Python dependencies via UV
  - Set up Streamlit workflow on port 5000
  - Verified application functionality
  - Configured autoscale deployment
  - AI assistant ready for GROQ_API_KEY configuration

## User Preferences
- Clean, professional interface suitable for statistical agencies
- Focus on methodological soundness and data integrity
- Educational explanations of statistical concepts
- Comprehensive audit trails for reproducibility

## Usage Notes
1. Upload dataset via the Data Upload page
2. Review and adjust column type detection
3. Analyze columns individually with detailed statistics
4. Apply cleaning methods using the guided wizard
5. Get AI assistance for expert guidance (requires API key)
6. Generate comprehensive reports of cleaning operations
7. Manage survey weights if applicable

The application maintains statistical rigor appropriate for survey data used by government and statistical agencies.
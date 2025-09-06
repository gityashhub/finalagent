# Overview

The Intelligent Survey Data Cleaning Assistant is a Streamlit-based web application designed specifically for statistical agencies to perform AI-powered data cleaning operations. The application emphasizes individual column analysis rather than applying blanket cleaning methods across datasets, providing context-specific recommendations and maintaining comprehensive audit trails for survey data processing workflows.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit with multi-page application structure
- **Navigation**: Five main pages (Data Upload, Column Analysis, Cleaning Wizard, AI Assistant, Reports)
- **State Management**: Streamlit session state for maintaining dataset, analysis results, and cleaning history
- **Visualization**: Plotly for interactive charts and Matplotlib/Seaborn for statistical plots

## Backend Architecture
- **Modular Design**: Separate modules for core functionality (utils, data_analyzer, cleaning_engine, visualization, ai_assistant, report_generator)
- **Column-Centric Processing**: Individual column analysis engine that treats each column uniquely
- **Cleaning Engine**: Multiple cleaning strategies per data type (missing values, outliers, data quality issues)
- **Analysis Engine**: Comprehensive statistical analysis including distribution analysis, outlier detection, and relationship mapping

## Data Processing Strategy
- **Per-Column Intelligence**: Each column analyzed separately with tailored cleaning recommendations
- **Multiple Detection Methods**: IQR, Z-score, Modified Z-score for outliers; pattern-based missing value analysis
- **Context-Aware Recommendations**: Cleaning suggestions based on data type, missing percentage, outlier severity, and column relationships
- **Operation Tracking**: Full undo/redo functionality with operation history and audit trails

## AI Integration
- **Primary AI Service**: Groq API with llama-3.1-8b-instant model for conversational assistance
- **Fallback Strategy**: HuggingFace transformers for offline capability
- **Context Management**: AI maintains awareness of current dataset state and specific column analysis
- **Educational Role**: Provides explanations of statistical concepts and methodology reasoning

# External Dependencies

## AI Services
- **Groq API**: Primary conversational AI service using llama-3.1-8b-instant model for data cleaning guidance
- **HuggingFace Transformers**: Fallback AI service for offline operation

## Data Processing Libraries
- **Pandas**: Core data manipulation and analysis
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Machine learning algorithms for imputation (KNN), outlier detection (Isolation Forest), and preprocessing
- **SciPy**: Statistical functions and advanced mathematical operations

## Visualization Dependencies
- **Plotly**: Interactive plotting and dashboard components
- **Matplotlib/Seaborn**: Statistical visualizations and distribution plots
- **Streamlit**: Web framework for the entire application interface

## Utility Libraries
- **Jinja2**: Template engine for report generation
- **JSON**: Configuration management and data serialization
- **Base64/IO**: File handling and export functionality
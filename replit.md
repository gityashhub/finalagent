# Survey Data Cleaning Assistant

## Overview
This is an AI-powered survey data cleaning tool designed specifically for statistical agencies. The application provides intelligent, context-aware cleaning recommendations for survey datasets with comprehensive audit trails and statistical rigor.

## Recent Changes
- September 8, 2025: Initial setup for Replit environment
  - Configured UV package manager dependencies
  - Set up Streamlit configuration for Replit proxy (port 5000, all hosts allowed)
  - Fixed syntax error in Reports page (positional/keyword argument ordering)
  - Configured deployment settings for autoscale

## Project Architecture
- **Backend**: Python 3.11 with Streamlit framework
- **Package Manager**: UV for fast dependency management
- **Structure**: Multi-page Streamlit app with modular components
- **Key Pages**:
  - Data Upload: Dataset import and configuration
  - Column Analysis: Individual column examination
  - Cleaning Wizard: Interactive data cleaning with weights
  - AI Assistant: Context-aware guidance using Groq API
  - Reports: Comprehensive cleaning documentation

## Key Features
- Individual column analysis with tailored recommendations
- AI-powered assistance using advanced language models
- Multiple cleaning strategies for missing values and outliers
- Comprehensive audit trail with undo/redo functionality
- Statistical rigor maintenance for survey data
- Weight integration for statistical accuracy
- Multiple export formats (HTML, JSON, Markdown)

## Dependencies
Core packages managed via UV:
- streamlit: Web application framework
- pandas, numpy: Data manipulation
- scikit-learn, scipy: Statistical analysis
- matplotlib, seaborn, plotly: Visualization
- groq: AI assistance API
- jinja2: Template rendering

## Configuration
- Port: 5000 (required for Replit)
- Host: 0.0.0.0 (allows proxy access)
- CORS disabled for Replit compatibility
- Deployment: Autoscale (stateless web app)

## Current State
Fully functional and deployed. All modules working correctly with proper error handling and user guidance.
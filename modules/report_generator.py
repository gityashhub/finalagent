import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from jinja2 import Template
import base64
import io

class ReportGenerator:
    """Comprehensive report generation for data cleaning operations"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._get_executive_template(),
            'detailed_analysis': self._get_detailed_template(),
            'methodology': self._get_methodology_template(),
            'audit_trail': self._get_audit_template()
        }
    
    def generate_executive_summary(self, df: pd.DataFrame, cleaning_history: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary report"""
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(df, cleaning_history, analysis_results)
        
        template = Template(self.report_templates['executive_summary'])
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_shape=df.shape,
            summary_stats=summary_stats,
            cleaning_history=cleaning_history
        )
    
    def generate_detailed_analysis(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                                 cleaning_history: Dict[str, Any]) -> str:
        """Generate detailed column-by-column analysis report"""
        
        template = Template(self.report_templates['detailed_analysis'])
        
        # Prepare column details
        column_details = []
        for col, analysis in analysis_results.items():
            column_detail = {
                'name': col,
                'analysis': analysis,
                'cleaning_applied': cleaning_history.get(col, [])
            }
            column_details.append(column_detail)
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_shape=df.shape,
            column_details=column_details
        )
    
    def generate_methodology_report(self, cleaning_history: Dict[str, Any]) -> str:
        """Generate methodology documentation"""
        
        template = Template(self.report_templates['methodology'])
        
        # Extract unique methods used
        methods_used = set()
        for col_history in cleaning_history.values():
            for operation in col_history:
                methods_used.add(operation.get('method_name', 'unknown'))
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cleaning_history=cleaning_history,
            methods_used=list(methods_used)
        )
    
    def generate_audit_trail(self, cleaning_history: Dict[str, Any]) -> str:
        """Generate complete audit trail"""
        
        template = Template(self.report_templates['audit_trail'])
        
        # Create chronological audit trail
        all_operations = []
        for column, operations in cleaning_history.items():
            for op in operations:
                op['column'] = column
                all_operations.append(op)
        
        # Sort by timestamp
        all_operations.sort(key=lambda x: x.get('timestamp', ''))
        
        return template.render(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            all_operations=all_operations
        )
    
    def generate_complete_report(self, df: pd.DataFrame, original_df: pd.DataFrame,
                               analysis_results: Dict[str, Any], 
                               cleaning_history: Dict[str, Any]) -> Dict[str, str]:
        """Generate all report types"""
        
        reports = {}
        
        try:
            reports['executive_summary'] = self.generate_executive_summary(
                df, cleaning_history, analysis_results
            )
        except Exception as e:
            reports['executive_summary'] = f"Error generating executive summary: {str(e)}"
        
        try:
            reports['detailed_analysis'] = self.generate_detailed_analysis(
                df, analysis_results, cleaning_history
            )
        except Exception as e:
            reports['detailed_analysis'] = f"Error generating detailed analysis: {str(e)}"
        
        try:
            reports['methodology'] = self.generate_methodology_report(cleaning_history)
        except Exception as e:
            reports['methodology'] = f"Error generating methodology report: {str(e)}"
        
        try:
            reports['audit_trail'] = self.generate_audit_trail(cleaning_history)
        except Exception as e:
            reports['audit_trail'] = f"Error generating audit trail: {str(e)}"
        
        return reports
    
    def _calculate_summary_stats(self, df: pd.DataFrame, cleaning_history: Dict[str, Any],
                               analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the report"""
        
        total_operations = sum(len(ops) for ops in cleaning_history.values())
        columns_cleaned = len(cleaning_history)
        
        # Data quality scores
        quality_scores = []
        for col, analysis in analysis_results.items():
            if 'data_quality' in analysis:
                quality_scores.append(analysis['data_quality'].get('score', 0))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Missing data summary
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
        
        return {
            'total_operations': total_operations,
            'columns_cleaned': columns_cleaned,
            'avg_quality_score': round(avg_quality, 2),
            'total_missing': total_missing,
            'missing_percentage': round(missing_percentage, 2),
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
    
    def _get_executive_template(self) -> str:
        """Executive summary template"""
        return """
# Data Cleaning Executive Summary

**Report Generated:** {{ report_date }}
**Dataset Dimensions:** {{ dataset_shape[0] }} rows × {{ dataset_shape[1] }} columns

## Summary Statistics

- **Total Cleaning Operations:** {{ summary_stats.total_operations }}
- **Columns Cleaned:** {{ summary_stats.columns_cleaned }} / {{ summary_stats.total_columns }}
- **Average Data Quality Score:** {{ summary_stats.avg_quality_score }}/100
- **Missing Data:** {{ summary_stats.total_missing }} values ({{ summary_stats.missing_percentage }}%)

## Key Achievements

{% if summary_stats.avg_quality_score >= 80 %}
✅ **High Data Quality:** Average quality score of {{ summary_stats.avg_quality_score }} indicates excellent data quality.
{% elif summary_stats.avg_quality_score >= 60 %}
⚠️ **Moderate Data Quality:** Average quality score of {{ summary_stats.avg_quality_score }} indicates room for improvement.
{% else %}
❌ **Low Data Quality:** Average quality score of {{ summary_stats.avg_quality_score }} indicates significant quality issues.
{% endif %}

{% if summary_stats.missing_percentage < 5 %}
✅ **Low Missing Data:** Only {{ summary_stats.missing_percentage }}% missing values.
{% elif summary_stats.missing_percentage < 15 %}
⚠️ **Moderate Missing Data:** {{ summary_stats.missing_percentage }}% missing values handled.
{% else %}
❌ **High Missing Data:** {{ summary_stats.missing_percentage }}% missing values - significant imputation required.
{% endif %}

## Cleaning Operations Summary

{% for column, operations in cleaning_history.items() %}
**{{ column }}:** {{ operations|length }} operation(s) applied
{% endfor %}

## Recommendations

1. **Continue Monitoring:** Regularly assess data quality for new data ingestion
2. **Validate Results:** Review cleaned data for consistency with business rules
3. **Document Methodology:** Maintain clear documentation of cleaning procedures
4. **Automate Processes:** Consider automating successful cleaning procedures

---
*This report was generated by the Intelligent Survey Data Cleaning Assistant*
"""
    
    def _get_detailed_template(self) -> str:
        """Detailed analysis template"""
        return """
# Detailed Data Cleaning Analysis Report

**Report Generated:** {{ report_date }}
**Dataset Dimensions:** {{ dataset_shape[0] }} rows × {{ dataset_shape[1] }} columns

## Column-by-Column Analysis

{% for column_detail in column_details %}
## Column: {{ column_detail.name }}

### Basic Information
- **Data Type:** {{ column_detail.analysis.basic_info.dtype }}
- **Total Values:** {{ column_detail.analysis.basic_info.count }}
- **Missing Values:** {{ column_detail.analysis.basic_info.missing_count }} ({{ "%.2f"|format(column_detail.analysis.basic_info.missing_percentage) }}%)
- **Unique Values:** {{ column_detail.analysis.basic_info.unique_count }} ({{ "%.2f"|format(column_detail.analysis.basic_info.unique_percentage) }}%)

{% if column_detail.analysis.basic_info.get('mean') is not none %}
### Statistical Summary
- **Mean:** {{ "%.3f"|format(column_detail.analysis.basic_info.mean) }}
- **Median:** {{ "%.3f"|format(column_detail.analysis.basic_info.median) }}
- **Standard Deviation:** {{ "%.3f"|format(column_detail.analysis.basic_info.std) }}
- **Range:** {{ "%.3f"|format(column_detail.analysis.basic_info.min) }} to {{ "%.3f"|format(column_detail.analysis.basic_info.max) }}
{% endif %}

### Data Quality Assessment
- **Quality Score:** {{ column_detail.analysis.data_quality.score }}/100 (Grade: {{ column_detail.analysis.data_quality.grade }})
{% if column_detail.analysis.data_quality.issues %}
- **Issues Identified:**
{% for issue in column_detail.analysis.data_quality.issues %}
  - {{ issue }}
{% endfor %}
{% endif %}

### Missing Data Analysis
- **Pattern Type:** {{ column_detail.analysis.missing_analysis.get('pattern_type', 'N/A') }}
{% if column_detail.analysis.missing_analysis.get('max_consecutive', 0) > 0 %}
- **Max Consecutive Missing:** {{ column_detail.analysis.missing_analysis.max_consecutive }}
{% endif %}

{% if column_detail.analysis.outlier_analysis.get('method_results') %}
### Outlier Detection Results
{% for method, results in column_detail.analysis.outlier_analysis.method_results.items() %}
- **{{ results.method }}:** {{ results.outlier_count }} outliers ({{ "%.2f"|format(results.outlier_percentage) }}%)
{% endfor %}
{% endif %}

### Cleaning Operations Applied
{% if column_detail.cleaning_applied %}
{% for operation in column_detail.cleaning_applied %}
- **{{ operation.timestamp }}:** Applied {{ operation.get('method_name', 'unknown method') }}
  - Result: {{ operation.get('result', 'N/A') }}
{% if operation.get('impact_stats') %}
  - Rows affected: {{ operation.impact_stats.rows_affected }}
{% endif %}
{% endfor %}
{% else %}
- No cleaning operations applied to this column
{% endif %}

---
{% endfor %}

*End of detailed analysis*
"""
    
    def _get_methodology_template(self) -> str:
        """Methodology documentation template"""
        return """
# Data Cleaning Methodology Report

**Report Generated:** {{ report_date }}

## Methodology Overview

This report documents the systematic approach used for cleaning the survey dataset, ensuring reproducibility and transparency of all data processing operations.

## Cleaning Methods Applied

{% for method in methods_used %}
### {{ method|title|replace('_', ' ') }}

{% if method == 'median_imputation' %}
**Purpose:** Replace missing values with the median of the observed values
**Rationale:** Robust to outliers and appropriate for skewed distributions
**Applicability:** Numeric columns with moderate missing data rates
{% elif method == 'mean_imputation' %}
**Purpose:** Replace missing values with the mean of the observed values
**Rationale:** Simple and preserves the overall distribution center
**Applicability:** Numeric columns with normal distribution and low missing rates
{% elif method == 'mode_imputation' %}
**Purpose:** Replace missing values with the most frequent value
**Rationale:** Preserves the dominant category in categorical data
**Applicability:** Categorical columns with clear mode
{% elif method == 'knn_imputation' %}
**Purpose:** Impute missing values using K-Nearest Neighbors algorithm
**Rationale:** Preserves relationships between variables
**Applicability:** Numeric columns with available correlated features
{% elif method == 'winsorization' %}
**Purpose:** Cap extreme values at specified percentiles
**Rationale:** Reduces outlier impact while preserving sample size
**Applicability:** Numeric columns with extreme outliers
{% elif method == 'iqr_removal' %}
**Purpose:** Remove outliers beyond 1.5 * IQR from quartiles
**Rationale:** Statistical definition of outliers for normal distributions
**Applicability:** Numeric columns with approximately normal distribution
{% endif %}

{% endfor %}

## Column-Specific Applications

{% for column, operations in cleaning_history.items() %}
### {{ column }}

{% for operation in operations %}
**Operation {{ loop.index }}:** {{ operation.get('method_name', 'Unknown') }}
- **Timestamp:** {{ operation.get('timestamp', 'N/A') }}
- **Rationale:** {{ operation.get('rationale', 'Applied based on column analysis') }}
{% if operation.get('parameters') %}
- **Parameters:** {{ operation.parameters }}
{% endif %}
{% if operation.get('impact_stats') %}
- **Impact:** {{ operation.impact_stats.rows_affected }} rows affected
{% endif %}

{% endfor %}
{% endfor %}

## Quality Assurance

1. **Pre-cleaning Analysis:** Each column was individually analyzed for patterns, outliers, and data quality issues
2. **Method Selection:** Cleaning methods were selected based on column-specific characteristics, not generic rules
3. **Impact Assessment:** All operations were evaluated for their statistical impact
4. **Audit Trail:** Complete documentation of all operations for reproducibility

## Statistical Considerations

- **Survey Weights:** All cleaning operations preserve the integrity of survey weights
- **Sampling Design:** Methods were chosen considering the complex survey design
- **Bias Reduction:** Imputation methods were selected to minimize introduction of bias
- **Variance Preservation:** Methods that maintain appropriate variance were prioritized

## Limitations and Assumptions

1. Missing data patterns were assumed to be missing at random (MAR) unless evidence suggested otherwise
2. Outlier detection assumed underlying normal distribution where applicable
3. Relationships between variables were assumed stable during imputation processes

---
*This methodology ensures reproducible and statistically sound data cleaning procedures*
"""
    
    def _get_audit_template(self) -> str:
        """Audit trail template"""
        return """
# Data Cleaning Audit Trail

**Report Generated:** {{ report_date }}

## Complete Operation Log

This section provides a chronological record of all data cleaning operations performed on the dataset.

{% for operation in all_operations %}
### Operation {{ loop.index }}

**Timestamp:** {{ operation.get('timestamp', 'N/A') }}
**Column:** {{ operation.get('column', 'N/A') }}
**Method:** {{ operation.get('method_name', 'N/A') }}
**Type:** {{ operation.get('method_type', 'N/A') }}

{% if operation.get('parameters') %}
**Parameters:**
{% for key, value in operation.parameters.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if operation.get('impact_stats') %}
**Impact Statistics:**
- Rows affected: {{ operation.impact_stats.rows_affected }}
- Percentage changed: {{ "%.2f"|format(operation.impact_stats.percentage_changed) }}%
- Missing before: {{ operation.impact_stats.missing_before }}
- Missing after: {{ operation.impact_stats.missing_after }}
- Missing change: {{ operation.impact_stats.missing_change }}

{% if operation.impact_stats.get('mean_before') is not none %}
**Statistical Impact:**
- Mean before: {{ "%.3f"|format(operation.impact_stats.mean_before) }}
- Mean after: {{ "%.3f"|format(operation.impact_stats.mean_after) }}
- Std before: {{ "%.3f"|format(operation.impact_stats.std_before) }}
- Std after: {{ "%.3f"|format(operation.impact_stats.std_after) }}
{% endif %}
{% endif %}

**Result:** {{ operation.get('result', 'Operation completed') }}

---
{% endfor %}

## Summary Statistics

- **Total Operations:** {{ all_operations|length }}
- **Unique Columns Affected:** {{ all_operations|map(attribute='column')|list|unique|length }}
- **Operation Types Used:** {{ all_operations|map(attribute='method_type')|list|unique|join(', ') }}

## Verification

All operations listed above can be independently verified and reproduced using the same parameters and methods on the original dataset.

---
*Complete audit trail for data cleaning operations*
"""
    
    def export_to_html(self, reports: Dict[str, str], title: str = "Data Cleaning Report") -> str:
        """Export reports to HTML format"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; }
        h3 { color: #7f8c8d; }
        .report-section { margin-bottom: 50px; }
        .metadata { background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    {% for report_type, content in reports.items() %}
    <div class="report-section">
        <h2>{{ report_type|title|replace('_', ' ') }}</h2>
        <div>{{ content|markdown }}</div>
    </div>
    {% endfor %}
    
    <div class="metadata">
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Tool:</strong> Intelligent Survey Data Cleaning Assistant</p>
    </div>
</body>
</html>
"""
        
        template = Template(html_template)
        return template.render(
            title=title,
            reports=reports,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def export_to_json(self, reports: Dict[str, str], metadata: Dict[str, Any] = None) -> str:
        """Export reports to JSON format"""
        
        export_data = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'tool': 'Intelligent Survey Data Cleaning Assistant',
                'version': '1.0.0'
            },
            'reports': reports
        }
        
        if metadata:
            export_data['metadata'].update(metadata)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)

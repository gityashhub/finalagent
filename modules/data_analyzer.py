import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class ColumnAnalyzer:
    """Individual column analysis engine with multiple detection methods"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_column(self, df: pd.DataFrame, column: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Comprehensive analysis of a single column"""
        cache_key = f"{column}_{len(df)}_{df[column].isnull().sum()}"
        
        if not force_refresh and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        series = df[column]
        analysis = {
            'column_name': column,
            'basic_info': self._get_basic_info(series),
            'missing_analysis': self._analyze_missing_patterns(df, column),
            'outlier_analysis': self._detect_outliers(series),
            'distribution_analysis': self._analyze_distribution(series),
            'data_quality': self._assess_data_quality(series),
            'relationships': self._analyze_relationships(df, column),
            'cleaning_recommendations': []
        }
        
        # Generate specific cleaning recommendations
        analysis['cleaning_recommendations'] = self._generate_cleaning_recommendations(analysis)
        
        self.analysis_cache[cache_key] = analysis
        return analysis
    
    def _get_basic_info(self, series: pd.Series) -> Dict[str, Any]:
        """Get basic information about the column"""
        info = {
            'dtype': str(series.dtype),
            'count': len(series),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
            'memory_usage': series.memory_usage(deep=True)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                info.update({
                    'mean': non_null_series.mean(),
                    'median': non_null_series.median(),
                    'std': non_null_series.std(),
                    'min': non_null_series.min(),
                    'max': non_null_series.max(),
                    'q25': non_null_series.quantile(0.25),
                    'q75': non_null_series.quantile(0.75),
                    'skewness': stats.skew(non_null_series),
                    'kurtosis': stats.kurtosis(non_null_series)
                })
        
        return info
    
    def _analyze_missing_patterns(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze missing data patterns for the specific column"""
        series = df[column]
        missing_mask = series.isnull()
        
        if missing_mask.sum() == 0:
            return {'pattern_type': 'none', 'analysis': 'No missing values found'}
        
        analysis = {
            'total_missing': missing_mask.sum(),
            'percentage': (missing_mask.sum() / len(series)) * 100,
            'pattern_type': 'unknown',
            'consecutive_missing': [],
            'missing_by_position': {}
        }
        
        # Find consecutive missing values
        consecutive = []
        current_streak = 0
        for is_missing in missing_mask:
            if is_missing:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consecutive.append(current_streak)
        
        analysis['consecutive_missing'] = consecutive
        analysis['max_consecutive'] = max(consecutive) if consecutive else 0
        
        # Analyze missing pattern type
        if analysis['percentage'] < 5:
            analysis['pattern_type'] = 'sporadic'
        elif analysis['max_consecutive'] > len(series) * 0.1:
            analysis['pattern_type'] = 'systematic_blocks'
        elif missing_mask.iloc[:int(len(series) * 0.1)].sum() > missing_mask.iloc[int(len(series) * 0.9):].sum() * 2:
            analysis['pattern_type'] = 'front_loaded'
        elif missing_mask.iloc[int(len(series) * 0.9):].sum() > missing_mask.iloc[:int(len(series) * 0.1)].sum() * 2:
            analysis['pattern_type'] = 'tail_loaded'
        else:
            analysis['pattern_type'] = 'random'
        
        return analysis
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        if not pd.api.types.is_numeric_dtype(series):
            return {'method_results': {}, 'summary': 'Not applicable for non-numeric data'}
        
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            return {'method_results': {}, 'summary': 'Insufficient data for outlier detection'}
        
        outlier_results = {}
        
        # IQR Method
        Q1 = non_null_series.quantile(0.25)
        Q3 = non_null_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
        outlier_results['iqr'] = {
            'method': 'Interquartile Range (IQR)',
            'outlier_count': len(iqr_outliers),
            'outlier_percentage': (len(iqr_outliers) / len(non_null_series)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': iqr_outliers.tolist()[:20]  # Limit to 20 for display
        }
        
        # Z-Score Method
        z_scores = np.abs(stats.zscore(non_null_series))
        z_outliers = non_null_series[z_scores > 3]
        outlier_results['zscore'] = {
            'method': 'Z-Score (|z| > 3)',
            'outlier_count': len(z_outliers),
            'outlier_percentage': (len(z_outliers) / len(non_null_series)) * 100,
            'threshold': 3,
            'outlier_values': z_outliers.tolist()[:20]
        }
        
        # Modified Z-Score Method
        median = np.median(non_null_series)
        mad = np.median(np.abs(non_null_series - median))
        modified_z_scores = 0.6745 * (non_null_series - median) / mad if mad != 0 else np.zeros(len(non_null_series))
        modified_z_outliers = non_null_series[np.abs(modified_z_scores) > 3.5]
        outlier_results['modified_zscore'] = {
            'method': 'Modified Z-Score (|Mz| > 3.5)',
            'outlier_count': len(modified_z_outliers),
            'outlier_percentage': (len(modified_z_outliers) / len(non_null_series)) * 100,
            'threshold': 3.5,
            'outlier_values': modified_z_outliers.tolist()[:20]
        }
        
        # Statistical summary
        total_outliers = set()
        for method_result in outlier_results.values():
            total_outliers.update(method_result['outlier_values'])
        
        summary = {
            'methods_agree': len(set(iqr_outliers.tolist()) & set(z_outliers.tolist())) > 0,
            'consensus_outliers': len(total_outliers),
            'severity': 'high' if max([r['outlier_percentage'] for r in outlier_results.values()]) > 10 else 'moderate' if max([r['outlier_percentage'] for r in outlier_results.values()]) > 5 else 'low'
        }
        
        return {
            'method_results': outlier_results,
            'summary': summary
        }
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution characteristics of the column"""
        if not pd.api.types.is_numeric_dtype(series):
            # For categorical data
            value_counts = series.value_counts()
            return {
                'type': 'categorical',
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequency_distribution': value_counts.head(10).to_dict(),
                'entropy': stats.entropy(value_counts.values)
            }
        
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            return {'type': 'insufficient_data'}
        
        # Statistical tests for normality
        shapiro_stat, shapiro_p = stats.shapiro(non_null_series.sample(min(5000, len(non_null_series))))
        
        analysis = {
            'type': 'numeric',
            'skewness': stats.skew(non_null_series),
            'kurtosis': stats.kurtosis(non_null_series),
            'normality_test': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        }
        
        # Distribution characterization
        skew_val = abs(analysis['skewness'])
        if skew_val < 0.5:
            analysis['distribution_shape'] = 'approximately_normal'
        elif skew_val < 1:
            analysis['distribution_shape'] = 'moderately_skewed'
        else:
            analysis['distribution_shape'] = 'highly_skewed'
        
        return analysis
    
    def _assess_data_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess overall data quality for the column"""
        quality_score = 100
        issues = []
        
        # Missing data penalty
        missing_pct = (series.isnull().sum() / len(series)) * 100
        if missing_pct > 50:
            quality_score -= 40
            issues.append(f"High missing data rate ({missing_pct:.1f}%)")
        elif missing_pct > 20:
            quality_score -= 20
            issues.append(f"Moderate missing data rate ({missing_pct:.1f}%)")
        elif missing_pct > 5:
            quality_score -= 10
            issues.append(f"Low missing data rate ({missing_pct:.1f}%)")
        
        # Data type consistency
        if series.dtype == 'object':
            # Check for mixed types in object columns
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                sample_types = set(type(x).__name__ for x in non_null_series.sample(min(1000, len(non_null_series))))
                if len(sample_types) > 1:
                    quality_score -= 15
                    issues.append("Mixed data types detected")
        
        # Uniqueness assessment
        unique_pct = (series.nunique() / len(series)) * 100
        if unique_pct > 95 and len(series) > 100:
            issues.append("Very high uniqueness - possible identifier column")
        elif unique_pct < 1:
            quality_score -= 10
            issues.append("Very low uniqueness - mostly repeated values")
        
        return {
            'score': max(0, quality_score),
            'grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 80 else 'C' if quality_score >= 70 else 'D' if quality_score >= 60 else 'F',
            'issues': issues
        }
    
    def _analyze_relationships(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze relationships with other columns"""
        series = df[column]
        relationships = {}
        
        # Find numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if column in numeric_cols and len(numeric_cols) > 1:
            numeric_cols.remove(column)
            correlations = {}
            for col in numeric_cols[:10]:  # Limit to top 10 for performance
                corr = series.corr(df[col])
                if not pd.isna(corr) and abs(corr) > 0.1:
                    correlations[col] = corr
            
            relationships['correlations'] = dict(sorted(correlations.items(), 
                                                      key=lambda x: abs(x[1]), 
                                                      reverse=True)[:5])
        
        return relationships
    
    def _generate_cleaning_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate context-specific cleaning recommendations for the column"""
        recommendations = []
        
        basic_info = analysis['basic_info']
        missing_analysis = analysis['missing_analysis']
        outlier_analysis = analysis['outlier_analysis']
        quality = analysis['data_quality']
        
        # Missing value recommendations
        if missing_analysis.get('percentage', 0) > 0:
            missing_pct = missing_analysis['percentage']
            pattern_type = missing_analysis.get('pattern_type', 'unknown')
            
            if missing_pct < 5:
                if pd.api.types.is_numeric_dtype(pd.Series(dtype=basic_info['dtype'])):
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'median_imputation',
                        'priority': 'high',
                        'description': 'Use median imputation for low missing rate in numeric column',
                        'pros': ['Simple', 'Robust to outliers', 'Preserves distribution center'],
                        'cons': ['Reduces variance', 'May not preserve relationships'],
                        'applicability_score': 85
                    })
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'knn_imputation',
                        'priority': 'medium',
                        'description': 'Use KNN imputation to preserve relationships',
                        'pros': ['Preserves relationships', 'More sophisticated'],
                        'cons': ['Computationally expensive', 'Sensitive to scaling'],
                        'applicability_score': 75
                    })
                else:
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'mode_imputation',
                        'priority': 'high',
                        'description': 'Use mode imputation for categorical column',
                        'pros': ['Preserves most common category', 'Simple'],
                        'cons': ['May increase bias toward common values'],
                        'applicability_score': 80
                    })
            
            elif missing_pct < 20:
                if pattern_type == 'systematic_blocks':
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'interpolation',
                        'priority': 'high',
                        'description': 'Use interpolation for systematic missing blocks',
                        'pros': ['Good for time series', 'Preserves trends'],
                        'cons': ['May not work for non-sequential data'],
                        'applicability_score': 70
                    })
                else:
                    recommendations.append({
                        'type': 'missing_values',
                        'method': 'regression_imputation',
                        'priority': 'medium',
                        'description': 'Use regression-based imputation',
                        'pros': ['Preserves relationships', 'More accurate'],
                        'cons': ['Complex', 'Requires related columns'],
                        'applicability_score': 75
                    })
            else:
                recommendations.append({
                    'type': 'missing_values',
                    'method': 'missing_category',
                    'priority': 'high',
                    'description': 'Create "Missing" category for high missing rate',
                    'pros': ['Preserves all data', 'Explicit about missingness'],
                    'cons': ['Changes interpretation', 'May need special handling'],
                    'applicability_score': 90
                })
        
        # Outlier recommendations
        if 'method_results' in outlier_analysis and outlier_analysis['method_results']:
            severity = outlier_analysis.get('summary', {}).get('severity', 'low')
            
            if severity == 'high':
                recommendations.append({
                    'type': 'outliers',
                    'method': 'winsorization',
                    'priority': 'high',
                    'description': 'Use winsorization to cap extreme values',
                    'pros': ['Preserves sample size', 'Reduces extreme influence'],
                    'cons': ['Changes distribution', 'May mask important patterns'],
                    'applicability_score': 85
                })
                recommendations.append({
                    'type': 'outliers',
                    'method': 'removal',
                    'priority': 'medium',
                    'description': 'Remove outliers (use with caution)',
                    'pros': ['Clean dataset', 'Improves normality'],
                    'cons': ['Loses information', 'Reduces sample size'],
                    'applicability_score': 60
                })
            elif severity == 'moderate':
                recommendations.append({
                    'type': 'outliers',
                    'method': 'log_transformation',
                    'priority': 'medium',
                    'description': 'Use log transformation to reduce outlier impact',
                    'pros': ['Natural approach', 'Improves normality'],
                    'cons': ['Changes interpretation', 'Requires positive values'],
                    'applicability_score': 75
                })
        
        # Data quality recommendations
        if quality['score'] < 70:
            for issue in quality['issues']:
                if 'mixed data types' in issue.lower():
                    recommendations.append({
                        'type': 'data_quality',
                        'method': 'type_standardization',
                        'priority': 'high',
                        'description': 'Standardize data types in column',
                        'pros': ['Consistent processing', 'Prevents errors'],
                        'cons': ['May lose information', 'Manual review needed'],
                        'applicability_score': 95
                    })
        
        # Sort recommendations by priority and applicability
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['applicability_score']), reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations

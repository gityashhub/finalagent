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
            'rule_violations': self._detect_rule_violations(series, column),
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
            return {
                'total_missing': 0,
                'percentage': 0,
                'pattern_type': 'none', 
                'consecutive_missing': [],
                'missing_by_position': {},
                'max_consecutive': 0,
                'analysis': 'No missing values found'
            }
        
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
            return {
                'method_results': {}, 
                'summary': {
                    'methods_agree': False,
                    'consensus_outliers': 0,
                    'severity': 'low',
                    'analysis': 'Not applicable for non-numeric data'
                }
            }
        
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            return {
                'method_results': {}, 
                'summary': {
                    'methods_agree': False,
                    'consensus_outliers': 0,
                    'severity': 'low',
                    'analysis': 'Insufficient data for outlier detection'
                }
            }
        
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
            outlier_values = method_result.get('outlier_values', [])
            if outlier_values:
                total_outliers.update(outlier_values)
        
        # Calculate severity based on outlier percentages
        outlier_percentages = [r.get('outlier_percentage', 0) for r in outlier_results.values()]
        max_percentage = max(outlier_percentages) if outlier_percentages else 0
        
        summary = {
            'methods_agree': len(set(iqr_outliers.tolist()) & set(z_outliers.tolist())) > 0,
            'consensus_outliers': len(total_outliers),
            'severity': 'high' if max_percentage > 10 else 'moderate' if max_percentage > 5 else 'low'
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
                'entropy': stats.entropy(value_counts.values) if len(value_counts) > 0 else 0
            }
        
        non_null_series = series.dropna()
        if len(non_null_series) < 10:
            return {'type': 'insufficient_data'}
        
        # Statistical tests for normality (limit sample size for performance)
        sample_size = min(5000, len(non_null_series))
        sample_data = non_null_series.sample(sample_size) if len(non_null_series) > sample_size else non_null_series
        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
        
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
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
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
                try:
                    corr = series.corr(df[col])
                    if pd.notna(corr) and abs(corr) > 0.1:
                        correlations[col] = corr
                except Exception:
                    continue
            
            relationships['correlations'] = dict(sorted(correlations.items(), 
                                                      key=lambda x: abs(x[1]), 
                                                      reverse=True)[:5])
        
        return relationships
    
    def _detect_rule_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Detect rule-based violations in the column data"""
        violations = {
            'total_violations': 0,
            'violation_types': [],
            'severity': 'low',
            'details': {}
        }
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return violations
        
        # Numeric range violations
        if pd.api.types.is_numeric_dtype(series):
            violations.update(self._check_numeric_range_violations(non_null_series, column_name))
        
        # Text format violations
        elif series.dtype == 'object':
            violations.update(self._check_text_format_violations(non_null_series, column_name))
        
        # Categorical consistency violations
        violations.update(self._check_categorical_violations(non_null_series, column_name))
        
        # Determine overall severity
        if violations['total_violations'] > len(series) * 0.1:
            violations['severity'] = 'high'
        elif violations['total_violations'] > len(series) * 0.05:
            violations['severity'] = 'moderate'
        else:
            violations['severity'] = 'low'
            
        return violations
    
    def detect_inter_column_violations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect violations based on inter-column dependencies"""
        violations = {
            'total_violations': 0,
            'violation_types': [],
            'details': {},
            'severity': 'low'
        }
        
        # Check age vs birth_year consistency
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        birth_year_cols = [col for col in df.columns if 'birth_year' in col.lower() or 'birth' in col.lower()]
        
        if age_cols and birth_year_cols:
            from datetime import datetime
            current_year = datetime.now().year
            
            for age_col in age_cols:
                for birth_col in birth_year_cols:
                    if pd.api.types.is_numeric_dtype(df[age_col]) and pd.api.types.is_numeric_dtype(df[birth_col]):
                        # Calculate expected age from birth year
                        expected_ages = current_year - df[birth_col]
                        age_diff = abs(df[age_col] - expected_ages)
                        # Allow for 1-2 years difference (depending on birth month)
                        inconsistent_mask = age_diff > 2
                        
                        if inconsistent_mask.sum() > 0:
                            violations['total_violations'] += inconsistent_mask.sum()
                            violations['violation_types'].append(f'Age-birth year inconsistency ({age_col} vs {birth_col})')
                            violations['details'][f'{age_col}_{birth_col}_inconsistency'] = {
                                'count': inconsistent_mask.sum(),
                                'rule': 'Age should match birth year within 1-2 years',
                                'affected_rows': df.index[inconsistent_mask].tolist()[:10]
                            }
        
        # Check start_date vs end_date consistency
        start_cols = [col for col in df.columns if 'start' in col.lower() and ('date' in col.lower() or 'time' in col.lower())]
        end_cols = [col for col in df.columns if 'end' in col.lower() and ('date' in col.lower() or 'time' in col.lower())]
        
        for start_col in start_cols:
            for end_col in end_cols:
                try:
                    start_dates = pd.to_datetime(df[start_col], errors='coerce')
                    end_dates = pd.to_datetime(df[end_col], errors='coerce')
                    
                    # Check if end dates are before start dates
                    invalid_dates = (end_dates < start_dates) & start_dates.notna() & end_dates.notna()
                    
                    if invalid_dates.sum() > 0:
                        violations['total_violations'] += invalid_dates.sum()
                        violations['violation_types'].append(f'End date before start date ({start_col} vs {end_col})')
                        violations['details'][f'{start_col}_{end_col}_date_logic'] = {
                            'count': invalid_dates.sum(),
                            'rule': 'End date should be after start date',
                            'affected_rows': df.index[invalid_dates].tolist()[:10]
                        }
                except:
                    pass  # Skip if date conversion fails
        
        # Check income vs education level consistency
        income_cols = [col for col in df.columns if any(word in col.lower() for word in ['income', 'salary', 'wage'])]
        education_cols = [col for col in df.columns if 'education' in col.lower() or 'degree' in col.lower()]
        
        if income_cols and education_cols:
            for income_col in income_cols:
                for edu_col in education_cols:
                    if pd.api.types.is_numeric_dtype(df[income_col]) and df[edu_col].dtype == 'object':
                        # Simple check: higher education should generally correlate with higher income
                        education_levels = df[edu_col].str.lower()
                        high_education = education_levels.str.contains('master|phd|doctorate|graduate', na=False)
                        low_education = education_levels.str.contains('high school|elementary|primary', na=False)
                        
                        high_edu_low_income = high_education & (df[income_col] < df[income_col].quantile(0.25))
                        low_edu_high_income = low_education & (df[income_col] > df[income_col].quantile(0.75))
                        
                        total_anomalies = high_edu_low_income.sum() + low_edu_high_income.sum()
                        if total_anomalies > 0:
                            violations['total_violations'] += total_anomalies
                            violations['violation_types'].append(f'Education-income mismatch ({edu_col} vs {income_col})')
                            violations['details'][f'{edu_col}_{income_col}_mismatch'] = {
                                'count': total_anomalies,
                                'rule': 'Education level and income should generally correlate',
                                'high_edu_low_income': high_edu_low_income.sum(),
                                'low_edu_high_income': low_edu_high_income.sum()
                            }
        
        # Determine overall severity
        if violations['total_violations'] > len(df) * 0.1:
            violations['severity'] = 'high'
        elif violations['total_violations'] > len(df) * 0.05:
            violations['severity'] = 'moderate'
        else:
            violations['severity'] = 'low'
        
        return violations
    
    def _check_numeric_range_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for numeric range violations based on column type inference"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        # Infer expected ranges based on column name patterns
        column_lower = column_name.lower()
        
        # Age-related columns
        if any(keyword in column_lower for keyword in ['age', 'years_old', 'birth_year']):
            if 'age' in column_lower:
                invalid_ages = series[(series < 0) | (series > 120)]
                if len(invalid_ages) > 0:
                    violations['total_violations'] += len(invalid_ages)
                    violations['violation_types'].append('Invalid age range')
                    violations['details']['age_violations'] = {
                        'count': len(invalid_ages),
                        'rule': 'Age should be between 0 and 120',
                        'invalid_values': invalid_ages.tolist()[:10]
                    }
            
            elif 'birth_year' in column_lower:
                from datetime import datetime
                current_year = datetime.now().year
                invalid_years = series[(series < 1900) | (series > current_year)]
                if len(invalid_years) > 0:
                    violations['total_violations'] += len(invalid_years)
                    violations['violation_types'].append('Invalid birth year')
                    violations['details']['birth_year_violations'] = {
                        'count': len(invalid_years),
                        'rule': f'Birth year should be between 1900 and {current_year}',
                        'invalid_values': invalid_years.tolist()[:10]
                    }
        
        # Percentage columns
        elif any(keyword in column_lower for keyword in ['percent', 'percentage', 'rate', 'ratio']):
            invalid_percentages = series[(series < 0) | (series > 100)]
            if len(invalid_percentages) > 0:
                violations['total_violations'] += len(invalid_percentages)
                violations['violation_types'].append('Invalid percentage range')
                violations['details']['percentage_violations'] = {
                    'count': len(invalid_percentages),
                    'rule': 'Percentage should be between 0 and 100',
                    'invalid_values': invalid_percentages.tolist()[:10]
                }
        
        # Score columns
        elif any(keyword in column_lower for keyword in ['score', 'rating', 'grade']):
            # Check for impossible negative scores
            negative_scores = series[series < 0]
            if len(negative_scores) > 0:
                violations['total_violations'] += len(negative_scores)
                violations['violation_types'].append('Negative scores detected')
                violations['details']['negative_score_violations'] = {
                    'count': len(negative_scores),
                    'rule': 'Scores should not be negative',
                    'invalid_values': negative_scores.tolist()[:10]
                }
        
        # Income/salary columns
        elif any(keyword in column_lower for keyword in ['income', 'salary', 'wage', 'earnings']):
            negative_income = series[series < 0]
            if len(negative_income) > 0:
                violations['total_violations'] += len(negative_income)
                violations['violation_types'].append('Negative income values')
                violations['details']['income_violations'] = {
                    'count': len(negative_income),
                    'rule': 'Income should not be negative',
                    'invalid_values': negative_income.tolist()[:10]
                }
        
        return violations
    
    def _check_text_format_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for text format violations"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        column_lower = column_name.lower()
        
        # Email format validation
        if any(keyword in column_lower for keyword in ['email', 'e_mail', 'mail']):
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = series[~series.str.match(email_pattern, na=False)]
            if len(invalid_emails) > 0:
                violations['total_violations'] += len(invalid_emails)
                violations['violation_types'].append('Invalid email format')
                violations['details']['email_violations'] = {
                    'count': len(invalid_emails),
                    'rule': 'Email should follow valid email format',
                    'invalid_values': invalid_emails.tolist()[:10]
                }
        
        # Phone number format validation
        elif any(keyword in column_lower for keyword in ['phone', 'telephone', 'mobile']):
            import re
            # Basic phone number pattern (flexible for international formats)
            phone_pattern = r'^[\+]?[1-9][\d\s\-\(\)]{6,}$'
            invalid_phones = series[~series.str.match(phone_pattern, na=False)]
            if len(invalid_phones) > 0:
                violations['total_violations'] += len(invalid_phones)
                violations['violation_types'].append('Invalid phone format')
                violations['details']['phone_violations'] = {
                    'count': len(invalid_phones),
                    'rule': 'Phone number should follow valid format',
                    'invalid_values': invalid_phones.tolist()[:10]
                }
        
        # Postal/ZIP code validation
        elif any(keyword in column_lower for keyword in ['zip', 'postal', 'postcode']):
            import re
            # Basic postal code pattern (numbers and letters, 3-10 characters)
            postal_pattern = r'^[A-Z0-9\s\-]{3,10}$'
            invalid_postal = series[~series.str.upper().str.match(postal_pattern, na=False)]
            if len(invalid_postal) > 0:
                violations['total_violations'] += len(invalid_postal)
                violations['violation_types'].append('Invalid postal code format')
                violations['details']['postal_violations'] = {
                    'count': len(invalid_postal),
                    'rule': 'Postal code should be 3-10 alphanumeric characters',
                    'invalid_values': invalid_postal.tolist()[:10]
                }
        
        return violations
    
    def _check_categorical_violations(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Check for categorical consistency violations"""
        violations = {'total_violations': 0, 'violation_types': [], 'details': {}}
        
        column_lower = column_name.lower()
        
        # Gender consistency
        if any(keyword in column_lower for keyword in ['gender', 'sex']):
            valid_genders = {'male', 'm', 'female', 'f', 'other', 'non-binary', 'prefer not to say', 'unknown'}
            if series.dtype == 'object':
                invalid_genders = series[~series.str.lower().isin(valid_genders)]
                if len(invalid_genders) > 0:
                    violations['total_violations'] += len(invalid_genders)
                    violations['violation_types'].append('Invalid gender values')
                    violations['details']['gender_violations'] = {
                        'count': len(invalid_genders),
                        'rule': 'Gender should be from standard categories',
                        'invalid_values': invalid_genders.tolist()[:10]
                    }
        
        # Yes/No questions
        elif any(keyword in column_lower for keyword in ['yes_no', 'yn', 'boolean', 'flag']):
            valid_responses = {'yes', 'y', 'no', 'n', 'true', 'false', '1', '0'}
            if series.dtype == 'object':
                invalid_responses = series[~series.str.lower().isin(valid_responses)]
                if len(invalid_responses) > 0:
                    violations['total_violations'] += len(invalid_responses)
                    violations['violation_types'].append('Invalid yes/no responses')
                    violations['details']['yesno_violations'] = {
                        'count': len(invalid_responses),
                        'rule': 'Should be Yes/No or True/False',
                        'invalid_values': invalid_responses.tolist()[:10]
                    }
        
        # Check for unusual characters or encoding issues
        if series.dtype == 'object':
            import re
            # Check for unusual Unicode characters that might indicate encoding issues
            unusual_chars = series[series.str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
            if len(unusual_chars) > 0:
                violations['total_violations'] += len(unusual_chars)
                violations['violation_types'].append('Unusual character encoding')
                violations['details']['encoding_violations'] = {
                    'count': len(unusual_chars),
                    'rule': 'Contains non-ASCII characters that may indicate encoding issues',
                    'invalid_values': unusual_chars.tolist()[:5]
                }
        
        return violations
    
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

import streamlit as st
import os
from groq import Groq
from typing import Dict, List, Any, Optional
import json
import pandas as pd
from datetime import datetime

class AIAssistant:
    """AI-powered conversational assistant for data cleaning guidance"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.conversation_history = []
        self.context = {}
        
        # Initialize Groq client
        try:
            if self.groq_api_key:
                self.client = Groq(api_key=self.groq_api_key)
                self.model = "llama-3.1-8b-instant"
            else:
                st.warning("⚠️ GROQ_API_KEY not found. AI assistant will not be available.")
                self.client = None
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            self.client = None
    
    def set_context(self, dataset_info: Dict[str, Any], column_analysis: Optional[Dict[str, Any]] = None):
        """Set the current context for AI assistance"""
        self.context = {
            'dataset_shape': dataset_info.get('shape', 'Unknown'),
            'column_count': dataset_info.get('columns', 0),
            'missing_data_summary': dataset_info.get('missing_summary', {}),
            'column_types': dataset_info.get('column_types', {}),
            'current_column_analysis': column_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def ask_question(self, question: str, column_specific: Optional[str] = None, 
                   current_data_state: Optional[Dict[str, Any]] = None) -> str:
        """Ask the AI assistant a question with current context"""
        if not self.client:
            return "AI assistant is not available. Please set your GROQ_API_KEY in the secrets."
        
        try:
            # Update context with current data state if provided
            if current_data_state:
                self._update_context_with_current_state(current_data_state)
            
            # Build context-aware prompt
            system_prompt = self._build_system_prompt(column_specific)
            user_message = self._build_user_message(question, column_specific)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content or "No response received"
            
            # Store conversation
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'column': column_specific,
                'response': ai_response
            })
            
            return ai_response
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
    
    def _update_context_with_current_state(self, current_state: Dict[str, Any]):
        """Update AI context with current application state"""
        if 'current_dataset_stats' in current_state:
            self.context['current_dataset_stats'] = current_state['current_dataset_stats']
        
        if 'cleaning_history' in current_state:
            self.context['cleaning_history'] = current_state['cleaning_history']
        
        if 'inter_column_violations' in current_state:
            self.context['inter_column_violations'] = current_state['inter_column_violations']
        
        if 'weights_info' in current_state:
            self.context['weights_info'] = current_state['weights_info']
    
    def _build_system_prompt(self, column_specific: Optional[str] = None) -> str:
        """Build context-aware system prompt"""
        base_prompt = """You are an expert survey data analyst and statistician working for a statistical agency. 
        Your role is to provide column-specific, contextual guidance for data cleaning operations.

        CRITICAL REQUIREMENTS:
        - NEVER provide generic advice that applies to all columns
        - ALWAYS analyze each column individually based on its specific characteristics
        - Consider survey methodology and sampling design implications
        - Explain the statistical reasoning behind your recommendations
        - Compare different cleaning methods with specific pros and cons
        - Be educational - explain statistical concepts when relevant
        - Focus on maintaining data integrity for statistical agencies

        Your responses should be:
        - Specific to the column and data context provided
        - Methodologically sound for survey data
        - Clear and actionable
        - Educational when explaining statistical concepts
        """
        
        if self.context:
            context_info = f"""
            CURRENT DATASET CONTEXT:
            - Dataset shape: {self.context.get('dataset_shape', 'Unknown')}
            - Total columns: {self.context.get('column_count', 'Unknown')}
            - Column types: {json.dumps(self.context.get('column_types', {}), indent=2)}
            """
            base_prompt += context_info
        
        if column_specific and self.context.get('current_column_analysis'):
            column_analysis = self.context['current_column_analysis']
            column_prompt = f"""
            CURRENT COLUMN ANALYSIS FOR '{column_specific}':
            - Basic info: {json.dumps(column_analysis.get('basic_info', {}), indent=2)}
            - Missing data: {json.dumps(column_analysis.get('missing_analysis', {}), indent=2)}
            - Outliers: {json.dumps(column_analysis.get('outlier_analysis', {}).get('summary', {}), indent=2)}
            - Data quality: {json.dumps(column_analysis.get('data_quality', {}), indent=2)}
            
            Focus your response specifically on this column's characteristics and needs.
            """
            base_prompt += column_prompt
        
        return base_prompt
    
    def _build_user_message(self, question: str, column_specific: Optional[str] = None) -> str:
        """Build user message with context"""
        message = question
        
        if column_specific:
            message = f"For column '{column_specific}': {question}"
        
        return message
    
    def get_cleaning_recommendation(self, column: str, analysis: Dict[str, Any]) -> str:
        """Get AI recommendation for cleaning a specific column"""
        if not self.client:
            return "AI assistant is not available."
        
        # Update context with current column analysis
        self.context['current_column_analysis'] = analysis
        
        question = f"""Based on the analysis of column '{column}', what are the best cleaning strategies? 
        Please provide:
        1. Specific recommendations for this column's missing values, outliers, and data quality issues
        2. Pros and cons of each recommended method
        3. The reasoning behind your recommendations considering this column's characteristics
        4. Any survey methodology considerations
        5. Suggested order of operations for cleaning this column"""
        
        return self.ask_question(question, column)
    
    def compare_methods(self, column: str, method1: str, method2: str, analysis: Dict[str, Any]) -> str:
        """Compare two cleaning methods for a specific column"""
        self.context['current_column_analysis'] = analysis
        
        question = f"""For column '{column}', compare {method1} vs {method2}. 
        Consider:
        1. Which method is more appropriate for this column's specific characteristics?
        2. How would each method affect the data distribution and statistical properties?
        3. Survey methodology implications of each approach
        4. Computational complexity and practical considerations
        5. Your specific recommendation for this column and why"""
        
        return self.ask_question(question, column)
    
    def explain_concept(self, concept: str, context_column: Optional[str] = None) -> str:
        """Explain a statistical concept in the context of current data"""
        question = f"Please explain {concept} in the context of survey data cleaning"
        
        if context_column:
            question += f", specifically as it relates to column '{context_column}' in my current dataset"
        
        question += ". Please provide practical examples and when to use it."
        
        return self.ask_question(question, context_column)
    
    def assess_impact(self, column: str, proposed_method: str, analysis: Dict[str, Any]) -> str:
        """Assess the impact of a proposed cleaning method"""
        self.context['current_column_analysis'] = analysis
        
        question = f"""I'm planning to apply {proposed_method} to column '{column}'. 
        Please assess:
        1. How will this method affect the column's distribution and statistical properties?
        2. What are the potential impacts on survey estimates and analysis results?
        3. Are there any risks or unintended consequences I should consider?
        4. How might this affect relationships with other variables?
        5. Any alternative approaches that might be better for this specific column?"""
        
        return self.ask_question(question, column)
    
    def get_workflow_guidance(self, columns_analysis: Dict[str, Dict[str, Any]]) -> str:
        """Get guidance on cleaning workflow and column prioritization"""
        # Summarize all columns for context
        columns_summary = {}
        for col, analysis in columns_analysis.items():
            columns_summary[col] = {
                'missing_pct': analysis.get('missing_analysis', {}).get('percentage', 0),
                'quality_score': analysis.get('data_quality', {}).get('score', 100),
                'outlier_severity': analysis.get('outlier_analysis', {}).get('summary', {}).get('severity', 'low')
            }
        
        question = f"""Based on my dataset with columns: {json.dumps(columns_summary, indent=2)}
        
        Please provide workflow guidance:
        1. Which columns should I prioritize cleaning first and why?
        2. Are there any dependencies between columns that affect cleaning order?
        3. What's the recommended sequence of operations (missing values, outliers, etc.)?
        4. Any columns that require special attention or careful handling?
        5. How should I validate my cleaning results across all columns?"""
        
        return self.ask_question(question)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        return json.dumps(self.conversation_history, indent=2)

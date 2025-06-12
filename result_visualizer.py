import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.settings import RESULTS_DIR
import os

class ResultVisualizer:
    def __init__(self):
        self.style = 'whitegrid'
        self.palette = 'viridis'
        self.figure_size = (10, 6)
        
    def plot_risk_distribution(self, risk_probabilities, save_path=None):
        plt.figure(figsize=self.figure_size)
        sns.set_style(self.style)
        
        plt.hist(risk_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(risk_probabilities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(risk_probabilities):.3f}')
        plt.xlabel('Risk Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dry Eye Risk Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importance, top_n=10, save_path=None):
        if isinstance(feature_importance, dict):
            features = list(feature_importance.keys())[:top_n]
            importance = list(feature_importance.values())[:top_n]
        else:
            features = [f'Feature_{i}' for i in range(min(top_n, len(feature_importance)))]
            importance = feature_importance[:top_n]
            
        plt.figure(figsize=self.figure_size)
        sns.set_style(self.style)
        
        bars = plt.barh(features, importance, color='lightcoral')
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importance for Dry Eye Prediction')
        plt.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_risk_gauge(self, risk_probability, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Low Risk', 'Medium Risk', 'High Risk']
        colors = ['green', 'orange', 'red']
        sizes = [30, 30, 40]
        
        if risk_probability <= 0.3:
            explode = (0.1, 0, 0)
        elif risk_probability <= 0.6:
            explode = (0, 0.1, 0)
        else:
            explode = (0, 0, 0.1)
            
        wedges, texts, autotexts = ax.pie(sizes, labels=categories, colors=colors, 
                                         explode=explode, autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Risk Assessment\nProbability: {risk_probability:.3f}', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_factor_contributions(self, factor_contributions, save_path=None):
        factors = list(factor_contributions.keys())
        contributions = [factor_contributions[f]['contribution'] for f in factors]
        
        plt.figure(figsize=self.figure_size)
        sns.set_style(self.style)
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(factors)))
        bars = plt.bar(factors, contributions, color=colors)
        
        plt.xlabel('Risk Factors')
        plt.ylabel('Contribution to Risk')
        plt.title('Factor Contributions to Dry Eye Risk')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{contrib:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_model_performance(self, metrics, save_path=None):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.figure(figsize=self.figure_size)
        sns.set_style(self.style)
        
        bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.7)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_severity_distribution(self, severity_predictions, save_path=None):
        severity_labels = ['No Risk', 'Mild', 'Moderate', 'Severe']
        unique, counts = np.unique(severity_predictions, return_counts=True)
        
        plt.figure(figsize=self.figure_size)
        sns.set_style(self.style)
        
        colors = ['lightgreen', 'yellow', 'orange', 'red']
        selected_colors = [colors[i] for i in unique]
        selected_labels = [severity_labels[i] for i in unique]
        
        plt.pie(counts, labels=selected_labels, colors=selected_colors, 
                autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Severity Levels')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def create_comprehensive_dashboard(self, risk_assessment, feature_importance, 
                                    factor_contributions, model_metrics, save_dir=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dry Eye Disease Risk Assessment Dashboard', fontsize=16, fontweight='bold')
        
        # Risk gauge
        ax1 = axes[0, 0]
        risk_prob = risk_assessment.get('risk_probability', 0)
        categories = ['Low Risk', 'Medium Risk', 'High Risk']
        colors = ['green', 'orange', 'red']
        sizes = [30, 30, 40]
        ax1.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Risk Level\nProbability: {risk_prob:.3f}')
        
        # Feature importance
        ax2 = axes[0, 1]
        if feature_importance:
            features = list(feature_importance.keys())[:5]
            importance = list(feature_importance.values())[:5]
            ax2.barh(features, importance, color='lightcoral')
            ax2.set_title('Top 5 Risk Factors')
            ax2.set_xlabel('Importance')
        
        # Factor contributions
        ax3 = axes[1, 0]
        if factor_contributions:
            factors = list(factor_contributions.keys())[:5]
            contributions = [factor_contributions[f]['contribution'] for f in factors]
            ax3.bar(factors, contributions, color='steelblue')
            ax3.set_title('Factor Contributions')
            ax3.set_ylabel('Contribution')
            ax3.tick_params(axis='x', rotation=45)
        
        # Model performance
        ax4 = axes[1, 1]
        if model_metrics:
            metrics = list(model_metrics.keys())
            values = list(model_metrics.values())
            ax4.bar(metrics, values, color='lightgreen')
            ax4.set_title('Model Performance')
            ax4.set_ylabel('Score')
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_dir:
            dashboard_path = os.path.join(save_dir, 'comprehensive_dashboard.png')
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            
        return fig
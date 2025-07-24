#!/usr/bin/env python3
"""
Runner script for the Enzymatic Hydrolysis Complexity Analysis

This script provides a simple interface to run the complexity analysis
and export results to various formats.

Usage:
    python run_complexity_analysis.py

Author: Ethanol AI Project
Date: 2024
"""

import json
import pandas as pd
from complexity_analysis import EnzymaticHydrolysisComplexityAnalyzer


def export_results_to_json(analyzer, filename="complexity_analysis_results.json"):
    """Export analysis results to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(analyzer.results, f, indent=2, default=str)
        print(f"📄 Results exported to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error exporting to JSON: {e}")
        return False


def export_results_to_csv(analyzer, filename="complexity_analysis_summary.csv"):
    """Export key metrics to CSV file."""
    try:
        summary_data = []
        
        # Dimensionality metrics
        if 'dimensionality' in analyzer.results:
            dim = analyzer.results['dimensionality']
            summary_data.append({
                'Metric': 'Number of Input Variables',
                'Value': dim['n_input_variables'],
                'Category': 'Dimensionality'
            })
            summary_data.append({
                'Metric': 'Number of Output Variables', 
                'Value': dim['n_output_variables'],
                'Category': 'Dimensionality'
            })
            summary_data.append({
                'Metric': 'Data Density',
                'Value': round(dim['data_density'], 3),
                'Category': 'Dimensionality'
            })
        
        # Linear relationships
        if 'linear_relationships' in analyzer.results:
            linear = analyzer.results['linear_relationships']
            for output_var, relationships in linear.items():
                if 'multiple_inputs' in relationships:
                    summary_data.append({
                        'Metric': f'{output_var} - Multiple Linear R²',
                        'Value': round(relationships['multiple_inputs']['r_squared'], 4),
                        'Category': 'Linear Relationships'
                    })
        
        # Variability
        if 'variability' in analyzer.results:
            var = analyzer.results['variability']
            for output_var, stats in var.items():
                summary_data.append({
                    'Metric': f'{output_var} - CV',
                    'Value': round(stats['cv'], 4),
                    'Category': 'Variability'
                })
        
        # Entropy
        if 'entropy' in analyzer.results:
            entropy = analyzer.results['entropy']
            for output_var, entropies in entropy.items():
                summary_data.append({
                    'Metric': f'{output_var} - Entropy (10 bins)',
                    'Value': round(entropies.get('bins_10', 0), 3),
                    'Category': 'Entropy'
                })
        
        # Temporal dependencies
        if 'temporal_dependencies' in analyzer.results:
            temporal = analyzer.results['temporal_dependencies']
            for output_var, temp_stats in temporal.items():
                best_poly = max(temp_stats['polynomial_fits'].values(), 
                              key=lambda x: x['r_squared'], default={'r_squared': 0})
                nonlinearity = best_poly['r_squared'] - temp_stats['linear_r_squared']
                summary_data.append({
                    'Metric': f'{output_var} - Non-linearity Improvement',
                    'Value': round(nonlinearity, 4),
                    'Category': 'Temporal Dependencies'
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"📊 Summary exported to: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")
        return False


def print_detailed_insights(analyzer):
    """Print detailed insights from the analysis."""
    print("\n" + "="*70)
    print("DETAILED INSIGHTS AND RECOMMENDATIONS")
    print("="*70)
    
    results = analyzer.results
    
    # Data quality assessment
    print("\n🔍 DATA QUALITY ASSESSMENT:")
    if 'dimensionality' in results:
        dim = results['dimensionality']
        samples_per_dim = dim['samples_per_dimension']
        
        if samples_per_dim < 5:
            print("   ⚠️  WARNING: Low samples per dimension ratio.")
            print("      Recommendation: Collect more data or reduce dimensionality.")
        elif samples_per_dim < 10:
            print("   ⚠️  CAUTION: Moderate samples per dimension ratio.")
            print("      Consider collecting additional data for better model reliability.")
        else:
            print("   ✅ GOOD: Adequate samples per dimension ratio.")
    
    # Variable importance insights
    print("\n📊 VARIABLE IMPORTANCE:")
    if 'linear_relationships' in results:
        linear = results['linear_relationships']
        
        # Find most important variables for each output
        for output_var, relationships in linear.items():
            print(f"\n   {output_var}:")
            
            # Sort individual variable R² values
            individual_r2 = {var: rel['r_squared'] for var, rel in relationships.items() 
                           if var != 'multiple_inputs' and isinstance(rel, dict)}
            
            sorted_vars = sorted(individual_r2.items(), key=lambda x: x[1], reverse=True)
            
            for i, (var, r2) in enumerate(sorted_vars[:3]):  # Top 3
                importance = "High" if r2 > 0.3 else "Medium" if r2 > 0.1 else "Low"
                print(f"      {i+1}. {var}: R² = {r2:.3f} ({importance} importance)")
    
    # Model complexity recommendations
    print("\n🎯 MODELING RECOMMENDATIONS:")
    
    if 'linear_relationships' in results:
        avg_r2 = sum(rel.get('multiple_inputs', {}).get('r_squared', 0) 
                    for rel in linear.values()) / len(linear)
        
        if avg_r2 < 0.5:
            print("   📈 Consider non-linear models (Random Forest, Neural Networks)")
            print("      Linear relationships are weak - non-linear methods may perform better.")
        elif avg_r2 < 0.8:
            print("   📈 Linear models with polynomial features may be effective")
            print("      Some linear relationship exists but could benefit from feature engineering.")
        else:
            print("   📈 Linear models should perform well")
            print("      Strong linear relationships detected.")
    
    if 'temporal_dependencies' in results:
        temporal = results['temporal_dependencies']
        strong_nonlinear = sum(1 for temp_stats in temporal.values()
                              if max(temp_stats['polynomial_fits'].values(), 
                                    key=lambda x: x['r_squared'])['improvement_over_linear'] > 0.2)
        
        if strong_nonlinear > 0:
            print("   ⏰ Time-based features are highly non-linear")
            print("      Consider polynomial time features or time-series models.")
    
    # Complexity summary
    print("\n📋 COMPLEXITY SUMMARY:")
    
    complexity_factors = []
    
    if 'dimensionality' in results:
        if results['dimensionality']['n_input_variables'] > 5:
            complexity_factors.append("High-dimensional input space")
    
    if 'variability' in results:
        high_var_outputs = sum(1 for stats in results['variability'].values() 
                              if stats['cv'] > 0.7)
        if high_var_outputs > 0:
            complexity_factors.append("High output variability")
    
    if 'entropy' in results:
        high_entropy_outputs = sum(1 for entropies in results['entropy'].values()
                                  if entropies.get('bins_10', 0) > 3.0)
        if high_entropy_outputs > 0:
            complexity_factors.append("High information entropy")
    
    if complexity_factors:
        print("   🚨 COMPLEXITY FACTORS IDENTIFIED:")
        for factor in complexity_factors:
            print(f"      • {factor}")
        
        print("\n   💡 MITIGATION STRATEGIES:")
        print("      • Use ensemble methods for robustness")
        print("      • Implement cross-validation for model selection")  
        print("      • Consider dimensionality reduction techniques")
        print("      • Use regularization to prevent overfitting")
    else:
        print("   ✅ MODERATE COMPLEXITY: Standard modeling approaches should work well")


def main():
    """Main function to run analysis and export results."""
    print("🧪 Enzymatic Hydrolysis Complexity Analysis Runner")
    print("="*60)
    
    # Initialize and run analysis
    analyzer = EnzymaticHydrolysisComplexityAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if not results:
        print("❌ Analysis failed!")
        return None
    
    # Export results
    print(f"\n📤 EXPORTING RESULTS...")
    export_results_to_json(analyzer)
    export_results_to_csv(analyzer)
    
    # Print detailed insights
    print_detailed_insights(analyzer)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Results available in:")
    print(f"   • complexity_analysis_results.json (complete results)")
    print(f"   • complexity_analysis_summary.csv (key metrics)")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
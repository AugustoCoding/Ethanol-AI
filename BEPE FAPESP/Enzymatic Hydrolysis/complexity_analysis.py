#!/usr/bin/env python3
"""
Complexity Analysis of Enzymatic Hydrolysis Problem

This script analyzes the complexity of the enzymatic hydrolysis problem using 
experimental data. The analysis focuses on:

1. Dimensionality: Number of input variables and their impact
2. Linear relationships: R-squared coefficients for outputs vs inputs
3. Variability: Statistical analysis of the outputs
4. Entropy: Entropy of output distributions
5. Temporal dependencies: Non-linear relationship with time

Author: Ethanol AI Project
Date: 2024
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import special
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


class EnzymaticHydrolysisComplexityAnalyzer:
    """
    A comprehensive analyzer for the complexity of enzymatic hydrolysis processes.
    """
    
    def __init__(self, data_path="Experimental Data.csv"):
        """
        Initialize the analyzer with experimental data.
        
        Args:
            data_path (str): Path to the experimental data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.input_variables = [
            'Cellulose', 'Hemicellulose', 'Lignin', 
            'Solids Loading [g/L]', 'Enzyme Loading [g/L]', 'Time [h]'
        ]
        self.output_variables = [
            'Glucose Concentration [g/L]', 
            'Xylose Concentration [g/L]', 
            'Cellobiose Concentration [g/L]'
        ]
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the experimental data."""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Clean column names (remove extra spaces)
            self.data.columns = self.data.columns.str.strip()
            
            print(f"Data loaded successfully: {len(self.data)} samples")
            print(f"Columns: {list(self.data.columns)}")
            
            # Validate that all required columns exist
            missing_inputs = [col for col in self.input_variables if col not in self.data.columns]
            missing_outputs = [col for col in self.output_variables if col not in self.data.columns]
            
            if missing_inputs:
                print(f"Warning: Missing input columns: {missing_inputs}")
            if missing_outputs:
                print(f"Warning: Missing output columns: {missing_outputs}")
                
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_dimensionality(self):
        """
        Analyze the dimensionality of the problem and input variable impacts.
        
        Returns:
            dict: Dimensionality analysis results
        """
        print("\n=== DIMENSIONALITY ANALYSIS ===")
        
        # Basic dimensionality metrics
        n_samples = len(self.data)
        n_inputs = len(self.input_variables)
        n_outputs = len(self.output_variables)
        
        # Calculate input variable ranges and variance
        input_stats = {}
        for var in self.input_variables:
            if var in self.data.columns:
                values = self.data[var]
                input_stats[var] = {
                    'min': values.min(),
                    'max': values.max(),
                    'range': values.max() - values.min(),
                    'std': values.std(),
                    'variance': values.var(),
                    'cv': values.std() / values.mean() if values.mean() != 0 else 0  # Coefficient of variation
                }
        
        # Calculate curse of dimensionality indicators
        data_density = n_samples / (n_inputs ** 2) if n_inputs > 0 else 0
        samples_per_dimension = n_samples / n_inputs if n_inputs > 0 else 0
        
        dimensionality_results = {
            'n_samples': n_samples,
            'n_input_variables': n_inputs,
            'n_output_variables': n_outputs,
            'data_density': data_density,
            'samples_per_dimension': samples_per_dimension,
            'input_variable_stats': input_stats
        }
        
        # Print results
        print(f"Number of samples: {n_samples}")
        print(f"Number of input variables: {n_inputs}")
        print(f"Number of output variables: {n_outputs}")
        print(f"Data density (samples/dimensions²): {data_density:.2f}")
        print(f"Samples per dimension: {samples_per_dimension:.2f}")
        
        print(f"\nInput Variable Impact Analysis:")
        for var, stats_dict in input_stats.items():
            print(f"  {var}:")
            print(f"    Range: {stats_dict['range']:.3f}")
            print(f"    CV: {stats_dict['cv']:.3f}")
        
        self.results['dimensionality'] = dimensionality_results
        return dimensionality_results
    
    def analyze_linear_relationships(self):
        """
        Compute R-squared coefficients for linear relationships between inputs and outputs.
        
        Returns:
            dict: Linear relationship analysis results
        """
        print("\n=== LINEAR RELATIONSHIPS ANALYSIS ===")
        
        linear_results = {}
        
        for output_var in self.output_variables:
            if output_var not in self.data.columns:
                continue
                
            linear_results[output_var] = {}
            y = self.data[output_var].values
            
            print(f"\nR-squared values for {output_var}:")
            
            # Individual input analysis
            for input_var in self.input_variables:
                if input_var not in self.data.columns:
                    continue
                    
                X = self.data[input_var].values.reshape(-1, 1)
                
                # Fit linear regression
                reg = LinearRegression()
                reg.fit(X, y)
                y_pred = reg.predict(X)
                r2 = r2_score(y, y_pred)
                
                linear_results[output_var][input_var] = {
                    'r_squared': r2,
                    'coefficient': reg.coef_[0],
                    'intercept': reg.intercept_
                }
                
                print(f"  {input_var}: R² = {r2:.4f}")
            
            # Multiple input analysis (all inputs combined)
            available_inputs = [var for var in self.input_variables if var in self.data.columns]
            if len(available_inputs) > 1:
                X_multi = self.data[available_inputs].values
                reg_multi = LinearRegression()
                reg_multi.fit(X_multi, y)
                y_pred_multi = reg_multi.predict(X_multi)
                r2_multi = r2_score(y, y_pred_multi)
                
                linear_results[output_var]['multiple_inputs'] = {
                    'r_squared': r2_multi,
                    'coefficients': reg_multi.coef_,
                    'intercept': reg_multi.intercept_
                }
                
                print(f"  Multiple inputs combined: R² = {r2_multi:.4f}")
        
        self.results['linear_relationships'] = linear_results
        return linear_results
    
    def analyze_variability(self):
        """
        Perform statistical analysis of output variability.
        
        Returns:
            dict: Variability analysis results
        """
        print("\n=== VARIABILITY ANALYSIS ===")
        
        variability_results = {}
        
        for output_var in self.output_variables:
            if output_var not in self.data.columns:
                continue
                
            values = self.data[output_var].values
            
            # Basic statistics
            stats_dict = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'variance': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
            
            variability_results[output_var] = stats_dict
            
            print(f"\n{output_var}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            print(f"  CV: {stats_dict['cv']:.3f}")
            print(f"  Skewness: {stats_dict['skewness']:.3f}")
            print(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")
        
        self.results['variability'] = variability_results
        return variability_results
    
    def calculate_entropy(self, values, bins=10):
        """
        Calculate the entropy of a distribution.
        
        Args:
            values (array): Data values
            bins (int): Number of bins for histogram
            
        Returns:
            float: Entropy value
        """
        # Create histogram
        counts, _ = np.histogram(values, bins=bins)
        
        # Calculate probabilities
        probabilities = counts / np.sum(counts)
        
        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def analyze_entropy(self):
        """
        Measure the entropy of output distributions.
        
        Returns:
            dict: Entropy analysis results
        """
        print("\n=== ENTROPY ANALYSIS ===")
        
        entropy_results = {}
        
        for output_var in self.output_variables:
            if output_var not in self.data.columns:
                continue
                
            values = self.data[output_var].values
            
            # Calculate entropy with different bin sizes
            entropies = {}
            for bins in [5, 10, 15, 20]:
                entropy = self.calculate_entropy(values, bins)
                entropies[f'bins_{bins}'] = entropy
            
            # Use adaptive binning (Sturges' rule)
            n_samples = len(values)
            sturges_bins = int(np.ceil(np.log2(n_samples) + 1))
            entropy_sturges = self.calculate_entropy(values, sturges_bins)
            entropies['sturges_rule'] = entropy_sturges
            
            entropy_results[output_var] = entropies
            
            print(f"\n{output_var}:")
            print(f"  Entropy (10 bins): {entropies['bins_10']:.3f} bits")
            print(f"  Entropy (Sturges rule, {sturges_bins} bins): {entropy_sturges:.3f} bits")
        
        self.results['entropy'] = entropy_results
        return entropy_results
    
    def analyze_temporal_dependencies(self):
        """
        Assess non-linear relationships of outputs with time.
        
        Returns:
            dict: Temporal dependency analysis results
        """
        print("\n=== TEMPORAL DEPENDENCIES ANALYSIS ===")
        
        if 'Time [h]' not in self.data.columns:
            print("Time variable not found in data")
            return {}
        
        temporal_results = {}
        time_values = self.data['Time [h]'].values
        
        for output_var in self.output_variables:
            if output_var not in self.data.columns:
                continue
                
            y_values = self.data[output_var].values
            
            # Linear relationship (for comparison)
            reg_linear = LinearRegression()
            reg_linear.fit(time_values.reshape(-1, 1), y_values)
            y_pred_linear = reg_linear.predict(time_values.reshape(-1, 1))
            r2_linear = r2_score(y_values, y_pred_linear)
            
            # Polynomial relationships (non-linear)
            poly_results = {}
            for degree in [2, 3, 4]:
                try:
                    poly_features = PolynomialFeatures(degree=degree)
                    X_poly = poly_features.fit_transform(time_values.reshape(-1, 1))
                    
                    reg_poly = LinearRegression()
                    reg_poly.fit(X_poly, y_values)
                    y_pred_poly = reg_poly.predict(X_poly)
                    r2_poly = r2_score(y_values, y_pred_poly)
                    
                    poly_results[f'degree_{degree}'] = {
                        'r_squared': r2_poly,
                        'improvement_over_linear': r2_poly - r2_linear
                    }
                except:
                    poly_results[f'degree_{degree}'] = {'r_squared': 0, 'improvement_over_linear': 0}
            
            # Correlation analysis
            correlation = np.corrcoef(time_values, y_values)[0, 1]
            
            temporal_results[output_var] = {
                'linear_r_squared': r2_linear,
                'correlation': correlation,
                'polynomial_fits': poly_results
            }
            
            print(f"\n{output_var} vs Time:")
            print(f"  Linear R²: {r2_linear:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            for degree, result in poly_results.items():
                print(f"  Polynomial {degree} R²: {result['r_squared']:.4f} "
                      f"(improvement: +{result['improvement_over_linear']:.4f})")
        
        self.results['temporal_dependencies'] = temporal_results
        return temporal_results
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all analyses.
        
        Returns:
            str: Formatted summary report
        """
        print("\n" + "="*60)
        print("ENZYMATIC HYDROLYSIS COMPLEXITY ANALYSIS SUMMARY")
        print("="*60)
        
        if 'dimensionality' in self.results:
            dim = self.results['dimensionality']
            print(f"\n📊 DIMENSIONALITY:")
            print(f"   • Problem dimension: {dim['n_input_variables']} inputs → {dim['n_output_variables']} outputs")
            print(f"   • Data density: {dim['data_density']:.2f} (samples per dimension²)")
            print(f"   • Sample efficiency: {dim['samples_per_dimension']:.1f} samples per dimension")
        
        if 'linear_relationships' in self.results:
            linear = self.results['linear_relationships']
            print(f"\n📈 LINEAR RELATIONSHIPS:")
            for output_var, relationships in linear.items():
                if 'multiple_inputs' in relationships:
                    r2_multi = relationships['multiple_inputs']['r_squared']
                    print(f"   • {output_var}: R² = {r2_multi:.3f} (multiple linear regression)")
        
        if 'variability' in self.results:
            var = self.results['variability']
            print(f"\n📐 OUTPUT VARIABILITY:")
            for output_var, stats in var.items():
                cv = stats['cv']
                complexity_level = "Low" if cv < 0.3 else "Medium" if cv < 0.7 else "High"
                print(f"   • {output_var}: CV = {cv:.3f} ({complexity_level} variability)")
        
        if 'entropy' in self.results:
            entropy = self.results['entropy']
            print(f"\n🔀 INFORMATION ENTROPY:")
            for output_var, entropies in entropy.items():
                ent_value = entropies.get('bins_10', 0)
                print(f"   • {output_var}: {ent_value:.3f} bits")
        
        if 'temporal_dependencies' in self.results:
            temporal = self.results['temporal_dependencies']
            print(f"\n⏰ TEMPORAL COMPLEXITY:")
            for output_var, temp_stats in temporal.items():
                linear_r2 = temp_stats['linear_r_squared']
                best_poly = max(temp_stats['polynomial_fits'].values(), 
                              key=lambda x: x['r_squared'], default={'r_squared': 0})
                nonlinearity = best_poly['r_squared'] - linear_r2
                print(f"   • {output_var}: Non-linearity improvement = +{nonlinearity:.3f}")
        
        # Overall complexity assessment
        print(f"\n🎯 OVERALL COMPLEXITY ASSESSMENT:")
        
        # Dimensionality complexity
        if 'dimensionality' in self.results:
            dim_complexity = "High" if dim['n_input_variables'] > 5 else "Medium" if dim['n_input_variables'] > 3 else "Low"
            print(f"   • Dimensionality: {dim_complexity} ({dim['n_input_variables']} variables)")
        
        # Linearity assessment
        if 'linear_relationships' in self.results:
            avg_r2 = np.mean([rel.get('multiple_inputs', {}).get('r_squared', 0) 
                             for rel in linear.values()])
            linearity = "High" if avg_r2 > 0.8 else "Medium" if avg_r2 > 0.5 else "Low"
            print(f"   • Linearity: {linearity} (avg R² = {avg_r2:.3f})")
        
        # Variability assessment
        if 'variability' in self.results:
            avg_cv = np.mean([stats['cv'] for stats in var.values()])
            var_complexity = "High" if avg_cv > 0.7 else "Medium" if avg_cv > 0.3 else "Low"
            print(f"   • Variability: {var_complexity} (avg CV = {avg_cv:.3f})")
        
        print("\n" + "="*60)
        
        return self.results
    
    def run_complete_analysis(self):
        """
        Run the complete complexity analysis pipeline.
        
        Returns:
            dict: Complete analysis results
        """
        print("Starting Enzymatic Hydrolysis Complexity Analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Run all analyses
        self.analyze_dimensionality()
        self.analyze_linear_relationships()
        self.analyze_variability()
        self.analyze_entropy()
        self.analyze_temporal_dependencies()
        
        # Generate summary
        self.generate_summary_report()
        
        return self.results


def main():
    """Main function to run the analysis."""
    
    # Initialize analyzer
    analyzer = EnzymaticHydrolysisComplexityAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"Results saved to analyzer.results")
    else:
        print("❌ Analysis failed!")
        return None
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
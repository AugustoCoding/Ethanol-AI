# Enzymatic Hydrolysis Complexity Analysis

This directory contains a comprehensive complexity analysis of the enzymatic hydrolysis problem using experimental data. The analysis addresses five key aspects of problem complexity as specified in the research requirements.

## 📊 Analysis Overview

The complexity analysis focuses on:

1. **Dimensionality**: Assessment of input variables (6 variables) and their impact
2. **Linear relationships**: R-squared coefficients for outputs vs inputs
3. **Variability**: Statistical analysis of output distributions
4. **Entropy**: Information entropy measurement of output distributions
5. **Temporal dependencies**: Non-linear relationship assessment with time

## 🗂️ Files Description

### Core Analysis Files
- `complexity_analysis.py` - Main analysis implementation (comprehensive class-based analyzer)
- `run_complexity_analysis.py` - User-friendly runner script with export capabilities
- `Experimental Data.csv` - Source experimental data (70 samples, 6 inputs, 3 outputs)

### Output Files (Generated)
- `complexity_analysis_results.json` - Complete analysis results in JSON format
- `complexity_analysis_summary.csv` - Key metrics summary in CSV format

## 🚀 Usage

### Quick Start
```bash
# Navigate to the analysis directory
cd "BEPE FAPESP/Enzymatic Hydrolysis"

# Run the complete analysis
python run_complexity_analysis.py
```

### Advanced Usage
```python
# Import and use the analyzer class directly
from complexity_analysis import EnzymaticHydrolysisComplexityAnalyzer

# Initialize analyzer
analyzer = EnzymaticHydrolysisComplexityAnalyzer("Experimental Data.csv")

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access specific analysis results
dimensionality = analyzer.results['dimensionality']
linear_relationships = analyzer.results['linear_relationships']
variability = analyzer.results['variability']
entropy = analyzer.results['entropy']
temporal_dependencies = analyzer.results['temporal_dependencies']
```

## 📈 Key Findings

### Dimensionality Analysis
- **Problem dimension**: 6 inputs → 3 outputs
- **Data density**: 1.94 samples per dimension²
- **Sample efficiency**: 11.7 samples per dimension
- **Assessment**: Adequate data density for reliable analysis

### Input Variable Importance
1. **Time [h]**: Most critical variable (R² = 0.65-0.29 for different outputs)
2. **Enzyme Loading [g/L]**: Moderate importance (R² = 0.05-0.15)
3. **Solids Loading [g/L]**: Low importance (R² = 0.001-0.034)
4. **Cellulose, Hemicellulose, Lignin**: No variation in experimental data

### Linear Relationships (Multiple Linear Regression)
- **Glucose Concentration**: R² = 0.736 (Good linear predictability)
- **Xylose Concentration**: R² = 0.645 (Moderate linear predictability)
- **Cellobiose Concentration**: R² = 0.469 (Lower linear predictability)

### Output Variability
- **Glucose**: CV = 0.681 (Medium variability)
- **Xylose**: CV = 0.668 (Medium variability)
- **Cellobiose**: CV = 0.812 (High variability - most complex output)

### Information Entropy
- **Xylose**: 3.260 bits (highest entropy - most unpredictable)
- **Glucose**: 3.166 bits (high entropy)
- **Cellobiose**: 2.877 bits (lower entropy - more predictable patterns)

### Temporal Dependencies
Strong non-linear relationships with time detected:
- **Cellobiose**: +0.308 improvement with polynomial features (most non-linear)
- **Xylose**: +0.210 improvement
- **Glucose**: +0.159 improvement

## 🎯 Complexity Assessment

### Overall Complexity: **HIGH**

**Contributing Factors:**
- High-dimensional input space (6 variables)
- High output variability (avg CV = 0.72)
- High information entropy in outputs
- Significant temporal non-linearities

### Modeling Recommendations

1. **Model Selection**:
   - Linear models with polynomial time features
   - Consider ensemble methods for robustness
   - Non-linear models may outperform linear approaches

2. **Feature Engineering**:
   - Polynomial time features (up to degree 4)
   - Interaction terms between key variables
   - Time-based derived features

3. **Validation Strategy**:
   - Use cross-validation for model selection
   - Implement regularization to prevent overfitting
   - Consider time-series validation approaches

## 📋 Dependencies

Required Python packages:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical functions
- `scikit-learn` - Machine learning tools

All dependencies are listed in the main `requirements.txt` file.

## 🔬 Technical Details

### Data Structure
- **Samples**: 70 experimental observations
- **Input Variables**: 6 (Cellulose, Hemicellulose, Lignin, Solids Loading, Enzyme Loading, Time)
- **Output Variables**: 3 (Glucose, Xylose, Cellobiose concentrations)
- **Time Range**: 0-96 hours
- **Missing Values**: None

### Analysis Methods
- **Linear Regression**: Individual and multiple variable analysis
- **Polynomial Regression**: Degrees 2-4 for temporal analysis
- **Statistical Analysis**: Comprehensive descriptive statistics
- **Information Theory**: Shannon entropy calculation
- **Correlation Analysis**: Pearson correlation coefficients

### Reproducibility
All analyses use fixed random states where applicable and provide deterministic results. The complete methodology is documented in the code with detailed comments.

## 📞 Support

For questions about the analysis or implementation:
1. Review the detailed code comments in `complexity_analysis.py`
2. Check the analysis output and insights in the generated files
3. Refer to the scientific literature on enzymatic hydrolysis modeling

---

*This analysis was developed as part of the Ethanol AI project, a research initiative by UFSCar and DTU with FAPESP funding.*
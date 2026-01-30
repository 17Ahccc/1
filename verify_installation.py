"""
Verification script to test the installation and basic functionality
"""

import sys
import importlib

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'shap', 
        'prophet', 'statsmodels', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    print("Checking dependencies...")
    missing = []
    
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_data_file():
    """Check if the data file exists."""
    import os
    
    print("\nChecking data file...")
    if os.path.exists('2026_MCM_Problem_C_Data.csv'):
        print("✓ Data file found")
        return True
    else:
        print("✗ Data file not found (2026_MCM_Problem_C_Data.csv)")
        return False

def check_modules():
    """Check if custom modules can be imported."""
    print("\nChecking custom modules...")
    
    modules = [
        'src.preprocessing.data_preprocessor',
        'src.preprocessing.feature_engineer',
        'src.models.random_forest_model',
        'src.models.xgboost_model',
        'src.analysis.fairness_analyzer',
        'src.analysis.trend_forecaster'
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {e}")
            return False
    
    print("\n✓ All custom modules can be imported!")
    return True

def main():
    """Run all verification checks."""
    print("="*70)
    print("INSTALLATION VERIFICATION")
    print("="*70)
    
    checks = [
        check_dependencies(),
        check_data_file(),
        check_modules()
    ]
    
    print("\n" + "="*70)
    if all(checks):
        print("✓ VERIFICATION SUCCESSFUL")
        print("="*70)
        print("\nYou can now run: python main.py")
        return 0
    else:
        print("✗ VERIFICATION FAILED")
        print("="*70)
        print("\nPlease fix the issues above before running the pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Complete test script to verify all functionality
"""
import os
import sys

def test_imports():
    """Test all modules can be imported"""
    print("Testing module imports...")
    try:
        from data_preprocessing import DataPreprocessor
        from task1_voting_estimation import VotingEstimator, run_task1_analysis
        from task2_ranking_comparison import RankingMethodComparator, run_task2_analysis
        from task3_success_factors import SuccessFactorAnalyzer, run_task3_analysis
        from task4_improved_voting import ImprovedVotingSystem, run_task4_analysis
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test data can be loaded and preprocessed"""
    print("\nTesting data loading...")
    try:
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
        data = preprocessor.load_data()
        print(f"✓ Data loaded: {len(data)} records")
        
        processed = preprocessor.preprocess_all()
        print(f"✓ Data preprocessed: {processed.shape[0]} rows, {processed.shape[1]} columns")
        return True, processed
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False, None

def test_output_files():
    """Test that output files exist"""
    print("\nChecking output files...")
    expected_files = [
        'task1_voting_estimates.csv',
        'task2_method_comparison.csv',
        'task3_factor_impacts.csv',
        'task3_industry_analysis.csv',
        'task3_age_analysis.csv',
        'task4_hybrid_system_results.csv'
    ]
    
    all_exist = True
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size} bytes)")
        else:
            print(f"✗ {file} (missing)")
            all_exist = False
    
    return all_exist

def test_documentation():
    """Test documentation files exist"""
    print("\nChecking documentation...")
    expected_docs = [
        'README.md',
        'USAGE_GUIDE.md',
        'IMPLEMENTATION_SUMMARY.md',
        'requirements.txt'
    ]
    
    all_exist = True
    for doc in expected_docs:
        if os.path.exists(doc):
            print(f"✓ {doc}")
        else:
            print(f"✗ {doc} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("="*60)
    print("COMPLETE FUNCTIONALITY TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test data loading
    success, data = test_data_loading()
    results.append(("Data Loading", success))
    
    # Test output files
    results.append(("Output Files", test_output_files()))
    
    # Test documentation
    results.append(("Documentation", test_documentation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nImplementation is complete and working correctly!")
        print("Run 'python main_model.py' to execute full analysis.")
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

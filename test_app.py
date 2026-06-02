"""
Comprehensive test script for Turbulence Forecast application
"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("TURBULENCE FORECAST APPLICATION - COMPREHENSIVE TEST")
print("=" * 60)

# Test 1: Import all utilities
print("\n[1/6] Testing imports...")
try:
    from utils.ml_models import TurbulencePredictor, turbulence_model
    from utils.turbulence_calculator import TurbulenceCalculator
    from utils.weather_api import WeatherAPI
    print("✅ All utility imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Test TurbulencePredictor initialization
print("\n[2/6] Testing TurbulencePredictor initialization...")
try:
    predictor = TurbulencePredictor()
    print("✅ TurbulencePredictor initialized successfully")
except Exception as e:
    print(f"❌ TurbulencePredictor initialization error: {e}")
    sys.exit(1)

# Test 3: Test synthetic data generation
print("\n[3/6] Testing synthetic data generation...")
try:
    X, y = predictor._generate_synthetic_data(n_samples=100)
    assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    print(f"✅ Synthetic data generation successful: X{X.shape}, y{y.shape}")
except Exception as e:
    print(f"❌ Synthetic data generation error: {e}")
    sys.exit(1)

# Test 4: Test model accuracies
print("\n[4/6] Testing model accuracy metrics...")
try:
    accuracies = predictor.get_model_accuracies()
    
    # Check required keys
    assert 'random_forest' in accuracies, "Missing 'random_forest' key"
    assert 'gradient_boosting' in accuracies, "Missing 'gradient_boosting' key"
    
    # Check random_forest metrics
    rf_metrics = ['accuracy', 'r2_score', 'rmse', 'confidence', 'normalized_rmse']
    for metric in rf_metrics:
        assert metric in accuracies['random_forest'], f"Missing '{metric}' in random_forest"
    
    # Check gradient_boosting metrics
    gb_metrics = ['accuracy', 'r2_score', 'rmse', 'confidence', 'normalized_rmse']
    for metric in gb_metrics:
        assert metric in accuracies['gradient_boosting'], f"Missing '{metric}' in gradient_boosting"
    
    print("✅ Model accuracy metrics successful")
    print(f"   Random Forest Accuracy: {accuracies['random_forest']['accuracy']:.2f}%")
    print(f"   Gradient Boosting Accuracy: {accuracies['gradient_boosting']['accuracy']:.2f}%")
except Exception as e:
    print(f"❌ Model accuracy error: {e}")
    sys.exit(1)

# Test 5: Test feature importance
print("\n[5/6] Testing feature importance...")
try:
    importance = predictor.get_feature_importance()
    assert importance is not None, "Feature importance returned None"
    print(f"✅ Feature importance successful (top feature: {importance.iloc[0]['feature']})")
except Exception as e:
    print(f"❌ Feature importance error: {e}")
    sys.exit(1)

# Test 6: Test TurbulenceCalculator
print("\n[6/6] Testing TurbulenceCalculator...")
try:
    calc = TurbulenceCalculator()
    test_turbulence = 2.5
    severity = calc.get_turbulence_severity(test_turbulence)
    severity_color = calc.get_severity_color(severity)
    comfort = calc.calculate_passenger_comfort_index(test_turbulence, 60)
    fuel_impact = calc.estimate_fuel_consumption_impact(test_turbulence, 2)
    print(f"✅ TurbulenceCalculator successful")
    print(f"   Severity: {severity} ({severity_color})")
    print(f"   Passenger Comfort: {comfort:.1f}%")
    print(f"   Fuel Impact: {fuel_impact:.1f}%")
except Exception as e:
    print(f"❌ TurbulenceCalculator error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
print("\nThe Turbulence Forecast application is ready to use.")
print("No errors detected in core functionality.\n")

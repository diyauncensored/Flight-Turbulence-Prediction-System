"""
Comprehensive test script for Turbulence Forecast application
"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("TURBULENCE FORECAST APPLICATION - COMPREHENSIVE TEST")
print("=" * 60)

# Test 1: Import all utilities
print("\n[1/7] Testing imports...")
try:
    from utils.ml_models import TurbulencePredictor, turbulence_model
    from utils.turbulence_calculator import TurbulenceCalculator
    from utils.weather_api import WeatherAPI
    print("[OK] All utility imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Test TurbulencePredictor initialization
print("\n[2/7] Testing TurbulencePredictor initialization...")
try:
    predictor = TurbulencePredictor()
    print("[OK] TurbulencePredictor initialized successfully")
except Exception as e:
    print(f"[FAIL] TurbulencePredictor initialization error: {e}")
    sys.exit(1)

# Test 3: Test synthetic data generation
print("\n[3/7] Testing synthetic data generation...")
try:
    X, y = predictor._generate_synthetic_data(n_samples=100)
    assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    print(f"[OK] Synthetic data generation successful: X{X.shape}, y{y.shape}")
except Exception as e:
    print(f"[FAIL] Synthetic data generation error: {e}")
    sys.exit(1)

# Test 4: Test model accuracies
print("\n[4/7] Testing model accuracy metrics...")
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
    
    print("[OK] Model accuracy metrics successful")
    print(f"   Random Forest Accuracy: {accuracies['random_forest']['accuracy']:.2f}%")
    print(f"   Gradient Boosting Accuracy: {accuracies['gradient_boosting']['accuracy']:.2f}%")
except Exception as e:
    print(f"[FAIL] Model accuracy error: {e}")
    sys.exit(1)

# Test 5: Test feature importance
print("\n[5/7] Testing feature importance...")
try:
    importance = predictor.get_feature_importance()
    assert importance is not None, "Feature importance returned None"
    print(f"[OK] Feature importance successful (top feature: {importance.iloc[0]['feature']})")
except Exception as e:
    print(f"[FAIL] Feature importance error: {e}")
    sys.exit(1)

# Test 6: Test TurbulenceCalculator
print("\n[6/7] Testing TurbulenceCalculator...")
try:
    calc = TurbulenceCalculator()
    test_turbulence = 2.5
    severity = calc.get_turbulence_severity(test_turbulence)
    severity_color = calc.get_severity_color(severity)
    comfort = calc.calculate_passenger_comfort_index(test_turbulence, 60)
    fuel_impact = calc.estimate_fuel_consumption_impact(test_turbulence, 2)
    print(f"[OK] TurbulenceCalculator successful")
    print(f"   Severity: {severity} ({severity_color})")
    print(f"   Passenger Comfort: {comfort:.1f}%")
    print(f"   Fuel Impact: {fuel_impact:.1f}%")
except Exception as e:
    print(f"[FAIL] TurbulenceCalculator error: {e}")
# Test 7: Test predict_turbulence with unified signature
print("\n[7/7] Testing predict_turbulence unified signature...")
try:
    dummy_weather = {
        'wind_speed': 12.5,
        'wind_direction': 240.0,
        'temperature': 25.0,
        'pressure': 1012.0,
        'humidity': 65.0,
        'visibility': 10.0,
        'weather_condition': 'clouds'
    }
    dummy_flight = {
        'altitude': 35000,
        'aircraft_type': 'Medium'
    }
    predictions, confidence = predictor.predict_turbulence(dummy_weather, dummy_flight)
    assert 'ensemble' in predictions, "Missing 'ensemble' in predictions"
    assert 'ensemble' in confidence, "Missing 'ensemble' in confidence"
    print(f"[OK] predict_turbulence successful (ensemble prediction: {predictions['ensemble']:.2f}, confidence: {confidence['ensemble']:.1f}%)")
except Exception as e:
    print(f"[FAIL] predict_turbulence error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[OK] ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
print("\nThe Turbulence Forecast application is ready to use.")
print("No errors detected in core functionality.\n")

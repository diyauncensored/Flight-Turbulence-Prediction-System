import numpy as np
import os
import pickle
from datetime import datetime

# Try importing TensorFlow/Keras with fallback
try:
    from tensorflow import keras
    from keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None
    layers = None

class LSTMTurbulencePredictor:
    """LSTM Neural Network for turbulence prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sequence_length = 10
        self.model_path = "models/lstm_turbulence.keras"
        self.scaler_path = "models/scaler.pkl"
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load pre-trained LSTM model"""
        if not TF_AVAILABLE:
            self.model = None
            self.scaler = {
                'mean': np.array([15, 180, 25, 1013, 60, 10, 35000, 12]),
                'std': np.array([10, 180, 15, 20, 25, 5, 10000, 8])
            }
            return
        
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except:
                self._build_model()
        else:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM neural network architecture"""
        if not TF_AVAILABLE:
            return
        
        # Input features: wind_speed, wind_direction, temperature, pressure, humidity, visibility, altitude, time_of_day
        input_shape = (self.sequence_length, 8)
        
        model = keras.Sequential([
            # First LSTM layer with dropout
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            # Third LSTM layer
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='linear')  # Turbulence index 0-10
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        # Create simple scaler for normalization
        self.scaler = {
            'mean': np.array([15, 180, 25, 1013, 60, 10, 35000, 12]),
            'std': np.array([10, 180, 15, 20, 25, 5, 10000, 8])
        }
    
    def prepare_sequence(self, weather_history):
        """Prepare sequence data for LSTM input"""
        # If we don't have enough history, pad with synthetic data
        if len(weather_history) < self.sequence_length:
            # Generate synthetic historical data based on current conditions
            current = weather_history[-1] if weather_history else self._get_default_weather()
            
            synthetic_history = []
            for i in range(self.sequence_length - len(weather_history)):
                # Add slight variations to create realistic sequence
                variation = 1 - (i * 0.05)
                synthetic_point = {
                    'wind_speed': current['wind_speed'] * variation + np.random.normal(0, 1),
                    'wind_direction': current['wind_direction'] + np.random.normal(0, 10),
                    'temperature': current['temperature'] + np.random.normal(0, 1),
                    'pressure': current['pressure'] + np.random.normal(0, 2),
                    'humidity': current['humidity'] + np.random.normal(0, 5),
                    'visibility': current.get('visibility', 10) + np.random.normal(0, 1),
                    'altitude': current.get('altitude', 35000),
                    'time_of_day': current.get('time_of_day', 12)
                }
                synthetic_history.append(synthetic_point)
            
            weather_history = synthetic_history + weather_history
        
        # Convert to feature matrix
        features = []
        for weather in weather_history[-self.sequence_length:]:
            features.append([
                weather['wind_speed'],
                weather['wind_direction'],
                weather['temperature'],
                weather['pressure'],
                weather['humidity'],
                weather.get('visibility', 10),
                weather.get('altitude', 35000),
                weather.get('time_of_day', 12)
            ])
        
        return np.array(features)
    
    def _get_default_weather(self):
        """Get default weather conditions"""
        return {
            'wind_speed': 10,
            'wind_direction': 180,
            'temperature': 25,
            'pressure': 1013,
            'humidity': 60,
            'visibility': 10,
            'altitude': 35000,
            'time_of_day': 12
        }
    
    def normalize_data(self, data):
        """Normalize input data"""
        return (data - self.scaler['mean']) / self.scaler['std']
    
    def denormalize_prediction(self, prediction):
        """Denormalize output prediction"""
        # Prediction is turbulence index 0-10, no denormalization needed
        return np.clip(prediction, 0, 10)
    
    def predict(self, weather_data, flight_params):
        """Predict turbulence using LSTM model"""
        # Add flight parameters to weather data
        current_weather = {
            'wind_speed': weather_data.get('wind_speed', 10),
            'wind_direction': weather_data.get('wind_direction', 180),
            'temperature': weather_data.get('temperature', 25),
            'pressure': weather_data.get('pressure', 1013),
            'humidity': weather_data.get('humidity', 60),
            'visibility': weather_data.get('visibility', 10),
            'altitude': flight_params.get('altitude', 35000),
            'time_of_day': datetime.now().hour
        }
        
        # Fallback to simple calculation if TF not available
        if not TF_AVAILABLE or self.model is None:
            # Simple turbulence calculation
            turbulence_index = self._calculate_turbulence_from_conditions([
                current_weather['wind_speed'],
                current_weather['wind_direction'],
                current_weather['temperature'],
                current_weather['pressure'],
                current_weather['humidity'],
                current_weather['visibility'],
                current_weather['altitude'],
                current_weather['time_of_day']
            ])
            
            return {
                'turbulence_index': float(turbulence_index),
                'confidence': 0.75,
                'std_dev': 0.5,
                'model_type': 'Simple Calculation (TensorFlow not available)'
            }
        
        # Create sequence (using current + synthetic history)
        weather_history = [current_weather]
        sequence = self.prepare_sequence(weather_history)
        
        # Normalize
        sequence_normalized = self.normalize_data(sequence)
        
        # Reshape for LSTM input [batch_size, sequence_length, features]
        sequence_input = sequence_normalized.reshape(1, self.sequence_length, 8)
        
        # Predict
        prediction = self.model.predict(sequence_input, verbose=0)
        turbulence_index = self.denormalize_prediction(prediction[0][0])
        
        # Calculate confidence based on model uncertainty
        # Use multiple predictions with dropout enabled for uncertainty estimation
        predictions = []
        for _ in range(10):
            pred = self.model.predict(sequence_input, verbose=0)
            predictions.append(self.denormalize_prediction(pred[0][0]))
        
        predictions = np.array(predictions)
        std_dev = np.std(predictions)
        
        # Confidence inversely related to uncertainty
        confidence = max(0.5, min(0.95, 1 - (std_dev / 10)))
        
        return {
            'turbulence_index': float(turbulence_index),
            'confidence': float(confidence),
            'std_dev': float(std_dev),
            'model_type': 'LSTM Neural Network'
        }
    
    def train_on_new_data(self, X_train, y_train, epochs=50, batch_size=32):
        """Train model on new data (for future improvements)"""
        # Ensure model is built
        if self.model is None:
            self._build_model()
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Save model
        os.makedirs("models", exist_ok=True)
        self.model.save(self.model_path)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return history
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data for initial model"""
        X_train = []
        y_train = []
        
        for _ in range(num_samples):
            # Generate random weather sequence
            sequence = []
            
            # Base conditions
            base_wind = np.random.uniform(0, 40)
            base_temp = np.random.uniform(-40, 40)
            base_pressure = np.random.uniform(900, 1040)
            
            for i in range(self.sequence_length):
                wind_speed = base_wind + np.random.normal(0, 5)
                wind_direction = np.random.uniform(0, 360)
                temperature = base_temp + np.random.normal(0, 2)
                pressure = base_pressure + np.random.normal(0, 5)
                humidity = np.random.uniform(20, 100)
                visibility = np.random.uniform(1, 15)
                altitude = np.random.uniform(20000, 45000)
                time_of_day = np.random.uniform(0, 24)
                
                sequence.append([
                    wind_speed, wind_direction, temperature, pressure,
                    humidity, visibility, altitude, time_of_day
                ])
            
            # Calculate turbulence index based on conditions
            final_conditions = sequence[-1]
            turbulence = self._calculate_turbulence_from_conditions(final_conditions)
            
            X_train.append(sequence)
            y_train.append(turbulence)
        
        return np.array(X_train), np.array(y_train)
    
    def _calculate_turbulence_from_conditions(self, conditions):
        """Calculate turbulence index from weather conditions"""
        wind_speed, wind_dir, temp, pressure, humidity, visibility, altitude, time = conditions
        
        # Turbulence factors
        wind_factor = min(abs(wind_speed) / 25, 3)
        temp_factor = min(abs(temp + 20) / 30, 2)  # Extreme temps increase turbulence
        pressure_factor = min(abs(pressure - 1013) / 50, 2)
        visibility_factor = min((15 - visibility) / 10, 1.5)
        altitude_factor = 1.2 if 25000 < altitude < 40000 else 0.8
        
        # Time factor (afternoon hours often more turbulent)
        time_factor = 1.2 if 12 <= time <= 18 else 1.0
        
        turbulence = (wind_factor + temp_factor + pressure_factor + 
                     visibility_factor) * altitude_factor * time_factor
        
        # Add some randomness
        turbulence += np.random.normal(0, 0.5)
        
        return np.clip(turbulence, 0, 10)
    
    def pretrain_model(self, num_samples=5000):
        """Pre-train model with synthetic data"""
        X_train, y_train = self.generate_training_data(num_samples)
        
        # Normalize
        X_train_normalized = np.array([self.normalize_data(seq) for seq in X_train])
        
        # Train
        history = self.train_on_new_data(X_train_normalized, y_train, epochs=100, batch_size=64)
        
        return history

# Global instance
lstm_predictor = LSTMTurbulencePredictor()

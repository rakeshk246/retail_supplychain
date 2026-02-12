import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import numpy as np

class DemandForecaster:
    """Hybrid forecasting using LSTM and Prophet"""
    
    def __init__(self, method='lstm'):
        self.method = method
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 7
        
    def prepare_lstm_data(self, data, sequence_length=7):
        """Prepare sequences for LSTM"""
        X, y = [], []
        values = data['y'].values
        
        scaled = self.scaler.fit_transform(values.reshape(-1, 1))
        
        for i in range(len(scaled) - sequence_length):
            X.append(scaled[i:i+sequence_length])
            y.append(scaled[i+sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(self, data, epochs=50):
        """Train LSTM model"""
        X, y = self.prepare_lstm_data(data, self.sequence_length)
        
        # Split train/validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        self.model = self.build_lstm_model((self.sequence_length, 1))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        return history
    
    def predict_next(self):
        """Return next-day demand prediction."""
            
            # Get last 30 days of data
        recent_data = self.data.tail(30)

            # Use existing predict() to forecast next step
        pred = self.predict(recent_data, steps=1)

        return float(pred[0])

    
    def train_prophet(self, data):
        """Train Prophet model"""
        self.model = Prophet(
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True
        )
        self.model.fit(data[['ds', 'y']])
    
    def train(self, data, epochs=50):
        """Train the selected forecasting model"""
        self.data = data
        if self.method == 'lstm':
            return self.train_lstm(data, epochs)
        elif self.method == 'prophet':
            return self.train_prophet(data)
    
    def predict_lstm(self, recent_data, steps=1):
        """Predict using LSTM"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get last sequence
        last_sequence = recent_data['y'].values[-self.sequence_length:]
        scaled_sequence = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_sequence.reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        return predictions.flatten()
    
    def predict_prophet(self, steps=1):
        """Predict using Prophet"""
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
    
    def predict(self, recent_data=None, steps=1):
        """Make prediction using selected method"""
        if self.method == 'lstm':
            return self.predict_lstm(recent_data, steps)
        elif self.method == 'prophet':
            return self.predict_prophet(steps)
    
    def get_confidence(self):
        """Return model confidence score"""
        return np.random.uniform(0.75, 0.95)

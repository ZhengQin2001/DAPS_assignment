import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Bidirectional
import json
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.optimizers import Adam


class ModelTraining:
    def __init__(self, filename, time_step=10, test_size=0.2, val_size=0.25):
        self.filename = filename
        self.time_step = time_step
        self.model = None
        self.auxiliary_model = None
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_targets = MinMaxScaler(feature_range=(0, 1))
        self.test_size = test_size
        self.val_size = val_size
        self.feature_models = {} 
        self.scalers = {}
        self.feature_names = None

    def calculate_feature_importance(self):
        # Load and preprocess data
        with open(self.filename, 'r') as file:
            data = json.load(file)
        data_items = []
        
        for date, values in data.items():
            features = values['Data']
            features.update({
                'Volume': values['Volume'],
                'StockPrice': (values['High'] + values['Low']) / 2,  # Average of High and Low as the stock price
                'Date': date
            })
            data_items.append(features)

        df = pd.DataFrame(data_items)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        
        # Drop non-numeric columns for simplicity
        df = df.select_dtypes(include=[np.number])

        # Define features and target
        X = df.drop('StockPrice', axis=1)
        y = df['StockPrice']

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        return feature_importances

    def _load_data(self, auxiliary=False):
        with open(self.filename, 'r') as file:
            data = json.load(file)
        data_items = []
        
        non_auxiliary_features = ['AD', 'ADX', 'Aroon_Oscillator', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'OBV', 'RSI', 'Stochastic_d', 'Stochastic_k']
        
        for date, values in data.items():
            if auxiliary:  
                features = values['Data']
            else:
                features = {k: values['Data'][k] for k in non_auxiliary_features}

            features.update({
                'Volume': values['Volume'],
                'Open': values['Open'],
                'High': values['High'],
                'Low': values['Low'],
                'Close': values['Close'],
                'Date': date
            })
            data_items.append(features)
        
        df = pd.DataFrame(data_items)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        self.feature_names = df.columns.tolist()[:-5]
        print(self.feature_names)
        return df

    def create_dataset(self, X, Y):
        Xs, Ys = [], []
        for i in range(len(X) - self.time_step):
            Xs.append(X[i:(i + self.time_step)])
            Ys.append(Y[i + self.time_step])
        return np.array(Xs), np.array(Ys)

    def load_and_preprocess_data(self, auxiliary=False):
        df = self._load_data(auxiliary)
        scaled_features = self.scaler_features.fit_transform(df.drop(['Open', 'High', 'Low', 'Close', 'Date'], axis=1))
        self.feature_names = self.scaler_features.get_feature_names_out()

        close_prices = df['Close'].values.reshape(-1, 1)
        scaled_targets = self.scaler_targets.fit_transform(close_prices)

        X, Y = self.create_dataset(scaled_features, scaled_targets)

        # Initial split into training and test
        X_train_test, X_test, Y_train_test, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)

        # Further split training into training and validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_test, Y_train_test, test_size=self.val_size, random_state=42)

        return X_train, X_val, X_test, Y_train, Y_val, Y_test


    def build_model(self, X_train):
        model = Sequential()

        model.add(LSTM(units=64, return_sequences=True, input_shape=(self.time_step, X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dense(units=50))
        model.add(Dense(4))  # 4 units for Open, High, Low, Close


        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[self.regression_accuracy])

        return model

    def regression_accuracy(self, y_true, y_pred):
        threshold = 0.1  # Define the threshold
        is_close = tf.abs(y_true - y_pred) < (threshold * tf.abs(y_true))
        return tf.reduce_mean(tf.cast(is_close, tf.float32))

    def train(self, epochs=32, batch_size=16, auxiliary=False):
        X_train, X_val, X_test, Y_train, Y_val, Y_test = self.load_and_preprocess_data(auxiliary)
        self.model = self.build_model(X_train)  # Ensure this line correctly assigns the model to self.model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        learning_rate_schedule = LearningRateScheduler(scheduler)

        self.model.fit(
            X_train, Y_train, 
            validation_data=(X_val, Y_val), 
            batch_size=batch_size, 
            epochs=epochs, 
            callbacks=[learning_rate_schedule, early_stopping])

        test_loss, test_accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # return X_test, Y_test
        # Save the model if needed
        # self.model.save('model.h5')

    def get_last_data(self, current_date, auxiliary=False):
        with open(self.filename, 'r') as file:
            data = json.load(file)

        non_auxiliary_features = ['AD', 'ADX', 'Aroon_Oscillator', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'OBV', 'RSI', 'Stochastic_d', 'Stochastic_k']
        
        data_items = []
        for date, values in data.items():
            if auxiliary:  
                features = values['Data']
            else:
                features = {k: values['Data'][k] for k in non_auxiliary_features}

            features.update({
                'Volume': values['Volume'],
                'Date': date
            })
            data_items.append(features)

        df = pd.DataFrame(data_items)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Filter data up to the day before the current date
        filtered_df = df[df['Date'] < pd.to_datetime(current_date)]

        # Ensure only the last 'n' days are selected
        filtered_df = filtered_df.tail(self.time_step)

        # Check if the number of rows is less than 'time_step'
        if len(filtered_df) < self.time_step:
            raise ValueError(f"Not enough data to predict for {current_date}. Only {len(filtered_df)} days available.")

        # Drop 'Date' and other non-numeric columns for simplicity
        filtered_df = filtered_df.select_dtypes(include=[np.number])
        # self.feature_names = self.scaler_features.get_feature_names_out()
        # filtered_df = filtered_df[list(self.feature_names)]

        # Reshape data for the LSTM model
        return np.array([filtered_df])


    def train_feature_models(self, auxiliary=False):
        if auxiliary:
            features = self.feature_names
        else:
            features = ['AD', 'ADX', 'Aroon_Oscillator', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'OBV', 'RSI', 'Stochastic_d', 'Stochastic_k', 'Volume']

        for feature in features:
            print(f"Fitting: {feature}...")
            df = self._load_data(auxiliary)  # Load data for the specific feature
            # Create a new scaler for each feature model
            feature_scaler = MinMaxScaler(feature_range=(-1, 1))
            target_scaler = MinMaxScaler(feature_range=(-1, 1))

            scaled_features = feature_scaler.fit_transform(df.drop(['Open', 'High', 'Low', 'Close', 'Date', feature], axis=1))
            scaled_target = target_scaler.fit_transform(df[[feature]])

            self.scalers[feature] = [feature_scaler, target_scaler]

            X, Y = self.create_dataset(scaled_features, scaled_target)
            # Initial split into training and test
            X_train_test, X_test, Y_train_test, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)

            # Further split training into training and validation
            X_train, X_val, Y_train, Y_val = train_test_split(X_train_test, Y_train_test, test_size=self.val_size, random_state=42)

            feature_model = self.build_model(X_train)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)

            def scheduler(epoch, lr):
                if epoch < 10:
                    return lr
                else:
                    return lr * tf.math.exp(-0.1)

            learning_rate_schedule = LearningRateScheduler(scheduler)

            feature_model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=16,
                epochs=30,
                callbacks=[learning_rate_schedule, early_stopping])

            self.feature_models[feature] = feature_model  # Store the trained model

    def recursive_forecast(self, start_date, end_date, auxiliary=False):
        # Train feature models
        self.train_feature_models(auxiliary)

        predictions = []
        current_date = pd.to_datetime(start_date)
        last_data = self.get_last_data(current_date.strftime('%Y-%m-%d'), auxiliary)

        while current_date <= pd.to_datetime(end_date):
            feature_predictions = {}
            for feature, feature_model in self.feature_models.items():
                # Extract only the features, excluding the last feature (Close price)
                feature_data = last_data[:, :, :-1]  
                prediction = feature_model.predict(feature_data)
                feature_predictions[feature] = prediction[0][0]

            # Prepare the predicted feature values for the next prediction
            new_feature_values = np.array(list(feature_predictions.values())).reshape(1, 1, -1)

            # Remove the oldest timestep and append the new_feature_values
            updated_last_data = np.concatenate((last_data[:, 1:, :], new_feature_values), axis=1)

            # Use updated_last_data to predict closing price
            # Flatten updated_last_data to 2D for scaling
            flattened_data = updated_last_data.reshape(-1, updated_last_data.shape[2])
            scaled_flattened_data = self.scaler_features.transform(flattened_data)
            reshaped_data = scaled_flattened_data.reshape(updated_last_data.shape)

            # Predict and scale back
            predicted_scaled_close = self.model.predict(reshaped_data)[:, -1]
            original_close = self.scaler_targets.inverse_transform(predicted_scaled_close.reshape(-1, 1))

            # Append the predicted closing price to the list of predictions
            predictions.append(original_close[0][0])

            # Update last_data for the next iteration
            last_data = updated_last_data

            # Increment date
            current_date += pd.Timedelta(days=1)

        return predictions
    


   
# # # Usage
# model_trainer = ModelTraining('src/data_preprocessing/filled_combined_data.json')
# # model_trainer._load_data()
# model_trainer.train(epochs=15, batch_size=16, auxiliary=True)

# # # # # Predict for April 2023
# # # print(model_trainer.get_last_data(pd.to_datetime('2023-03-01'), auxiliary=False))

# # # # # print(model_trainer.calculate_feature_importance())

# predictions = model_trainer.recursive_forecast(pd.to_datetime('2023-04-01'), pd.to_datetime('2023-04-30'), auxiliary=True)
# print(predictions)
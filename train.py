"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.api.models import Model
from keras.api.callbacks import EarlyStopping
import xgboost as xgb
warnings.filterwarnings("ignore")



def train_model(model, X_train, y_train, name, config, location):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[early],
        )
    
    model.save('model/' + name + '/' + location + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + '/' + location + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config, location):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto', restore_best_weights=True)

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(inputs=p.inputs,
                                       outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, temp, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config,location)


def add_datetime_features(X_train, train_file):
    """
    Add datetime features to the training data
    """
    import pandas as pd
    import numpy as np
    
    # Load the original training data to get timestamps
    train_data = pd.read_csv(train_file)
    
    # Check if there's a timestamp column
    timestamp_col = None
    for col in train_data.columns:
        if col.lower() in ['time', 'timestamp', 'date', 'datetime']:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("Warning: No timestamp column found. Using synthetic datetime features.")
        # If no timestamp column exists, create synthetic datetime features
        return add_synthetic_datetime_features(X_train)
    
    # Convert timestamps to datetime
    train_data[timestamp_col] = pd.to_datetime(train_data[timestamp_col])
    
    # Extract datetime features
    datetime_features = np.zeros((X_train.shape[0], 4))
    
    # We need to account for the lag, since X_train includes lagged features
    # The timestamps we want correspond to the target timestamps, which are lag steps ahead
    for i in range(X_train.shape[0]):
        sample_idx = i + 12  
        
        if sample_idx < len(train_data):
            timestamp = train_data[timestamp_col].iloc[sample_idx]
            
            # Extract features
            datetime_features[i, 0] = timestamp.hour
            datetime_features[i, 1] = timestamp.weekday()
            datetime_features[i, 2] = timestamp.month
            datetime_features[i, 3] = 1 if timestamp.weekday() >= 5 else 0  # Weekend feature
    
    # Concatenate the original features with datetime features
    X_train_with_datetime = np.concatenate([X_train, datetime_features], axis=1)
    
    return X_train_with_datetime

def add_synthetic_datetime_features(X_train):
    """
    Add synthetic datetime features when real timestamps are not available
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create a starting date
    start_date = datetime(2023, 1, 1)
    
    # Create datetime features for each sample
    datetime_features = np.zeros((X_train.shape[0], 4))
    
    for i in range(X_train.shape[0]):
        # Create a synthetic date that advances by 15 minutes for each sample
        current_date = start_date + timedelta(minutes=15 * i)
        
        # Extract features
        datetime_features[i, 0] = current_date.hour
        datetime_features[i, 1] = current_date.weekday()
        datetime_features[i, 2] = current_date.month
        datetime_features[i, 3] = 1 if current_date.weekday() >= 5 else 0  # Weekend feature
    
    # Concatenate the original features with datetime features
    X_train_with_datetime = np.concatenate([X_train, datetime_features], axis=1)
    
    return X_train_with_datetime



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(
        "--location",
        help="Scat location to train for"
    )
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 50}
    file1 = f'data/train_data/{args.location}_flow.csv'
    file2 = f'data/test_data/{args.location}_flow.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config,args.location)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config, args.location) 
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        # m = model.get_saes([12, 400, 400, 400, 1])
        m = model.get_saes([12, 256, 128, 64, 1])
        train_seas(m, X_train, y_train, args.model, config, args.location)
    if args.model == 'xgboost':
        import json
        # Add datetime features to training data
        X_train_with_datetime = add_datetime_features(X_train, file1)
        print(f"X_train_with_datetime shape: {X_train_with_datetime.shape}")
        
        # Reshape the data
        X_train_with_datetime = X_train_with_datetime.reshape(X_train_with_datetime.shape[0], X_train_with_datetime.shape[1])
        
        # Create DMatrix with the enhanced features
        dtrain = xgb.DMatrix(X_train_with_datetime, label=y_train)
       
        # Define XGBoost parameters
        params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100
                }
        
        # Train the model
        m = xgb.train(params, dtrain, num_boost_round=100)
        print(f"Number of features passed to XGBoost: {X_train_with_datetime.shape[1]}")

        # Save model
        m.save_model('model/' + args.model + '/' + args.location + '.json')
        
        # Save feature information for prediction
        feature_info = {
            'num_features': X_train_with_datetime.shape[1],
            'includes_datetime': True
        }
        
        with open('model/' + args.model + '/' + args.location + '_features.json', 'w') as f:
            json.dump(feature_info, f)
            
        print(f"XGBoost model training complete! Model saved with {X_train_with_datetime.shape[1]} features.")


def add_datetime_features(X_train, train_file):
    """
    Add datetime features to the training data
    """
    import pandas as pd
    import numpy as np
    
    # Load the original training data to get timestamps
    train_data = pd.read_csv(train_file)
    
    # Check if there's a timestamp column
    timestamp_col = None
    for col in train_data.columns:
        if col.lower() in ['time', 'timestamp', 'date', 'datetime']:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("Warning: No timestamp column found. Using synthetic datetime features.")
        # If no timestamp column exists, create synthetic datetime features
        return add_synthetic_datetime_features(X_train)
    
    # Convert timestamps to datetime
    train_data[timestamp_col] = pd.to_datetime(train_data[timestamp_col])
    
    # Extract datetime features
    datetime_features = np.zeros((X_train.shape[0], 4))


    
    # We need to account for the lag, since X_train includes lagged features
    # The timestamps we want correspond to the target timestamps, which are lag steps ahead
    for i in range(X_train.shape[0]):
        # Find the corresponding timestamp for this sample
        # The index in the original data depends on how process_data creates the training samples
        # This is an approximation - you may need to adjust based on your exact data processing
        sample_idx = i + 12  # Adjust this based on how your data is processed
        
        if sample_idx < len(train_data):
            timestamp = train_data[timestamp_col].iloc[sample_idx]
            
            # Extract features
            datetime_features[i, 0] = timestamp.hour
            datetime_features[i, 1] = timestamp.weekday()
            datetime_features[i, 2] = timestamp.month
            datetime_features[i, 3] = 1 if timestamp.weekday() >= 5 else 0  # Weekend feature
    
    print("Datetime features example:")
    print(datetime_features[:5])  # Print the first 5 rows to check the datetime features
    
    # Concatenate the original features with datetime features
    X_train_with_datetime = np.concatenate([X_train, datetime_features], axis=1)
    
    return X_train_with_datetime



def add_synthetic_datetime_features(X_train):
    """
    Add synthetic datetime features when real timestamps are not available
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create a starting date
    start_date = datetime(2023, 1, 1)
    
    # Create datetime features for each sample
    datetime_features = np.zeros((X_train.shape[0], 4))
    
    for i in range(X_train.shape[0]):
        # Create a synthetic date that advances by 15 minutes for each sample
        current_date = start_date + timedelta(minutes=15 * i)
        
        # Extract features
        datetime_features[i, 0] = current_date.hour
        datetime_features[i, 1] = current_date.weekday()
        datetime_features[i, 2] = current_date.month
        datetime_features[i, 3] = 1 if current_date.weekday() >= 5 else 0  # Weekend feature
    
    # Concatenate the original features with datetime features
    X_train_with_datetime = np.concatenate([X_train, datetime_features], axis=1)
    
    return X_train_with_datetime




if __name__ == '__main__':
    main(sys.argv)
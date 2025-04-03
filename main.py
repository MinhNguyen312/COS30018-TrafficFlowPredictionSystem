"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.api.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
from keras.api.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
import argparse
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def main():
    # lstm = load_model('model/lstm.h5')
    # gru = load_model('model/gru.h5')
    # saes = load_model('model/saes.h5')
    # models = [lstm, gru, saes]
    # names = ['LSTM', 'GRU', 'SAEs']

    # lag = 12
    # file1 = 'data/train.csv'
    # file2 = 'data/test.csv'


    #Testing LSTM 
    # Load all models
    # try:
    #     lstm = load_model('model/lstm.h5', 
    #                      custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    #     gru = load_model('model/gru.h5',
    #                     custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    #     # saes = load_model('model/network_saes.h5',
    #     #                  custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    #     xgboost = xgb.Booster()
    #     xgboost.load_model('model/xgboost.json')
        
    #     models = [lstm,gru,xgboost]
    #     names = ['LSTM', 'GRU','XGBoost']
    # except Exception as e:
    #     print(f"Error loading models: {e}")
    #     return

    # # Evaluation parameters
    # lag = 12
    # location = '2000'  # Example SCATS location
    
    # # Correct file order: train first, test second
    # file1 = f'data/train_data/{location}_flow.csv'
    # file2 = f'data/test_data/{location}_flow.csv'
    
    # # Process data
    # _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    # original_X = X_test.copy()  # Keep original shape
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # print(f"\nEvaluating models on SCATS location {location}")
    # print("=" * 50)
    
    # y_preds = []
    # for name, model in zip(names, models):
    #     print(f"\nModel: {name}")
    #     print("-" * 20)
        
    #     # Reshape data according to model type
    #     if name == 'SAEs':
    #         X_test = np.reshape(original_X, (original_X.shape[0], original_X.shape[1]))
    #         predicted = model.predict(X_test)
    #     elif name == 'XGBoost':
    #         X_test = np.reshape(original_X, (original_X.shape[0], original_X.shape[1]))
    #         dtest = xgb.DMatrix(X_test)
    #         predicted = model.predict(dtest)
    #         predicted = predicted.reshape(-1, 1)
    #     else:  # LSTM and GRU
    #         X_test = np.reshape(original_X, (original_X.shape[0], original_X.shape[1], 1))
    #         predicted = model.predict(X_test)
        
    #     # Scale back predictions and evaluate
    #     predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    #     y_preds.append(predicted[:288])  # First day predictions for plotting
        
    #     eva_regress(y_test, predicted)

    # # Plot results
    # plot_results(y_test[:288], y_preds, names)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
    )

    args = parser.parse_args()

    try:
        # Load models and location mapping
        lstm = load_model(f'model/{args.location}/lstm.h5', 
                         custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        gru = load_model(f'model/{args.location}/gru.h5',
                        custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        xgboost = xgb.Booster()
        xgboost.load_model(f'model/{args.location}/xgboost.json')
        
        # Load location mapping
        location_to_idx = np.load('model/location_mapping.npy', allow_pickle=True).item()
        
        models = [lstm, gru, xgboost]
        names = ['LSTM', 'GRU', 'XGBoost']
    except Exception as e:
        print(f"Error loading models or mapping: {e}")
        return

    # Evaluation parameters
    lag = 12
    
    # Create location one-hot encoding
    location_onehot = np.zeros(len(location_to_idx))
    location_onehot[location_to_idx[args.location]] = 1
    
    # Process data
    file1 = f'data/train_data/{args.location}_flow.csv'
    file2 = f'data/test_data/{args.location}_flow.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    original_X = X_test.copy()

    # Add location features to test data
    X_test_with_loc = np.hstack([X_test, np.tile(location_onehot, (X_test.shape[0], 1))])
    
    # Scale back y_test
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    
    print(f"\nEvaluating models on SCATS location {args.location}")
    print("=" * 50)
    
    y_preds = []
    for name, model in zip(names, models):
        print(f"\nModel: {name}")
        print("-" * 20)
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        if name == 'XGBoost':
            # Reshape data for XGBoost
            X_test_xgb = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            # Convert to DMatrix format required by XGBoost
            dtest = xgb.DMatrix(X_test_xgb)
            # Make predictions
            predicted = model.predict(dtest)
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # file = 'images/' + name + '.png'
        # plot_model(model, to_file=file, show_shapes=True)
            predicted = model.predict(X_test)

        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)


if __name__ == '__main__':
    main()

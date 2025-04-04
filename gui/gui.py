import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tkinter import * 
from tkinter import ttk
import tkinter as tk
import pandas as pd
from route_finding.world import World
from data.data import process_data
from keras.api.models import load_model 
import tensorflow as tf
import numpy as np
from datetime import datetime

def model_selection():
    selected_model = combo_box.get()
    return selected_model

def compute_origin():
    origin_id = origin_entry.get()
    return origin_id
    
def compute_destination():
    destination_id = destination_entry.get()
    return destination_id

def compute_date():
    date = date_entry.get()
    print(date)
    return date




# Read data file
site_data = pd.read_csv("../data/Scats Data.csv", encoding='utf-8', sep='\t').fillna(0)

app = tk.Tk()
app.geometry("600x600")

app.grid_columnconfigure(0, weight=1)
app.resizable(False, False)

app.title("Traffic Flow Prediction System")

origin_label = Label(app, text='Origin: *', font=("Arial", 13)).grid(row=0, pady=5)
destination_label = Label(app, text='Destination: *', font=("Arial", 13)).grid(row=2, pady=5)
combo_box_label = tk.Label(app, text="Select your preferred model: *", font=("Arial", 13)).grid(row=4, pady=5)
date_label = tk.Label(app, text="Date/Time: *", font=("Arial", 13)).grid(row=6, pady=5)

# Create a Combobox widget
combo_box = ttk.Combobox(app, width=30, values=["LSTM", "GRU", "SAES", "XGBOOST"], state="readonly")
combo_box.set("LSTM")
combo_box.grid(row=5, pady=5)

origin_entry = Entry(app, width=30,)
origin_entry.grid(row=1, pady=5)

destination_entry = Entry(app, width=30,)
destination_entry.grid(row=3, pady=5)

date_entry = Entry(app, width=30,)
date_entry.grid(row=7, pady=5)
date_entry.insert(0, "2025-04-04 12:00:00")


# Create a Text widget to display multi-line text, but set it to readonly (non-editable)
text_box = tk.Text(app, height=15, width=60)
text_box.config(state=tk.DISABLED)  # Set to read-only (non-editable)
text_box.grid(row=9, pady=20)

def find_route(site_data):
    origin_id = compute_origin()
    destination_id = compute_destination()
    date = compute_date()
    model = model_selection()

    if not origin_id:
        origin_id = 970
    if not destination_id:
        destination_id = 3001

    world = World(site_data, int(origin_id), int(destination_id),date, model)
    paths = world.search_no_param()

    return paths

def button_listener():
    result = find_route(site_data)
    text_data = ""

    if result:
        i = 1
        for path in result:
            print(f"{date_entry.get()}")
            print(f"----- Route {i} -----")
            text_data += f"----- Route {i} -----\n"
            print(path)
            text_data += str(path)
            text_data += "\n\n"
            print("\n")
            i += 1
    else:
        error_msg = "Error: Fail to compute path.\n"
        print(error_msg)
        print("-------------------------------")
        text_data += error_msg

    text_box.config(state=tk.NORMAL)  # Make the Text widget editable temporarily
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, text_data)


# def predict():
#     origin_entry = compute_origin()
#     date_entry = compute_date()
#     model_type = combo_box.get()
#     try:
#         # Convert the datetime string to a datetime object
#         prediction_datetime = datetime.strptime(date_entry, '%Y-%m-%d %H:%M:%S')
#     except ValueError:
#         print("Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS'.")
#         return
#     print(f"Origin: {origin_entry}, Date: {prediction_datetime}, Model: {model_type}")

#     # Extract features from the datetime for the model
#     hour_of_day = prediction_datetime.hour
#     day_of_week = prediction_datetime.weekday()  # 0=Monday, 6=Sunday
#     month_of_year = prediction_datetime.month
#     is_weekend = 1 if day_of_week >= 5 else 0  # 1 if Saturday/Sunday, else 0

#     # For example, you can create an array of these datetime-based features
#     datetime_features = np.array([[hour_of_day, day_of_week, month_of_year, is_weekend]])

#     # Get model and data settings (origin, date, model type)
#     origin_entry = compute_origin()  # Example function to get origin data
#     model_type = combo_box.get()  # Assuming combo_box is a drop-down menu with model options

#     print(f"Prediction Date/Time: {date_entry}, Model Type: {model_type}, Origin: {origin_entry}")
    
    

#     lag=12

#     file1 = f'../data/train_data/{origin_entry}_flow.csv'
#     file2 = f'../data/test_data/{origin_entry}_flow.csv'
#     _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
#     original_X = X_test.copy()
#     y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

#     model_type = model_type.lower()
#     if model_type == 'lstm':
#         lstm = load_model(f'../model/{model_type}/{origin_entry}.h5', 
#                         custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#         predicted = lstm.predict(X_test)

#     elif model_type == 'gru':
#         gru = load_model(f'../model/{model_type}/{origin_entry}.h5', 
#                         custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#         predicted = gru.predict(X_test)
        
#     elif model_type == 'saes':
#         pass
#     else:
#         print("Error: Invalid model name.")
            
#     # Inverse transform the predicted values to bring them back to original scale
#     predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

#     last_sample_index = original_X.shape[0] - 1
#     sample_to_predict = original_X[last_sample_index:last_sample_index+1, :]

#     # Now concatenate with datetime features (both will have first dimension = 1)
#     X_test_with_datetime = np.concatenate([sample_to_predict, datetime_features], axis=1)
    
#     # If the model expects more features (including datetime), reshape the input accordingly
#     if model_type == 'lstm':
#         X_test_with_datetime = np.reshape(X_test_with_datetime, 
#                                      (X_test_with_datetime.shape[0], 
#                                       X_test_with_datetime.shape[1], 
#                                       1))
#         predicted_with_datetime = lstm.predict(X_test_with_datetime)
#     if model_type == 'gru':
#         X_test_with_datetime = np.reshape(X_test_with_datetime, 
#                                      (X_test_with_datetime.shape[0], 
#                                       X_test_with_datetime.shape[1], 
#                                       1))
#         predicted_with_datetime = gru.predict(X_test_with_datetime)
    
#     else:
#         predicted_with_datetime = predicted  # If other models are used, handle accordingly

#     # Inverse transform the predicted values (with datetime) to original scale
#     predicted_with_datetime = scaler.inverse_transform(predicted_with_datetime.reshape(-1, 1)).reshape(1, -1)[0]
    
#     # Output predicted values to the GUI Text Box
#     text_box.config(state=tk.NORMAL)  # Make the Text widget editable temporarily
#     text_box.delete(1.0, tk.END)
#     text_box.insert(tk.END, f"Predicted Traffic Volume at {date_entry}: {predicted_with_datetime[-1]}")
    

#     text_box.config(state=tk.NORMAL)  # Make the Text widget editable temporarily
#     text_box.delete(1.0, tk.END)
#     text_box.insert(tk.END, predicted)


    

ttk.Button(app, width=15, text="Find routes", command=button_listener).grid(row=8, pady=10)
# ttk.Button(app, width=15, text="Predict Scat", command=predict).grid(row=8, pady=10)



app.mainloop()
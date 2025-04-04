import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tkinter import * 
from tkinter import ttk
import tkinter as tk
import pandas as pd
from route_finding.world import World 

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
    return date

# Read data file
site_data = pd.read_csv("../data/Scats Data.csv", encoding='utf-8', sep='\t').fillna(0)

app = tk.Tk()
app.geometry("600x750")

app.grid_columnconfigure(0, weight=1)
app.resizable(False, False)

app.title("Traffic Flow Prediction System")

origin_label = Label(app, text='Origin: *', font=("Arial", 13)).grid(row=0, pady=5)
destination_label = Label(app, text='Destination: *', font=("Arial", 13)).grid(row=2, pady=5)
combo_box_label = Label(app, text="Select your preferred model: *", font=("Arial", 13)).grid(row=4, pady=5)
date_label = Label(app, text="Date/Time: *", font=("Arial", 13)).grid(row=6, pady=5)
predict_flow_label = Label(app, text="Enter a Scat ID to predict traffic flow:", font=("Arial", 13)).grid(row=8, pady=5)

# Create a Combobox widget
combo_box = ttk.Combobox(app, width=30, values=["LSTM", "GRU", "SAES", "XGBoost"], state="readonly")
combo_box.set("LSTM")
combo_box.grid(row=5, pady=5)

origin_entry = Entry(app, width=30,)
origin_entry.grid(row=1, pady=5)

destination_entry = Entry(app, width=30,)
destination_entry.grid(row=3, pady=5)

date_entry = Entry(app, width=30,)
date_entry.grid(row=7, pady=5)

predict_scat_entry = Entry(app, width=30)
predict_scat_entry.grid(row=9, pady=5)

# Create a Text widget to display multi-line text, but set it to readonly (non-editable)
text_box = tk.Text(app, height=20, width=60)
text_box.config(state=tk.DISABLED)  # Set to read-only (non-editable)
text_box.grid(row=11, pady=20)

def find_route(site_data):
    origin_id = compute_origin()
    destination_id = compute_destination()
    date = compute_date()
    model = model_selection()

    if not origin_id:
        origin_id = 970
    else: 
        origin_id = int(origin_id)
    if not destination_id:
        destination_id = 3001
    else:
        destination_id = int(destination_id)

    world = World(site_data, origin_id, destination_id)
    paths = world.search_no_param()

    return paths

def button_listener():
    result = find_route(site_data)
    text_data = ""

    if result:
        i = 1
        for path in result:
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

#TODO: Implement predict module
def handle_predict():
    scat_to_predict = predict_scat_entry.get()
    date = compute_date()

    if not scat_to_predict:
        scat_to_predict = 3001
    else:
        scat_to_predict = int(scat_to_predict)

    if not date:
        date = "2022/5/10 10:00"

    world = World(site_data, scat_to_predict=scat_to_predict)

    prediction = world.predict_traffic_flow(date)
    text_data = f"Predicted flow at Scat site {scat_to_predict} at {date}: {prediction}"

    text_box.config(state=tk.NORMAL)
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, text_data)

    

button_frame = tk.Frame(app)
button_frame.grid(row=10,pady=10)
ttk.Button(button_frame, width=15, text="Find routes", command=button_listener).pack(side="left",padx=5)
ttk.Button(button_frame, width=15, text="Predict", command=handle_predict).pack(side="right",padx=5)




app.mainloop()
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
app.geometry("600x600")

app.grid_columnconfigure(0, weight=1)
app.resizable(False, False)

app.title("Traffic Flow Prediction System")

origin_label = Label(app, text='Origin: *', font=("Arial", 13)).grid(row=0, pady=5)
destination_label = Label(app, text='Destination: *', font=("Arial", 13)).grid(row=2, pady=5)
combo_box_label = tk.Label(app, text="Select your preferred model: *", font=("Arial", 13)).grid(row=4, pady=5)
date_label = tk.Label(app, text="Date/Time: *", font=("Arial", 13)).grid(row=6, pady=5)

# Create a Combobox widget
combo_box = ttk.Combobox(app, width=30, values=["LSTM", "GRU", "SAES"], state="readonly")
combo_box.set("LSTM")
combo_box.grid(row=5, pady=5)

origin_entry = Entry(app, width=30,)
origin_entry.grid(row=1, pady=5)

destination_entry = Entry(app, width=30,)
destination_entry.grid(row=3, pady=5)

date_entry = Entry(app, width=30,)
date_entry.grid(row=7, pady=5)

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

ttk.Button(app, width=15, text="Find routes", command=button_listener).grid(row=8, pady=10)



app.mainloop()
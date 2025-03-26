from tkinter import * 
from tkinter import ttk
import tkinter as tk
import pandas as pd

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

def get_data(site_data):
    origin_id = compute_origin()
    destination_id = compute_destination()
    date = compute_date()
    model = model_selection()

    if not origin_id:
        origin_id = 970
    if not destination_id:
        destination_id = 3001


    print(site_data)
    print("--------------------")
    print(f"Origin: {origin_id}\nDestination: {destination_id}\nDate: {date}\nModel: {model}")

# Read data file
site_data = pd.read_csv("../data/Scats Data.csv", encoding='utf-8', sep='\t').fillna(0)

app = tk.Tk()
app.geometry("600x400")

app.grid_columnconfigure(0, weight=1)
app.resizable(False, False)

app.title("Traffic Flow Prediction System")

origin_label = Label(app, text='Origin: *', font=("Arial", 13)).grid(row=0, pady=5)
destination_label = Label(app, text='Destination: *', font=("Arial", 13)).grid(row=2, pady=5)
combo_box_label = tk.Label(app, text="Select your preferred model: *", font=("Arial", 13)).grid(row=4, pady=5)
date_label = tk.Label(app, text="Date/Time: *", font=("Arial", 13)).grid(row=6, pady=5)

# Create a Combobox widget
combo_box = ttk.Combobox(app, width=30, values=["Option 1", "Option 2", "Option 3"])
combo_box.set("Option 1")
combo_box.grid(row=5, pady=5)

origin_entry = Entry(app, width=30,)
origin_entry.grid(row=1, pady=5)

destination_entry = Entry(app, width=30,)
destination_entry.grid(row=3, pady=5)

date_entry = Entry(app, width=30,)
date_entry.grid(row=7, pady=5)

ttk.Button(app, width=15, text="Find routes", command=get_data(site_data)).grid(row=8, pady=10)

app.mainloop()
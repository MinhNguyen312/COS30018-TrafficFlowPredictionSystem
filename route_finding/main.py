import pandas as pd
from world import World


if __name__ == '__main__':
    # Read data file
    data = pd.read_csv("../data/Scats Data.csv", encoding='utf-8', sep='\t').fillna(0)

    origin = 4201
    destination = 3001

    
    world = World(data)

    # print(data)

    # Iterate through each row
    # for index, row in data.iterrows():
    #     # Access individual values from each row by column name
    #     scats_id = row['SCATS_Number']
    #     neighbors = row['SCATS Neighbours']
    #     latitude = row['NB_LATITUDE']
    #     longitude = row['NB_LONGITUDE']

    #     converted_neighbors = neighbors.split(" ")
    #     # print(f"SCATS ID: {scats_id}, Latitude: {latitude}, Longitude: {longitude}, Neighbors: {neighbors}")
    #     print(converted_neighbors)
        
import pandas as pd
from world import World


if __name__ == '__main__':
    # Read data file
    data = pd.read_csv("../data/Scats Data.csv", encoding='utf-8', sep='\t').fillna(0)

    origin = 970
    destination = 3001

    
    world = World(data, origin, destination)

        
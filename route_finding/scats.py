import math

class Scats(object):
    def __init__(self, scats_id, scat_name, latitude, longitude, neighbors = []):
        self.scats_id = scats_id
        self.scat_name = scat_name
        self.latitude = latitude
        self.longitude = longitude
        self.neighbors = []  # List to store neighbors (other SCATS_sites)

        self.gCost = float('inf')
        self.hCost = 0
        self.fCost = float('inf') # f = g + h
        self.parent = None
        self.distance_travelled = 0
        self.travel_time = 0

        for string in neighbors:
            self.neighbors.append(int(string))

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    # def set_distance(self, distance):
    #     self.distance = distance


    def __repr__(self):
        return f"SCATS_ID: {self.scats_id}, Name: {self.scat_name}, Latitude: {self.latitude}, Longitude: {self.longitude}, Neighbors: {self.neighbors}\n"
    
     # Compare SCATS based on their total cost (f = g + h)
    def __lt__(self, other):
        return self.fCost < other.fCost  # Compare total cost (f) for heapq ordering
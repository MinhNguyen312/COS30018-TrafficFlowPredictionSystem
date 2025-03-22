from scats import Scats
import math, heapq
from geopy.distance import geodesic as GD

class World(object):
    def __init__(self, data, origin = 970, destination = 3001):
        self.data = data
        self.origin = origin
        self.destination = destination
        self.scats = []

        for index, row in data.iterrows():
            scats_id = row['SCATS_Number']
            scat_name = row['Location']
            neighbors = row['SCATS Neighbours']
            latitude = row['NB_LATITUDE']
            longitude = row['NB_LONGITUDE']

            converted_neighbors = neighbors.split(" ")
            
            self.scats.append(Scats(int(scats_id), scat_name, float(latitude), float(longitude), converted_neighbors))


        print(self.scats)
        print("-------------------------------")
        
        
        path, total_travel_time = self.search(self.origin, self.destination, self.scats)

        # Output the result
        if path:
            print("Path found:")
            total_distance_travelled = 0
            
            for scat in path:
                print(f"{scat.scats_id} - {scat.scat_name}")
                total_distance_travelled = scat.distance_travelled  # Get the total distance traveled for the last SCATS in the path
            
            print(f"Length: {len(path)}")
            print(f"Total distance traveled: {total_distance_travelled} km")
            print(f"Total travel time: {total_travel_time / 60} mins")
        else:
            print("No path found")
        
    def calculateHeuristicCost(self, scat, destination_scat, speed):
        hCost = (math.sqrt((destination_scat.latitude - scat.latitude)**2 + (destination_scat.longitude - scat.longitude)**2) / speed) * 3600 #converting from hour -> sec
        return hCost
    
    def calculateTravelTime(self, distance, speed):
        intersection_delay = 30
        travel_time = distance/speed * 3600 #convert to seconds
        travel_time += intersection_delay

        return travel_time

    def search(self, origin, destination, scats = [], speed = 60):
        open_list = []
        closed_list = set()

        scats_dict = {scat.scats_id: scat for scat in scats}

        origin_scat = scats_dict.get(origin)
        destination_scat = scats_dict.get(destination)  

        origin_scat.gCost = 0
        origin_scat.hCost = self.calculateHeuristicCost(origin_scat, destination_scat, 60)
        origin_scat.fCost = origin_scat.gCost + origin_scat.hCost
        origin_scat.travel_time = 0

        heapq.heappush(open_list, origin_scat)  # Add origin scat to open list (priority queue)

        while open_list:
            # Get the scat with the lowest f value
            current_scat = heapq.heappop(open_list)

            # If we've reached the goal, reconstruct the path
            if (current_scat.scats_id == destination):
                path = []
                total_travel_time = current_scat.travel_time
                while current_scat:
                    path.append(current_scat)
                    current_scat = current_scat.parent
                return path[::-1], total_travel_time  # Return the path from start to goal
        
            closed_list.add(current_scat.scats_id)  # Mark current node as evaluated

            for neighbor_id in current_scat.neighbors:
                neighbor_scat = scats_dict.get(neighbor_id)

                if (neighbor_id in closed_list):
                    continue

                # Calculate g (actual cost) and h (heuristic cost) for the neighbor
                
                # travel_distance = math.sqrt(((neighbor_scat.latitude - current_scat.latitude) * 111)**2 + ((neighbor_scat.longitude - current_scat.longitude) * 111 * math.cos(math.radians(neighbor_scat.latitude)))**2)
                travel_distance = GD((neighbor_scat.latitude, neighbor_scat.longitude), (current_scat.latitude, current_scat.longitude))
                g = current_scat.gCost + self.calculateHeuristicCost(current_scat, neighbor_scat, speed) 
                h = self.calculateHeuristicCost(neighbor_scat, destination_scat, speed)  # Heuristic based on distance to goal
                
                # Travel time for the current segment
                
                total_travel_time = self.calculateTravelTime(travel_distance.km, speed)

                # If the new path to the neighbor is better (lower cost), update its g, h, f, and parent
                if g < neighbor_scat.gCost:
                    neighbor_scat.gCost = g
                    neighbor_scat.hCost = h
                    neighbor_scat.fCost = g + h
                    neighbor_scat.parent = current_scat  # Set the parent for backtracking
                    neighbor_scat.distance_travelled = current_scat.distance_travelled + travel_distance.km  # Update total distance
                    neighbor_scat.travel_time = current_scat.travel_time + total_travel_time

                    # Add the neighbor SCAT to the open list if it's not already in it
                    if neighbor_scat not in open_list:
                        heapq.heappush(open_list, neighbor_scat)

        # print(f"origin: {origin_scat}")
        # print(f"destination: {destination_scat}")

        return None








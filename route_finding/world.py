from route_finding.scats import Scats
from route_finding.path import Path
import math, heapq
from geopy.distance import geodesic as GD

class World(object):
    def __init__(self, data, origin = 970, destination = 3001, scat_to_predict=3001):
        self.data = data
        self.origin = origin
        self.destination = destination
        self.scat_to_predict = scat_to_predict
        self.scats = []

        for index, row in data.iterrows():
            scats_id = row['SCATS_Number']
            scat_name = row['Location']
            neighbors = row['SCATS Neighbours']
            latitude = row['NB_LATITUDE']
            longitude = row['NB_LONGITUDE']

            converted_neighbors = neighbors.split(" ")

            
            self.scats.append(Scats(int(scats_id), scat_name, float(latitude), float(longitude), converted_neighbors))

            

    
    
        print("-------------------------------\n")
        
        
        # paths = self.search(self.origin, self.destination, self.scats)
        
        # if paths:
        #     print(f"Origin: {paths[0].origin_scat}Destination: {paths[0].destination_scat}")
    
        #     i = 1
        #     for path in paths:
        #         print(f"----- Route {i} -----")
        #         print(path)
        #         print("\n")
        #         i += 1
        # else:
        #     print("Error: Fail to compute path.\n")
        #     print("-------------------------------")
            
        
    def calculate_heuristic_cost(self, scat, destination_scat, speed):
        hCost = (math.sqrt((destination_scat.latitude - scat.latitude)**2 + (destination_scat.longitude - scat.longitude)**2) / speed) * 3600 #converting from hour -> sec
        return hCost
    
    def calculate_travel_time(self, distance, speed):
        intersection_delay = 30
        travel_time = distance/speed * 3600 #convert to seconds
        travel_time += intersection_delay

        return travel_time

    def search_no_param(self):
        return self.search(self.origin, self.destination, self.scats)
    
    # TODO: Implement predict traffic flow at one scat id
    def predict_traffic_flow(self):
        prediction = 60
        return prediction

    def search(self, origin, destination, scats, speed=60):
        paths = []
        blocked_edges = set()

        # Find the first shortest path
        root_path = self.search_a_star_with_blocking(origin, destination, scats, blocked_edges, speed)
        if root_path:
            paths.append(root_path)

        # Now, for each new path, block one edge from the root path and find the next shortest path
        for _ in range(4):  # We want 5 paths in total, so we find 4 more
            # Block one edge at a time from the root path
            for i in range(len(root_path.path) - 1):
                # Create a temporary blocked edges set
                temp_blocked_edges = set(blocked_edges)

                # Block the edge between two nodes
                edge = (root_path.path[i].scats_id, root_path.path[i + 1].scats_id)
                reverse_edge = (root_path.path[i + 1].scats_id, root_path.path[i].scats_id)

                # Add the edge to the blocked set
                temp_blocked_edges.add(edge)
                temp_blocked_edges.add(reverse_edge)

                # Find the new path with this edge blocked
                new_path = self.search_a_star_with_blocking(origin, destination, scats, temp_blocked_edges, speed)

                if new_path and new_path not in paths:
                    paths.append(new_path)
                    # Add the newly blocked edges to the main blocked set for future iterations
                    blocked_edges.add(edge)
                    blocked_edges.add(reverse_edge)

                # Break if we found 5 paths
                if len(paths) >= 5:
                    break

        return paths


    def search_a_star_with_blocking(self, origin, destination, scats, blocked_edges, speed):
        self.reset_props(scats) # resetting the costs of each scat after each search

        open_list = []
        closed_list = set()

        scats_dict = {scat.scats_id: scat for scat in scats}

        origin_scat = scats_dict.get(origin)
        destination_scat = scats_dict.get(destination)

        if (not origin_scat): 
            print(f"Error: Origin SCATS Site - {origin} is invalid.")
            return None
        
        if (not destination_scat):
            print(f"Error: Destination SCATS Site - {destination} is invalid.")

        origin_scat.gCost = 0
        origin_scat.hCost = self.calculate_heuristic_cost(origin_scat, destination_scat, speed)
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
                total_distance = current_scat.distance_travelled
                while current_scat:
                    path.append(current_scat)
                    current_scat = current_scat.parent
                return Path(path[::-1], origin_scat, destination_scat, total_distance, total_travel_time)  # Return the path from start to goal
        
            closed_list.add(current_scat.scats_id)  # Mark current node as evaluated

            for neighbor_id in current_scat.neighbors:
                # **Correct edge blocking condition**:
                # Check if the edge (current_scat -> neighbor_id) or (neighbor_id -> current_scat) is in blocked_edges
                if (current_scat.scats_id, neighbor_id) in blocked_edges or (neighbor_id, current_scat.scats_id) in blocked_edges:
                    continue  # Skip this neighbor since it's part of a blocked edge
                         
                
                neighbor_scat = scats_dict.get(neighbor_id)
            
                if (neighbor_id in closed_list):
                    continue

                # Calculate g (actual cost) and h (heuristic cost) for the neighbor
                
                # travel_distance = math.sqrt(((neighbor_scat.latitude - current_scat.latitude) * 111)**2 + ((neighbor_scat.longitude - current_scat.longitude) * 111 * math.cos(math.radians(neighbor_scat.latitude)))**2)
                travel_distance = GD((neighbor_scat.latitude, neighbor_scat.longitude), (current_scat.latitude, current_scat.longitude))
                g = current_scat.gCost + self.calculate_heuristic_cost(current_scat, neighbor_scat, speed) 
                h = self.calculate_heuristic_cost(neighbor_scat, destination_scat, speed)  # Heuristic based on distance to goal
                
                # Travel time for the current segment
                
                total_travel_time = self.calculate_travel_time(travel_distance.km, speed)

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
    
    def reset_props(self, scats):
        for scat in scats:
            scat.gCost = float('inf')
            scat.hCost = 0
            scat.fCost = float('inf') # f = g + h
            scat.parent = None
            scat.distance_travelled = 0
            scat.travel_time = 0
        








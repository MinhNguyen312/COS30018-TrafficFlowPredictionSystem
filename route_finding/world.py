from route_finding.scats import Scats
from route_finding.path import Path
import math, heapq
from geopy.distance import geodesic as GD
from keras.api.models import load_model
import tensorflow as tf
from data.data import process_data
import numpy as np
from datetime import datetime
from keras.api.losses import MeanSquaredError

class World(object):
    def __init__(self, data, origin=970, destination=3001, date=None, model='lstm'):
        self.data = data
        self.origin = origin
        self.destination = destination
        self.scats = []
        self.date = date
        self.model = model
        self.traffic_predictions = {}  # Cache for traffic predictions

        for index, row in data.iterrows():
            scats_id = row['SCATS_Number']
            scat_name = row['Location']
            neighbors = row['SCATS Neighbours']
            latitude = row['NB_LATITUDE']
            longitude = row['NB_LONGITUDE']

            converted_neighbors = neighbors.split(" ")
            
            self.scats.append(Scats(int(scats_id), scat_name, float(latitude), float(longitude), converted_neighbors))
        
        print("-------------------------------\n")
    
    def set_date(self, new_date):
        """Update the date and clear the traffic predictions cache"""
        if self.date != new_date:
            print(f"Updating date from {self.date} to {new_date}")
            self.date = new_date
            # Clear the traffic predictions cache when date changes
            self.traffic_predictions = {}
            print("Traffic predictions cache cleared")
            
    def calculate_heuristic_cost(self, scat, destination_scat, base_speed):
        hCost = (math.sqrt((destination_scat.latitude - scat.latitude)**2 + 
                          (destination_scat.longitude - scat.longitude)**2) / base_speed) * 3600
        return hCost
    
    def calculate_travel_time(self, distance, base_speed, current_scat_id=None):
        intersection_delay = 30
        
        # Use default speed if no traffic prediction is needed
        if current_scat_id is None or self.date is None:
            travel_time = distance/base_speed * 3600  # convert to seconds
            travel_time += intersection_delay
            return travel_time
        
        # Get or calculate traffic volume prediction for this SCAT
        if current_scat_id not in self.traffic_predictions:
            traffic_prediction = self.predict_traffic_flow(self.model, current_scat_id, self.date)
            self.traffic_predictions[current_scat_id] = traffic_prediction
            print(f"New prediction for SCAT {current_scat_id} at {self.date}")
        else:
            traffic_prediction = self.traffic_predictions[current_scat_id]
            print(f"Using cached prediction for SCAT {current_scat_id}")
        
        # Select the appropriate traffic volume based on the current time
        traffic_volume = self.select_traffic_volume_for_time(traffic_prediction, self.date)
        print(f"Selected traffic volume: {traffic_volume} for time: {self.date}")
        
        # Adjust speed based on traffic volume
        congestion_factor = self.calculate_congestion_factor(traffic_volume)
        adjusted_speed = base_speed * congestion_factor
        print(f"Congestion factor: {congestion_factor}, adjusted speed: {adjusted_speed}")
        
        # Calculate travel time with adjusted speed
        travel_time = distance/adjusted_speed * 3600  # convert to seconds
        travel_time += intersection_delay
        print(f"Final travel time: {travel_time}")

        return travel_time
    
    def select_traffic_volume_for_time(self, prediction_array, date_str):
        """
        Select the appropriate traffic volume from the prediction array based on the time
        Each element in the array represents a 15-minute interval
        """
        try:
            # Parse the date string to get the time
            prediction_datetime = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate the index in the prediction array
            # Each index represents a 15-minute interval
            hour = prediction_datetime.hour
            minute = prediction_datetime.minute
            
            # Calculate the time slot (0-95 for a day, 15-minute intervals)
            time_slot = (hour * 4) + (minute // 15)
            print(f"Calculated time slot: {time_slot} for {hour}:{minute}")
            
            # If the time_slot is within the range of our predictions, use it
            if time_slot < len(prediction_array):
                return prediction_array[time_slot]
            else:
                # Default to the last predicted value if time is beyond our prediction range
                return prediction_array[-1]
                
        except Exception as e:
            print(f"Error selecting traffic volume for time {date_str}: {str(e)}")
            # Return the middle value of the prediction array or the first value if array is empty
            if len(prediction_array) > 0:
                return prediction_array[len(prediction_array) // 2]
            else:
                return 0
    
    def calculate_congestion_factor(self, traffic_volume):
        """
        Calculate a congestion factor based on traffic volume
        Returns a factor between 0.2 (heavy congestion) and 1.0 (free flow)
        """
        # These thresholds should be calibrated based on your data
        if traffic_volume <= 100:
            return 1.0  # Free flow
        elif traffic_volume <= 300:
            return 0.8  # Light congestion
        elif traffic_volume <= 600:
            return 0.6  # Moderate congestion 
        elif traffic_volume <= 900:
            return 0.4  # Heavy congestion
        else:
            return 0.2  # Severe congestion

    def search_no_param(self):
        return self.search(self.origin, self.destination, self.scats)
    
    def predict_traffic_flow(self, model_type, scat_id, date=None):
        """Predict traffic flow at a specific SCAT site for a given date"""
        if date is None:
            return np.array([0])  # Return default value if no date specified
            
        try:
            prediction_datetime = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print("Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS'.")
            return np.array([0])
            
        # Extract features from the datetime for the model
        hour_of_day = prediction_datetime.hour
        day_of_week = prediction_datetime.weekday()  # 0=Monday, 6=Sunday
        month_of_year = prediction_datetime.month
        is_weekend = 1 if day_of_week >= 5 else 0  # 1 if Saturday/Sunday, else 0

        # For example, you can create an array of these datetime-based features
        datetime_features = np.array([[hour_of_day, day_of_week, month_of_year, is_weekend]])
        print(f"Datetime features: hour={hour_of_day}, day={day_of_week}, month={month_of_year}, weekend={is_weekend}")

        lag = 12

        try:
            file1 = f'../data/train_data/{scat_id}_flow.csv'
            file2 = f'../data/test_data/{scat_id}_flow.csv'
            _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
            original_X = X_test.copy()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

            model_type = model_type.lower()
            predicted = None
            
            if model_type == 'lstm':
                lstm = load_model(f'../model/{model_type}/{scat_id}.h5', 
                                custom_objects={"mse": MeanSquaredError()})
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted = lstm.predict(X_test)
            elif model_type == 'gru':
                gru = load_model(f'../model/{model_type}/{scat_id}.h5', 
                                custom_objects={"mse": MeanSquaredError()})
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted = gru.predict(X_test)
            elif model_type == 'xgboost':
                import xgboost as xgb
                import json
                import os
                
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
                
                # Check if we have feature info saved
                feature_info_path = f'../model/{model_type}/{scat_id}_features.json'
                if os.path.exists(feature_info_path):
                    with open(feature_info_path, 'r') as f:
                        feature_info = json.load(f)
                    includes_datetime = feature_info.get('includes_datetime', False)
                else:
                    # Default to not using datetime features if info file doesn't exist
                    includes_datetime = False
                
                # Load the model
                xgb_model = xgb.Booster()
                xgb_model.load_model(f'../model/{model_type}/{scat_id}.json')
                
                # If model was trained with datetime features, add them now
                if includes_datetime:
                    # Get the last sample to use with datetime features
                    last_sample_index = X_test.shape[0] - 1
                    sample_to_predict = X_test[last_sample_index:last_sample_index+1, :]
                    
                    # Concatenate with datetime features
                    X_test_with_datetime = np.concatenate([sample_to_predict, datetime_features], axis=1)
                    dtest = xgb.DMatrix(X_test_with_datetime)
                else:
                    # Use original features only
                    dtest = xgb.DMatrix(X_test)
                    
                predicted = xgb_model.predict(dtest)


            elif model_type == 'saes':
                import os
                if not os.path.exists(f'../model/{model_type}/{scat_id}.h5'):
                    print(f"SAES model not found at {model_type}/{scat_id}.h5")

                # Implement SAES model loading and prediction
                saes = load_model(f'../model/{model_type}/{scat_id}.h5', 
                                custom_objects={"mse": MeanSquaredError()})
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
                # Use only the original 12 features from the lagged data
                predicted = saes.predict(X_test)

                def preprocess_features(original_features, datetime_features):    
                        # Get the last sample of original features
                        last_sample = original_features[-1:, :]  # Shape (1, 12)
                        
                        
                        hour = datetime_features[0, 0] / 24.0  # Normalize hour to [0, 1]
                        day = datetime_features[0, 1] / 6.0    # Normalize day to [0, 1]
                        month = datetime_features[0, 2] / 12.0  # Normalize month to [0, 1]
                        weekend = datetime_features[0, 3]      # Already binary
                        
                        # Modify original features based on datetime
                        modified_features = last_sample.copy()
                        for i in range(min(4, last_sample.shape[1])):
                            # Simple weighted combination
                            datetime_weight = 0.2  # How much influence datetime has
                            if i == 0:
                                modified_features[0, i] *= (1 + datetime_weight * hour)
                            elif i == 1:
                                modified_features[0, i] *= (1 + datetime_weight * day)
                            elif i == 2:
                                modified_features[0, i] *= (1 + datetime_weight * month)
                            elif i == 3:
                                modified_features[0, i] *= (1 + datetime_weight * weekend)
                        
                        return modified_features 
                
            else:
                print(f"Error: Invalid model name: {model_type}.")
                return np.array([0])
                
            # Inverse transform the predicted values to bring them back to original scale
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

            # Get the last sample to use with datetime features
            last_sample_index = original_X.shape[0] - 1
            sample_to_predict = original_X[last_sample_index:last_sample_index+1, :]

            # Concatenate with datetime features
            X_test_with_datetime = np.concatenate([sample_to_predict, datetime_features], axis=1)
            
            # Make predictions with datetime features for different model types
            if model_type == 'lstm':
                # Reshape for LSTM model
                X_test_with_datetime_reshaped = np.reshape(X_test_with_datetime, 
                                                (X_test_with_datetime.shape[0], 
                                                X_test_with_datetime.shape[1], 
                                                1))
                predicted_with_datetime = lstm.predict(X_test_with_datetime_reshaped)
            elif model_type == 'gru':
                # Reshape for GRU model
                X_test_with_datetime_reshaped = np.reshape(X_test_with_datetime, 
                                                (X_test_with_datetime.shape[0], 
                                                X_test_with_datetime.shape[1], 
                                                1))
                predicted_with_datetime = gru.predict(X_test_with_datetime_reshaped)
            elif model_type == 'xgboost':
                # Prepare for XGBoost
                dtest_with_datetime = xgb.DMatrix(X_test_with_datetime)
                predicted_with_datetime = xgb_model.predict(dtest_with_datetime)
            elif model_type == 'saes':
                # Prepare for SAES
                modified_features = preprocess_features(sample_to_predict, datetime_features)
                predicted_with_datetime = saes.predict(modified_features)
            else:
                return predicted  # Return the prediction without datetime adjustments

            # Inverse transform the predicted values
            predicted_with_datetime = scaler.inverse_transform(predicted_with_datetime.reshape(-1, 1)).reshape(1, -1)[0]
            
            # Print some debug info
            print(f"Prediction for SCAT {scat_id} at {date}: {predicted_with_datetime}")
            
            # Return the entire prediction array
            return predicted_with_datetime
            
        except Exception as e:
            print(f"Error predicting traffic for SCAT {scat_id}: {str(e)}")
            return np.array([0])  # Return default value on error

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

    def search_a_star_with_blocking(self, origin, destination, scats, blocked_edges, base_speed):
        self.reset_props(scats)  # resetting the costs of each scat after each search

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
            return None

        origin_scat.gCost = 0
        origin_scat.hCost = self.calculate_heuristic_cost(origin_scat, destination_scat, base_speed)
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
                return Path(path[::-1], origin_scat, destination_scat, total_distance, total_travel_time)
        
            closed_list.add(current_scat.scats_id)  # Mark current node as evaluated

            for neighbor_id in current_scat.neighbors:
                # Skip if neighbor id is not a valid integer
                try:
                    neighbor_id = int(neighbor_id)
                except ValueError:
                    continue
                    
                # Check if the edge is blocked
                if (current_scat.scats_id, neighbor_id) in blocked_edges or (neighbor_id, current_scat.scats_id) in blocked_edges:
                    continue  # Skip this neighbor since it's part of a blocked edge
                         
                neighbor_scat = scats_dict.get(neighbor_id)
                if not neighbor_scat or neighbor_id in closed_list:
                    continue

                # Calculate distance between current SCAT and neighbor
                travel_distance = GD((neighbor_scat.latitude, neighbor_scat.longitude), 
                                    (current_scat.latitude, current_scat.longitude))
                
                # Calculate travel time considering traffic predictions
                travel_time = self.calculate_travel_time(travel_distance.km, base_speed, neighbor_scat.scats_id)
                
                # Calculate costs
                g = current_scat.gCost + travel_time  # Use travel time as cost
                h = self.calculate_heuristic_cost(neighbor_scat, destination_scat, base_speed)

                # If the new path to the neighbor is better (lower cost), update its g, h, f, and parent
                if g < neighbor_scat.gCost:
                    neighbor_scat.gCost = g
                    neighbor_scat.hCost = h
                    neighbor_scat.fCost = g + h
                    neighbor_scat.parent = current_scat  # Set the parent for backtracking
                    neighbor_scat.distance_travelled = current_scat.distance_travelled + travel_distance.km
                    neighbor_scat.travel_time = current_scat.travel_time + travel_time

                    # Add the neighbor SCAT to the open list if it's not already in it
                    if neighbor_scat not in open_list:
                        heapq.heappush(open_list, neighbor_scat)

        return None
    
    def reset_props(self, scats):
        for scat in scats:
            scat.gCost = float('inf')
            scat.hCost = 0
            scat.fCost = float('inf')  # f = g + h
            scat.parent = None
            scat.distance_travelled = 0
            scat.travel_time = 0
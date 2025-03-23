class Path(object):
    def __init__(self, path, origin_scat, destination_scat, total_distance, time):
        
        self.path = path
        self.origin_scat = origin_scat
        self.destination_scat = destination_scat
        self.total_distance = total_distance
        self.time = time

    def __repr__(self):
        output = ""
        for scat in self.path:
            output += f"{scat.scats_id} - {scat.scat_name}\n"

        return f"{output}Length: {len(self.path)}\nTotal distance travelled: {self.total_distance} km\nTotal travel time: {self.time / 60} mins"
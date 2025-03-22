# Path finding algorithm for TFPS

Current version of the path finding algorithm successfully find the most optimal path, with distance traveled in that path, and the estimate time it takes to travel through the path, ignoring the traffic flow and other factors at the moment.

## Installation
1. Clone the repository.
```bash
git clone https://github.com/MinhNguyen312/COS30018-TrafficFlowPredictionSystem.git
```

2. Navigate to the cloned repository.
```bash
cd repository-path
```

3. Switch to the ```route-finding``` branch.
```bash
# If the branch already exists locally, use:
git checkout route-finding

# If the branch does not exist locally, use:
git checkout -b route-finding origin/route-finding
```
4. Navigate to the ```route_finding``` directory:
```bash
cd route_finding
```

5. Install the required libraries.
```bash
pip install -r requirements.txt
```

## Usage
Run the program with the following command: 
```bash 
python main.py
```


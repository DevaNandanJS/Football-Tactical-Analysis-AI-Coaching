from collections import deque
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional

class FormationDetector:
    """
    Detects the playing formation of a team based on player positions over time.
    """

    def __init__(self, buffer_size: int = 600):
        """
        Initializes the FormationDetector.

        Args:
            buffer_size (int): Max number of frames to keep in history. 
                               600 frames @ 30fps is ~20 seconds.
        """
        # History buffers: {team_id: deque([(x,y), ...])}
        self.history: Dict[int, deque] = {
            1: deque(maxlen=buffer_size * 10), # storing individual points, allowing for ~10 players * 600 frames
            2: deque(maxlen=buffer_size * 10)
        }
        
        # Define standard templates (Normalized 0-1, Bottom-to-Top: GK/Def -> Attack)
        # Note: We exclude GK in the templates as we cluster for 10 outfield players.
        # Format: (x, y) where x is width (0-1) and y is depth (0-1, 0=Defense, 1=Attack)
        self.templates = {
            "4-4-2": [
                (0.1, 0.1), (0.35, 0.1), (0.65, 0.1), (0.9, 0.1), # 4 Defenders
                (0.1, 0.5), (0.35, 0.5), (0.65, 0.5), (0.9, 0.5), # 4 Midfielders
                (0.35, 0.9), (0.65, 0.9)                          # 2 Attackers
            ],
            "4-3-3": [
                (0.1, 0.1), (0.35, 0.1), (0.65, 0.1), (0.9, 0.1), # 4 Defenders
                (0.3, 0.5), (0.5, 0.5), (0.7, 0.5),               # 3 Midfielders
                (0.2, 0.9), (0.5, 0.9), (0.8, 0.9)                # 3 Attackers
            ],
            "4-2-3-1": [
                (0.1, 0.1), (0.35, 0.1), (0.65, 0.1), (0.9, 0.1), # 4 Defenders
                (0.35, 0.4), (0.65, 0.4),                         # 2 DM
                (0.2, 0.7), (0.5, 0.7), (0.8, 0.7),               # 3 AM
                (0.5, 0.95)                                       # 1 Striker
            ],
            "3-5-2": [
                (0.2, 0.1), (0.5, 0.1), (0.8, 0.1),               # 3 Defenders
                (0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5), # 5 Midfielders
                (0.35, 0.9), (0.65, 0.9)                          # 2 Attackers
            ],
            "3-4-3": [
                (0.2, 0.1), (0.5, 0.1), (0.8, 0.1),               # 3 Defenders
                (0.15, 0.5), (0.35, 0.5), (0.65, 0.5), (0.85, 0.5), # 4 Midfielders
                (0.2, 0.9), (0.5, 0.9), (0.8, 0.9)                # 3 Attackers
            ],
             "5-3-2": [
                (0.1, 0.1), (0.3, 0.1), (0.5, 0.1), (0.7, 0.1), (0.9, 0.1), # 5 Defenders
                (0.3, 0.5), (0.5, 0.5), (0.7, 0.5),               # 3 Midfielders
                (0.35, 0.9), (0.65, 0.9)                          # 2 Attackers
            ]
        }
        
        # Convert templates to numpy arrays for easier math
        self.np_templates = {k: np.array(v) for k, v in self.templates.items()}

    def update(self, player_detections: List[Dict], frame_index: int):
        """
        Updates the history with new player positions.

        Args:
            player_detections (list): List of dicts, e.g., 
                                      [{'team_id': 1, 'position': (x, y)}, ...]
                                      Ensure only outfield players are passed here.
            frame_index (int): Current frame number.
        """
        for player in player_detections:
            team_id = player.get('team_id')
            pos = player.get('position')
            
            if team_id in self.history and pos is not None:
                self.history[team_id].append(pos)

    def compute_formation(self, team_id: int) -> str:
        """
        Calculates the most likely formation for the given team ID.

        Args:
            team_id (int): The team to analyze (1 or 2).

        Returns:
            str: Name of the formation (e.g., '4-4-2') or 'Analyzing...'
        """
        if team_id not in self.history:
            return "Unknown Team"

        data_points = list(self.history[team_id])
        
        # Heuristic: Need a decent amount of data to cluster effectively
        # 10 players * 30 frames = 300 points minimum seems reasonable for stability
        if len(data_points) < 300:
            return "Analyzing..."

        points = np.array(data_points)

        try:
            # 1. Cluster to find 10 outfield centroids
            kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
            kmeans.fit(points)
            centroids = kmeans.cluster_centers_

            # 2. Normalize centroids to 0-1 range
            min_vals = np.min(centroids, axis=0)
            max_vals = np.max(centroids, axis=0)
            
            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            
            norm_centroids = (centroids - min_vals) / range_vals

            # 3. Match against templates
            best_formation = "Unknown"
            min_dist = float('inf')

            # We need to handle orientation (Attack Up vs Attack Down)
            # Since we normalized to 0-1, "Defense" could be at Y=0 or Y=1.
            # We will test both the template and the flipped template.
            
            for name, template in self.np_templates.items():
                
                # Test 1: Standard Orientation
                dist_standard = self._calculate_match_distance(norm_centroids, template)
                
                # Test 2: Flipped Orientation (Rotate 180 degrees or flip Y)
                # Since we normalized to 0-1 box, flipping Y is (x, 1-y)
                flipped_template = template.copy()
                flipped_template[:, 1] = 1.0 - flipped_template[:, 1]
                dist_flipped = self._calculate_match_distance(norm_centroids, flipped_template)

                # Take the better fit of the two orientations
                current_min = min(dist_standard, dist_flipped)

                if current_min < min_dist:
                    min_dist = current_min
                    best_formation = name

            return best_formation

        except Exception as e:
            # Fallback if clustering fails (e.g., singular matrix, not enough distinct points)
            return "Analyzing..."

    def _calculate_match_distance(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """
        Calculates the minimum total distance between two sets of points 
        using the Hungarian algorithm (Linear Sum Assignment).
        """
        # Calculate distance matrix (cost matrix)
        # rows: points1 indices, cols: points2 indices
        # We need pairwise distances.
        
        # Expand dims for broadcasting
        # p1: (10, 1, 2), p2: (1, 10, 2) -> dists: (10, 10)
        dists = np.linalg.norm(points1[:, None] - points2[None, :], axis=2)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(dists)
        
        # Sum of distances for the optimal assignment
        total_dist = dists[row_ind, col_ind].sum()
        
        return total_dist

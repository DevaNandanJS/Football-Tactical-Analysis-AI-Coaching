import cv2
import numpy as np

class MovementHeatmapGenerator:
    """
    Generates movement heatmaps for football players based on their 2D field positions.
    Accumulates player positions over time to create a density map (heatmap) for each team
    and a combined view.
    """

    def __init__(self, map_width, map_height, team_names, sigma=20, decay_factor=1.0):
        """
        Initialize the MovementHeatmapGenerator.

        Args:
            map_width (int): Width of the 2D field/map.
            map_height (int): Height of the 2D field/map.
            team_names (list): List of two strings representing the names of the clubs 
                               (e.g., ['TeamA', 'TeamB']).
            sigma (int, optional): Standard deviation for the Gaussian kernel blob. Defaults to 20.
            decay_factor (float, optional): Factor to multiply the map by at each update. 
                                            1.0 means no decay (total history). Defaults to 1.0.
        """
        self.map_width = map_width
        self.map_height = map_height
        self.team_names = team_names
        self.decay_factor = decay_factor

        # Initialize accumulation maps for Team 1, Team 2, and Combined
        # Using float32 for precise accumulation
        self.map_team1 = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_team2 = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_combined = np.zeros((map_height, map_width), dtype=np.float32)

        # Pre-compute the Gaussian kernel for optimization
        # Kernel size is determined by 6*sigma to capture the significant part of the distribution
        # Ensure ksize is odd
        self.ksize = int(6 * sigma) | 1 
        self.half_ksize = self.ksize // 2
        
        # Create 1D Gaussian and then 2D via outer product
        # Scaling up the kernel so single detections are visible before normalization if needed,
        # but primarily rely on normalization at output. 
        # Ideally, we want the peak to be meaningful.
        gaussian_1d = cv2.getGaussianKernel(self.ksize, sigma)
        self.kernel = gaussian_1d @ gaussian_1d.T
        
        # Normalize kernel so peak is 1 (optional, depends on desired intensity accumulation)
        # Here we normalize so the max value is 1 to treat it as "presence intensity"
        self.kernel = self.kernel / self.kernel.max()


    def update(self, player_positions, team_assignments):
        """
        Update the heatmaps with new player positions.

        Args:
            player_positions (dict): Dictionary mapping tracker_id to (x, y) coordinates on the 2D map.
            team_assignments (dict): Dictionary mapping tracker_id to team_name.
        """
        # Apply decay if specified (for fading trails effect, though default is 1.0)
        if self.decay_factor < 1.0:
            self.map_team1 *= self.decay_factor
            self.map_team2 *= self.decay_factor
            self.map_combined *= self.decay_factor

        for track_id, position in player_positions.items():
            # Skip if we don't know the team for this player
            if track_id not in team_assignments:
                continue

            team_name = team_assignments[track_id]
            x, y = int(position[0]), int(position[1])

            # Determine bounds for the kernel on the map
            # Coordinate of the top-left corner of the kernel on the map
            x1 = x - self.half_ksize
            y1 = y - self.half_ksize
            x2 = x1 + self.ksize
            y2 = y1 + self.ksize

            # Determine bounds for the kernel itself (handling edge cases)
            k_x1 = 0
            k_y1 = 0
            k_x2 = self.ksize
            k_y2 = self.ksize

            # Clip to map boundaries
            if x1 < 0:
                k_x1 = -x1
                x1 = 0
            if y1 < 0:
                k_y1 = -y1
                y1 = 0
            if x2 > self.map_width:
                k_x2 -= (x2 - self.map_width)
                x2 = self.map_width
            if y2 > self.map_height:
                k_y2 -= (y2 - self.map_height)
                y2 = self.map_height

            # Check if valid region exists
            if x1 >= x2 or y1 >= y2:
                continue

            # Extract the valid slice of the kernel
            kernel_slice = self.kernel[k_y1:k_y2, k_x1:k_x2]

            # Add to combined map
            self.map_combined[y1:y2, x1:x2] += kernel_slice

            # Add to specific team map
            if team_name == self.team_names[0]:
                self.map_team1[y1:y2, x1:x2] += kernel_slice
            elif team_name == self.team_names[1]:
                self.map_team2[y1:y2, x1:x2] += kernel_slice


    def generate_heatmaps(self):
        """
        Generate the visualization images for the current state of the heatmaps.

        Returns:
            list: A list containing 3 BGR images [heatmap_team1, heatmap_team2, heatmap_combined].
                  The images are color-mapped (Jet).
        """
        def process_map(heatmap_array):
            # Avoid division by zero if map is empty
            if np.max(heatmap_array) == 0:
                return np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
            
            # Normalize to 0-255
            norm_map = cv2.normalize(heatmap_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            norm_map = norm_map.astype(np.uint8)
            
            # Apply color map
            colored_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            return colored_map

        return [
            process_map(self.map_team1),
            process_map(self.map_team2),
            process_map(self.map_combined)
        ]

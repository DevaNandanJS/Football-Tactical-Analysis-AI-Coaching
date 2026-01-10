import cv2
import numpy as np

class MovementHeatmapGenerator:
    """
    Generates movement heatmaps for football players based on their 2D field positions.
    Accumulates player positions over time to create a density map (heatmap) for each team
    and a combined view, overlaid on the football field image.
    Also visualizes current player positions as dots.
    """

    def __init__(self, map_width, map_height, team_names, field_image, sigma=20, decay_factor=1.0):
        """
        Initialize the MovementHeatmapGenerator.

        Args:
            map_width (int): Width of the 2D field/map.
            map_height (int): Height of the 2D field/map.
            team_names (list): List of two strings representing the names of the clubs 
                               (e.g., ['TeamA', 'TeamB']).
            field_image (np.ndarray): The background image of the football field.
            sigma (int, optional): Standard deviation for the Gaussian kernel blob. Defaults to 20.
            decay_factor (float, optional): Factor to multiply the map by at each update. 
                                            1.0 means no decay (total history). Defaults to 1.0.
        """
        self.map_width = map_width
        self.map_height = map_height
        self.team_names = team_names
        self.decay_factor = decay_factor
        self.field_image = field_image

        # Initialize accumulation maps for Team 1, Team 2, and Combined
        # Using float32 for precise accumulation
        self.map_team1 = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_team2 = np.zeros((map_height, map_width), dtype=np.float32)
        self.map_combined = np.zeros((map_height, map_width), dtype=np.float32)

        # Store current frame data for dot visualization
        self.current_positions = {}
        self.current_team_assignments = {}

        # Colors for dots (BGR)
        # Team 1: Yellow (0, 255, 255)
        # Team 2: Blue (255, 0, 0) - Using distinct Red/Blue might be better contrast, 
        # but let's stick to distinct colors. 
        self.color_team1 = (0, 255, 255) # Yellow
        self.color_team2 = (255, 0, 0)   # Blue
        self.color_outline = (0, 0, 0)   # Black outline

        # Pre-compute the Gaussian kernel for optimization
        self.ksize = int(6 * sigma) | 1 
        self.half_ksize = self.ksize // 2
        
        gaussian_1d = cv2.getGaussianKernel(self.ksize, sigma)
        self.kernel = gaussian_1d @ gaussian_1d.T
        self.kernel = self.kernel / self.kernel.max()


    def update(self, player_positions, team_assignments):
        """
        Update the heatmaps with new player positions and store positions for visualization.

        Args:
            player_positions (dict): Dictionary mapping tracker_id to (x, y) coordinates on the 2D map.
            team_assignments (dict): Dictionary mapping tracker_id to team_name.
        """
        # Store for rendering dots later
        self.current_positions = player_positions.copy()
        self.current_team_assignments = team_assignments.copy()

        # Apply decay if specified
        if self.decay_factor < 1.0:
            self.map_team1 *= self.decay_factor
            self.map_team2 *= self.decay_factor
            self.map_combined *= self.decay_factor

        for track_id, position in player_positions.items():
            if track_id not in team_assignments:
                continue

            team_name = team_assignments[track_id]
            x, y = int(position[0]), int(position[1])

            # --- Heatmap Accumulation Logic ---
            x1 = x - self.half_ksize
            y1 = y - self.half_ksize
            x2 = x1 + self.ksize
            y2 = y1 + self.ksize

            k_x1, k_y1 = 0, 0
            k_x2, k_y2 = self.ksize, self.ksize

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

            if x1 >= x2 or y1 >= y2:
                continue

            kernel_slice = self.kernel[k_y1:k_y2, k_x1:k_x2]

            # Add to maps
            self.map_combined[y1:y2, x1:x2] += kernel_slice
            if team_name == self.team_names[0]:
                self.map_team1[y1:y2, x1:x2] += kernel_slice
            elif team_name == self.team_names[1]:
                self.map_team2[y1:y2, x1:x2] += kernel_slice


    def generate_heatmaps(self):
        """
        Generate the visualization images: Field + Heatmap + Player Dots.

        Returns:
            list: [heatmap_team1, heatmap_team2, heatmap_combined]
        """
        # 1. Generate Base Maps (Field + Heatmap Overlay)
        img_team1 = self._render_base_map(self.map_team1)
        img_team2 = self._render_base_map(self.map_team2)
        img_combined = self._render_base_map(self.map_combined)

        # 2. Draw Player Dots
        for track_id, position in self.current_positions.items():
            if track_id not in self.current_team_assignments:
                continue
            
            team_name = self.current_team_assignments[track_id]
            center = (int(position[0]), int(position[1]))
            
            # Determine color
            if team_name == self.team_names[0]:
                color = self.color_team1
                # Draw on Team 1 map and Combined map
                self._draw_dot(img_team1, center, color)
                self._draw_dot(img_combined, center, color)
            elif team_name == self.team_names[1]:
                color = self.color_team2
                # Draw on Team 2 map and Combined map
                self._draw_dot(img_team2, center, color)
                self._draw_dot(img_combined, center, color)

        return [img_team1, img_team2, img_combined]

    def _render_base_map(self, heatmap_array):
        """Helper to blend heatmap density with the field image."""
        if np.max(heatmap_array) > 0:
            norm_map = cv2.normalize(heatmap_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            norm_map = norm_map.astype(np.uint8)
        else:
            norm_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        colored_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        
        # Calculate alpha for blending
        alpha = norm_map.astype(float) / 255.0
        alpha_3c = cv2.merge([alpha, alpha, alpha])
        
        # Ensure field is correct size
        field = self.field_image.copy()
        if field.shape[:2] != (self.map_height, self.map_width):
            field = cv2.resize(field, (self.map_width, self.map_height))
            
        # Blend: Field is base, Heatmap is added based on intensity
        weighted_map = (colored_heatmap * alpha_3c + field * (1.0 - alpha_3c)).astype(np.uint8)
        return weighted_map

    def _draw_dot(self, image, center, color):
        """Helper to draw a player dot with outline."""
        cv2.circle(image, center, 8, color, -1)  # Filled circle
        cv2.circle(image, center, 8, self.color_outline, 2) # Outline

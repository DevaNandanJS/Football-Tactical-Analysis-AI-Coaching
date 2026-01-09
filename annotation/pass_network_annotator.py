import cv2
import numpy as np
from typing import List, Tuple
from analysis.pass_event_detector import Event

class PassNetworkAnnotator:
    """
    Visualizes a football pass network on a 2D minimap.
    """

    def __init__(self, team1_color: Tuple[int, int, int], team2_color: Tuple[int, int, int], field_img_path: str):
        """
        Initializes the PassNetworkAnnotator.

        Args:
            team1_color (Tuple[int, int, int]): RGB color for team 1 (corresponding to team_id=0).
            team2_color (Tuple[int, int, int]): RGB color for team 2 (corresponding to team_id=1).
            field_img_path (str): Path to the field image file.
        """
        self.team1_color = team1_color
        self.team2_color = team2_color
        
        # Load the field image as a persistent background canvas
        # We read it once and keep it.
        self.field_image = cv2.imread(field_img_path)
        if self.field_image is None:
            raise FileNotFoundError(f"Could not load field image from {field_img_path}")
            
        self.events: List[Event] = []

    def draw(self, frame: np.ndarray, current_events: List[Event]) -> np.ndarray:
        """
        Updates the internal history with current_events and overlays the pass network mini-map 
        onto the top-right corner of the main frame.

        Args:
            frame (np.ndarray): The main video frame.
            current_events (List[Event]): List of new events detected in the current frame.

        Returns:
            np.ndarray: The frame with the mini-map overlay.
        """
        # Update internal history
        self.events.extend(current_events)

        # Create a copy of the persistent field canvas for this frame's drawing
        canvas = self.field_image.copy()

        # Iterate through all historical events
        for event in self.events:
            if event.start_xy is None or event.end_xy is None:
                continue
            
            # Convert float coordinates to integers for OpenCV
            start_pt = (int(event.start_xy[0]), int(event.start_xy[1]))
            end_pt = (int(event.end_xy[0]), int(event.end_xy[1]))

            # Determine color based on team_id
            # Assumption: team_id 0 -> team1_color, team_id 1 -> team2_color
            if event.team_id == 0:
                color = self.team1_color
            else:
                color = self.team2_color

            if event.type == "PASS":
                # Draw a line connecting Start->End in the team's color
                cv2.line(canvas, start_pt, end_pt, color, 2)
                # Draw a filled circle at the "End" location
                cv2.circle(canvas, end_pt, 5, color, -1)
                
            elif event.type == "INTERCEPTION":
                # Draw a line connecting Start->End in a MIXED color
                mixed_color = (
                    int((self.team1_color[0] + self.team2_color[0]) / 2),
                    int((self.team1_color[1] + self.team2_color[1]) / 2),
                    int((self.team1_color[2] + self.team2_color[2]) / 2)
                )
                cv2.line(canvas, start_pt, end_pt, mixed_color, 2)
                # Draw a circle at the "End" location
                cv2.circle(canvas, end_pt, 5, mixed_color, -1)

        # Overlay this mini-map onto the top-right corner of the main video frame
        return self._overlay_minimap(frame, canvas)

    def _overlay_minimap(self, frame: np.ndarray, minimap: np.ndarray) -> np.ndarray:
        """
        Resizes and overlays the minimap onto the frame.
        """
        # Target width for the mini-map (e.g., 350px)
        target_width = 350
        scale = target_width / minimap.shape[1]
        target_height = int(minimap.shape[0] * scale)

        # Resize the filled canvas
        minimap_resized = cv2.resize(minimap, (target_width, target_height))

        # Calculate position (Top-Right with margin)
        margin = 20
        h_frame, w_frame, _ = frame.shape
        
        y1 = margin
        y2 = margin + target_height
        x1 = w_frame - margin - target_width
        x2 = w_frame - margin

        # Ensure it fits within the frame
        if y2 <= h_frame and x1 >= 0:
            # Create a region of interest (ROI)
            roi = frame[y1:y2, x1:x2]
            
            # Optional: Add a border or transparency?
            # Prompt just says "Overlay". Let's do a simple overwrite for clarity 
            # or a slight transparency to look "techy". 
            # Let's do opaque to ensure the lines are visible.
            frame[y1:y2, x1:x2] = minimap_resized
            
            # Add a white border for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        return frame

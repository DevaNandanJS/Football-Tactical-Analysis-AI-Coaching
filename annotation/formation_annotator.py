import cv2
import numpy as np
from typing import Tuple

class FormationAnnotator:
    """
    Annotator responsible for displaying team formations on a video frame.
    """

    def __init__(self, team1_name: str, team2_name: str, 
                 team1_color: Tuple[int, int, int], team2_color: Tuple[int, int, int]):
        """
        Initializes the FormationAnnotator.

        Args:
            team1_name (str): Name of the first team.
            team2_name (str): Name of the second team.
            team1_color (Tuple[int, int, int]): BGR color for Team 1.
            team2_color (Tuple[int, int, int]): BGR color for Team 2.
        """
        self.team1_name = team1_name
        self.team2_name = team2_name
        self.team1_color = team1_color
        self.team2_color = team2_color

    def draw(self, frame: np.ndarray, formations: dict) -> np.ndarray:
        """
        Draws the formation panel on the right side of the frame.

        Args:
            frame (np.ndarray): The current video frame.
            formations (dict): A dictionary mapping team IDs (1, 2) to formation strings.
                               Example: {1: "4-4-2", 2: "Analyzing..."}

        Returns:
            np.ndarray: The annotated frame.
        """
        # Panel Configuration
        panel_w = 300
        panel_h = 100
        margin_right = 20
        margin_top = 20
        
        h, w, _ = frame.shape
        
        # Top-Left corner of the panel
        x1 = w - panel_w - margin_right
        # Center vertically
        y1 = (h - panel_h) // 2
        x2 = x1 + panel_w
        y2 = y1 + panel_h
        
        # Create Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black background
        
        # Apply Transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Header Text
        cv2.putText(frame, "Formations", (x1 + 10, y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Team 1 Info
        t1_formation = formations.get(1, "Unknown")
        cv2.putText(frame, f"{self.team1_name}: {t1_formation}", (x1 + 10, y1 + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.team1_color, 2)

        # Team 2 Info
        t2_formation = formations.get(2, "Unknown")
        cv2.putText(frame, f"{self.team2_name}: {t2_formation}", (x1 + 10, y1 + 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.team2_color, 2)

        return frame

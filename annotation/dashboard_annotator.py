import cv2
import numpy as np

class DashboardAnnotator:
    def __init__(self, stats_manager, team1_color=(0, 255, 255), team2_color=(255, 0, 0)):
        """
        Initializes the dashboard annotator.
        
        Args:
            stats_manager: Instance of TeamStatsManager.
            team1_color: BGR color tuple for Team 1 (default: Yellow).
            team2_color: BGR color tuple for Team 2 (default: Blue).
        """
        self.stats_manager = stats_manager
        self.team1_color = team1_color
        self.team2_color = team2_color
        self.opacity = 0.6
        self.text_color = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Layout config
        self.box_width = 220
        self.box_height = 140
        self.margin_x = 30
        self.margin_y = 30
        self.line_height = 25

    def draw_dashboard(self, frame):
        """
        Overlays the stats dashboard on the frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- Team 1 Panel (Bottom Left) ---
        t1_x1 = self.margin_x
        t1_y1 = h - self.box_height - self.margin_y
        t1_x2 = t1_x1 + self.box_width
        t1_y2 = h - self.margin_y
        
        cv2.rectangle(overlay, (t1_x1, t1_y1), (t1_x2, t1_y2), self.team1_color, -1)

        # --- Team 2 Panel (Bottom Right) ---
        t2_x1 = w - self.box_width - self.margin_x
        t2_y1 = h - self.box_height - self.margin_y
        t2_x2 = w - self.margin_x
        t2_y2 = h - self.margin_y

        cv2.rectangle(overlay, (t2_x1, t2_y1), (t2_x2, t2_y2), self.team2_color, -1)

        # --- Apply Transparency ---
        cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)

        # --- Draw Text ---
        stats1 = self.stats_manager.get_stats(1)
        stats2 = self.stats_manager.get_stats(2)

        self._draw_stats_text(frame, stats1, t1_x1, t1_y1, "Team 1")
        self._draw_stats_text(frame, stats2, t2_x1, t2_y1, "Team 2")

        return frame

    def _draw_stats_text(self, frame, stats, x, y, title):
        if not stats:
            return

        start_x = x + 10
        start_y = y + 25
        
        # Title
        cv2.putText(frame, title, (start_x, start_y), self.font, 0.7, self.text_color, 2)
        
        # Metrics
        # Conversion: 100 pixels = 1 meter
        distance_meters = stats['total_distance_pixels'] / 100.0
        if distance_meters >= 1000:
            dist_str = f"{distance_meters/1000:.2f}km"
        else:
            dist_str = f"{distance_meters:.1f}m"

        lines = [
            f"Poss: {stats['possession_percentage']}%",
            f"Passes: {stats['total_passes']}",
            f"Acc: {stats['pass_completion_rate']}%",
            f"Dist: {dist_str}"
        ]

        for i, line in enumerate(lines):
            # i+1 because title is at 0
            y_pos = start_y + ((i + 1) * self.line_height) + 5
            cv2.putText(frame, line, (start_x, y_pos), self.font, 0.5, self.text_color, 1)

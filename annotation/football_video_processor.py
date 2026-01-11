from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator
from .frame_number_annotator import FrameNumberAnnotator
from .pass_network_annotator import PassNetworkAnnotator
from .dashboard_annotator import DashboardAnnotator
from .formation_annotator import FormationAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner
from ball_to_player_assignment import BallToPlayerAssigner
from analysis.pass_event_detector import PassEventDetector
from analysis.movement_heatmap_generator import MovementHeatmapGenerator
from analysis.team_stats_manager import TeamStatsManager
from analysis.formation_detector import FormationDetector
from utils import rgb_bgr_converter

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    """
    A video processor for football footage that tracks objects and keypoints,
    estimates speed, assigns the ball to player, calculates the ball possession 
    and adds various annotations.
    """

    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker, 
                 club_assigner: ClubAssigner, ball_to_player_assigner: BallToPlayerAssigner, 
                 top_down_keypoints: np.ndarray, field_img_path: str, 
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:
        """
        Initializes the video processor with necessary components for tracking, annotations, and saving tracks.

        Args:
            obj_tracker (ObjectTracker): The object tracker for tracking players and balls.
            kp_tracker (KeypointsTracker): The keypoints tracker for detecting and tracking keypoints.
            club_assigner (ClubAssigner): Assigner to determine clubs for the tracked players.
            ball_to_player_assigner (BallToPlayerAssigner): Assigns the ball to a specific player based on tracking.
            top_down_keypoints (np.ndarray): Keypoints to map objects to top-down positions.
            field_img_path (str): Path to the image of the football field used for projection.
            save_tracks_dir (Optional[str]): Directory to save tracking information. If None, no tracks will be saved.
            draw_frame_num (bool): Whether or not to draw current frame number on the output video.
        """

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator() 

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)
        
        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        # Initialize the speed estimator with the field image's dimensions
        self.speed_estimator = SpeedEstimator(field_image.shape[1], field_image.shape[0])
        
        self.frame_num = 0

        self.field_image = field_image

        # Initialize Pass Network components
        self.pass_detector = PassEventDetector(possession_threshold=1)
        
        # Determine team colors (BGR) for annotator
        team1_rgb = self.club_assigner.club1.player_jersey_color
        team2_rgb = self.club_assigner.club2.player_jersey_color
        
        # Convert to BGR for OpenCV
        team1_bgr = rgb_bgr_converter(team1_rgb)
        team2_bgr = rgb_bgr_converter(team2_rgb)
        
        self.pass_annotator = PassNetworkAnnotator(team1_bgr, team2_bgr, field_img_path)
        
        # Initialize Formation Annotator
        self.formation_annotator = FormationAnnotator(
            team1_name=self.club_assigner.club1.name,
            team2_name=self.club_assigner.club2.name,
            team1_color=team1_bgr,
            team2_color=team2_bgr
        )

        # Initialize Heatmap Generator
        self.heatmap_generator = MovementHeatmapGenerator(
            map_width=self.field_image.shape[1],
            map_height=self.field_image.shape[0],
            team_names=[self.club_assigner.club1.name, self.club_assigner.club2.name],
            field_image=self.field_image
        )

        # Initialize Stats Manager and Dashboard
        self.stats_manager = TeamStatsManager()
        self.dashboard_annotator = DashboardAnnotator(self.stats_manager, team1_color=team1_bgr, team2_color=team2_bgr)
        
        # Initialize Formation Detector
        self.formation_detector = FormationDetector()
        self.team_formations = {1: "Analyzing...", 2: "Analyzing..."}
        
        # Store previous positions for distance calculation: {track_id: (x, y)}
        self.prev_player_positions = {}

    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video, used for speed estimation.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)

        processed_frames = []

        # Process each frame in the batch
        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):
            
            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)

            # Assign clubs to players based on their tracked position
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            all_tracks = self.obj_mapper.map(all_tracks)

            # Assign the ball to the closest player and calculate speed
            all_tracks['object'], assigned_player_id = self.ball_to_player_assigner.assign(
                all_tracks['object'], self.frame_num, 
                all_tracks['keypoints'].get(8, None),  # keypoint for player 1
                all_tracks['keypoints'].get(24, None)  # keypoint for player 2
            )

            # Estimate the speed of the tracked objects
            all_tracks['object'] = self.speed_estimator.calculate_speed(
                all_tracks['object'], self.frame_num, self.cur_fps
            )
            
            # --- Update Stats: Distance & Prepare for Formation Detection ---
            formation_update_list = []
            
            for obj_key in ['player', 'goalkeeper']:
                if obj_key in all_tracks['object']:
                    for track_id, entity_data in all_tracks['object'][obj_key].items():
                        if 'projection' in entity_data and 'club' in entity_data:
                            pos = entity_data['projection']
                            club_name = entity_data['club']
                            
                            # Determine Team ID
                            team_id = -1
                            if club_name == self.club_assigner.club1.name:
                                team_id = 1
                            elif club_name == self.club_assigner.club2.name:
                                team_id = 2
                            
                            if team_id != -1:
                                # Distance Stats
                                if track_id in self.prev_player_positions:
                                    prev_pos = self.prev_player_positions[track_id]
                                    # Euclidean distance in pixels
                                    dist_pixels = np.linalg.norm(np.array(pos) - np.array(prev_pos))
                                    self.stats_manager.update_distance(team_id, dist_pixels)
                                
                                self.prev_player_positions[track_id] = pos
                                
                                # Formation Detection (Outfield players only)
                                if obj_key == 'player':
                                    formation_update_list.append({
                                        'team_id': team_id,
                                        'position': pos
                                    })

            # Update Formation Detector
            self.formation_detector.update(formation_update_list, self.frame_num)
            
            # Compute Formation periodically (every 30 frames)
            if self.frame_num % 30 == 0:
                self.team_formations[1] = self.formation_detector.compute_formation(1)
                self.team_formations[2] = self.formation_detector.compute_formation(2)

            # --- Heatmap Logic ---
            # Extract positions and teams for heatmap update
            heatmap_positions = {}
            heatmap_teams = {}
            for obj_key in ['player', 'goalkeeper']:
                if obj_key in all_tracks['object']:
                    for track_id, entity_data in all_tracks['object'][obj_key].items():
                        if 'projection' in entity_data and 'club' in entity_data:
                            heatmap_positions[track_id] = entity_data['projection']
                            heatmap_teams[track_id] = entity_data['club']
            
            self.heatmap_generator.update(heatmap_positions, heatmap_teams)
            heatmaps = self.heatmap_generator.generate_heatmaps()
            # ---------------------
            
            # --- Pass Network Logic ---
            # Get data for pass detector
            assigned_team_id = -1
            if assigned_player_id != -1:
                # Find team ID based on club name
                player_data = None
                if assigned_player_id in all_tracks['object'].get('player', {}):
                     player_data = all_tracks['object']['player'][assigned_player_id]
                elif assigned_player_id in all_tracks['object'].get('goalkeeper', {}):
                     player_data = all_tracks['object']['goalkeeper'][assigned_player_id]
                
                if player_data and 'club' in player_data:
                    club_name = player_data['club']
                    if club_name == self.club_assigner.club1.name:
                        assigned_team_id = 0
                    elif club_name == self.club_assigner.club2.name:
                        assigned_team_id = 1

            # Update Possession Stats
            # Get latest possession stats from the assigner (returns percentages 0.0-1.0)
            possession_data = self.ball_to_player_assigner.get_ball_possessions()[-1]
            # possession_data[0] is Club 1 (Team 1), possession_data[1] is Club 2 (Team 2)
            self.stats_manager.set_possession_stats(possession_data[0], possession_data[1])

            # Get Ball Location (2D)
            ball_location_2d = (0.0, 0.0)
            if 'ball' in all_tracks['object'] and all_tracks['object']['ball']:
                 # Assuming single ball track, get the first one
                 for _, ball_data in all_tracks['object']['ball'].items():
                     if 'projection' in ball_data:
                         ball_location_2d = ball_data['projection']
                         break
            
            # Update Pass Detector
            pass_events = self.pass_detector.update(
                frame_detections=None, # Not using raw detections here
                ball_location_2d=ball_location_2d,
                assigned_player_id=assigned_player_id,
                assigned_team_id=assigned_team_id
            )

            # Update Pass/Interception Stats
            for event in pass_events:
                # event.team_id is 0 or 1.
                team_id = event.team_id + 1
                self.stats_manager.update_pass_event(event.type.lower(), team_id)

            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            self.frame_num += 1

            # Annotate the current frame with the tracking information
            # Pass detected events to annotation
            annotated_frame = self.annotate(frame, all_tracks, pass_events)
            
            # --- Overlay Heatmaps ---
            annotated_frame = self._overlay_heatmaps(annotated_frame, heatmaps)

            # --- Draw Dashboard ---
            annotated_frame = self.dashboard_annotator.draw_dashboard(annotated_frame)
            
            # --- Draw Formation Info ---
            annotated_frame = self.formation_annotator.draw(annotated_frame, self.team_formations)

            # Append the annotated frame to the processed frames list
            processed_frames.append(annotated_frame)

        return processed_frames

    
    def annotate(self, frame: np.ndarray, tracks: Dict, pass_events: List = []) -> np.ndarray:
        """
        Annotates the given frame with analised data

        Args:
            frame (np.ndarray): The current video frame to be annotated.
            tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.
            pass_events (List): List of pass events to annotate.

        Returns:
            np.ndarray: The annotated video frame.
        """
         
        # Draw the frame number if required
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(frame, {'frame_num': self.frame_num})
        
        # Annotate the frame with keypoint and object tracking information
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # Project the object positions onto the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image, tracks['object'])

        # Combine the frame and projection into a single canvas
        combined_frame = self._combine_frame_projection(frame, projection_frame)

        # Annotate possession on the combined frame
        combined_frame = self._annotate_possession(combined_frame)

        # --- Draw Pass Network ---
        combined_frame = self.pass_annotator.draw(combined_frame, pass_events)

        return combined_frame
    
    def _overlay_heatmaps(self, frame: np.ndarray, heatmaps: List[np.ndarray]) -> np.ndarray:
        """
        Overlays the movement heatmaps on the left side of the frame.
        """
        # Desired width for the mini-map
        map_w = 250
        # Calculate height based on aspect ratio of the first map (all are same size)
        h, w = heatmaps[0].shape[:2]
        if w > 0:
            ratio = h / w
            map_h = int(map_w * ratio)
        else:
            map_h = 150 # Fallback
        
        gap = 10
        start_x = 20
        start_y = 150 # Start below the possession bar
        
        labels = [f"{self.club_assigner.club1.name}", f"{self.club_assigner.club2.name}", "All"]
        
        for i, heatmap in enumerate(heatmaps):
            # Resize
            heatmap_resized = cv2.resize(heatmap, (map_w, map_h))
            
            # Position
            y = start_y + i * (map_h + gap + 25) # +25 for label space
            x = start_x
            
            # Check bounds
            if y + map_h + 20 > frame.shape[0]:
                break
            
            # Draw Label
            cv2.putText(frame, labels[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Region of Interest
            roi = frame[y:y+map_h, x:x+map_w]
            
            # Blend
            alpha = 0.8
            cv2.addWeighted(heatmap_resized, alpha, roi, 1 - alpha, 0, roi)
            
            # Draw border
            cv2.rectangle(frame, (x, y), (x + map_w, y + map_h), (255, 255, 255), 1)
            
        return frame
    

    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        """
        Combines the original video frame with the projection of player positions on the field image.

        Args:
            frame (np.ndarray): The original video frame.
            projection_frame (np.ndarray): The projected field image with annotations.

        Returns:
            np.ndarray: The combined frame.
        """
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 70% of its original size
        scale_proj = 0.7
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame
    

    def _annotate_possession(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotates the possession progress bar on the top-left of the frame.

        Args:
            frame (np.ndarray): The frame to be annotated.

        Returns:
            np.ndarray: The annotated frame with possession information.
        """
        frame = frame.copy()
        overlay = frame.copy()

        # Position and size for the possession overlay (top-left with 20px margin)
        overlay_width = 500
        overlay_height = 100
        gap_x = 20  # 20px from the left
        gap_y = 20  # 20px from the top

        # Draw background rectangle (black with transparency)
        cv2.rectangle(overlay, (gap_x, gap_y), (gap_x + overlay_width, gap_y + overlay_height), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Position for possession text
        text_x = gap_x + 15
        text_y = gap_y + 30

        # Display "Possession" above the progress bar
        cv2.putText(frame, 'Possession:', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)

        # Position and size for the possession bar (20px margin)
        bar_x = text_x
        bar_y = text_y + 25
        bar_width = overlay_width - bar_x
        bar_height = 15

        # Get possession data from the ball-to-player assigner
        possession = self.ball_to_player_assigner.get_ball_possessions()[-1]
        possession_club1 = possession[0]
        possession_club2 = possession[1]

        # Calculate sizes for each possession segment in pixels
        club1_width = int(bar_width * possession_club1)
        club2_width = int(bar_width * possession_club2)
        neutral_width = bar_width - club1_width - club2_width

        club1_color = self.club_assigner.club1.player_jersey_color
        club2_color = self.club_assigner.club2.player_jersey_color
        neutral_color = (128, 128, 128)

        # Convert Club Colors from RGB to BGR
        club1_color = rgb_bgr_converter(club1_color)
        club2_color = rgb_bgr_converter(club2_color)

        # Draw club 1's possession (left)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + club1_width, bar_y + bar_height), club1_color, -1)

        # Draw neutral possession (middle)
        cv2.rectangle(frame, (bar_x + club1_width, bar_y), (bar_x + club1_width + neutral_width, bar_y + bar_height), neutral_color, -1)

        # Draw club 2's possession (right)
        cv2.rectangle(frame, (bar_x + club1_width + neutral_width, bar_y), (bar_x + bar_width, bar_y + bar_height), club2_color, -1)

        # Draw outline for the entire progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)

        # Calculate the position for the possession text under the bars
        possession_club1_text = f'{int(possession_club1 * 100)}%'
        possession_club2_text = f'{int(possession_club2 * 100)}%'

        # Display possession percentages for each club
        self._display_possession_text(frame, club1_width, club2_width, neutral_width, bar_x, bar_y, possession_club1_text, possession_club2_text, club1_color, club2_color)

        return frame
    

    def _display_possession_text(self, frame: np.ndarray, club1_width: int, club2_width: int,
                                  neutral_width: int, bar_x: int, bar_y: int, 
                                 possession_club1_text: str, possession_club2_text: str, 
                                 club1_color: Tuple[int, int, int], club2_color: Tuple[int, int, int]) -> None:
        """
        Helper function to display possession percentages for each club below the progress bar.

        Args:
            frame (np.ndarray): The frame where the text will be displayed.
            club1_width (int): Width of club 1's possession bar.
            club2_width (int): Width of club 2's possession bar.
            neutral_width (int): Width of the neutral possession area.
            bar_x (int): X-coordinate of the progress bar.
            bar_y (int): Y-coordinate of the progress bar.
            possession_club1_text (str): Text for club 1's possession percentage.
            possession_club2_text (str): Text for club 2's possession percentage.
            club1_color (tuple): BGR color of club 1.
            club2_color (tuple): BGR color of club 2.
        """
        # Text for club 1
        club1_text_x = bar_x + club1_width // 2 - 10  # Center of club 1's possession bar
        club1_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)  # Club 1's color

        # Text for club 2
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 10  # Center of club 2's possession bar
        club2_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)  # Club 2's color



    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        """
        Saves the tracking information for objects and keypoints to the specified directory.

        Args:
            all_tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.
        """
        self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
        self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])

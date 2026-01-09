from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

@dataclass
class Event:
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    type: str  # "PASS" or "INTERCEPTION"
    team_id: int # The team ID of the receiver (or the team that performed the action)

class PassEventDetector:
    """
    Tracks ball possession over time to identify passes and interceptions.
    """

    def __init__(self, possession_threshold: int = 10):
        """
        Initialize the PassEventDetector.

        Args:
            possession_threshold (int): A player must be the "closest" player for this 
                                        many consecutive frames to be considered the "owner".
        """
        self.possession_threshold = possession_threshold
        
        # State variables
        self.current_owner_id: Optional[int] = None
        self.current_owner_team_id: Optional[int] = None
        self.current_owner_location: Optional[Tuple[float, float]] = None
        
        # Helper for threshold logic
        self.potential_owner_id: Optional[int] = None
        self.consecutive_frames: int = 0
        
        self.pending_events: List[Event] = []

    def update(self, 
               frame_detections: Any, 
               ball_location_2d: Tuple[float, float], 
               assigned_player_id: int, 
               assigned_team_id: int) -> List[Event]:
        """
        Updates the state with new frame data and returns any detected events.

        Args:
            frame_detections: The raw detections for the current frame (can be used for extra context).
            ball_location_2d: The (x, y) coordinates of the ball on the 2D map.
            assigned_player_id: The ID of the player currently assigned to the ball (-1 if none).
            assigned_team_id: The team ID of the assigned player (-1 if none/unknown).

        Returns:
            List[Event]: A list of valid events detected in this update.
        """
        new_events = []

        # 1. Check if assigned_player_id is consistent
        if assigned_player_id != -1:
            if assigned_player_id == self.potential_owner_id:
                self.consecutive_frames += 1
            else:
                self.potential_owner_id = assigned_player_id
                self.consecutive_frames = 1
        else:
            # Ball is loose or lost (e.g., occlusion or mid-air).
            # We do NOT reset the counter immediately. This acts as a "grace period".
            # If the ball reappears with the same player, the count resumes.
            # If it reappears with a different player, the logic above (assigned_player_id != potential) will reset it.
            pass

        
        # Check if the location is valid (projection succeeded)
        is_loc_valid = (ball_location_2d[0] != 0 or ball_location_2d[1] != 0)

        # 2. Confirm new owner if threshold reached
        if self.consecutive_frames >= self.possession_threshold:
            new_owner_id = self.potential_owner_id
            
            # If the confirmed owner is different from the current "official" owner
            if self.current_owner_id is not None and new_owner_id != self.current_owner_id:
                
                # Logic: We have a change in possession from A -> B
                
                # Determine event type
                event_type = "PASS"
                if self.current_owner_team_id != -1 and assigned_team_id != -1:
                     if self.current_owner_team_id == assigned_team_id:
                         event_type = "PASS"
                     else:
                         event_type = "INTERCEPTION"
                
                # Create the event ONLY if we have valid coordinates for both start and end
                if self.current_owner_location is not None and is_loc_valid:
                    event = Event(
                        start_xy=self.current_owner_location,
                        end_xy=ball_location_2d,
                        type=event_type,
                        team_id=assigned_team_id
                    )
                    
                    new_events.append(event)
                    self.pending_events.append(event)

            # Update current owner state
            # (Happens first time we confirm an owner, or after an event)
            if self.current_owner_id != new_owner_id:
                self.current_owner_id = new_owner_id
                self.current_owner_team_id = assigned_team_id
                
                # We assume the location at the moment of 'acquisition' (confirmation)
                # is the start point for their possession.
                # Only update if valid. If invalid, we wait for a frame where it IS valid.
                if is_loc_valid:
                    self.current_owner_location = ball_location_2d
        
        # Optional: If the current owner is still holding the ball, we *could* update 
        # current_owner_location to the latest ball position. 
        # This makes "start_xy" of the *next* pass more accurate (the release point, not receipt point).
        # We will do this to ensure passes originate from where the player IS, not where they WERE.
        if self.current_owner_id == assigned_player_id and assigned_player_id != -1:
             if is_loc_valid:
                 self.current_owner_location = ball_location_2d

        return new_events
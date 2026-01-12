class TeamStatsManager:
    def __init__(self):
        self.stats = {
            1: {
                "total_passes": 0,
                "interceptions": 0,
                "possession_percentage": 0.0,
                "total_distance_pixels": 0.0
            },
            2: {
                "total_passes": 0,
                "interceptions": 0,
                "possession_percentage": 0.0,
                "total_distance_pixels": 0.0
            }
        }

    def set_possession_stats(self, team1_pct, team2_pct):
        """
        Directly sets the possession percentage for both teams.
        Expects percentages in range 0-1 or 0-100 (will be normalized if needed).
        Here we assume the input is 0.0-1.0 from BallToPlayerAssigner.
        """
        # Convert 0.45 -> 45.0
        self.stats[1]["possession_percentage"] = team1_pct * 100
        self.stats[2]["possession_percentage"] = team2_pct * 100

    def update_pass_event(self, event_type, team_id):
        """
        Updates pass stats based on event type.
        event_type: "pass" or "interception"
        """
        if team_id not in self.stats:
            return

        if event_type == "pass":
            self.stats[team_id]["total_passes"] += 1
        elif event_type == "interception":
            self.stats[team_id]["interceptions"] += 1

    def update_distance(self, team_id, distance_pixels):
        """Adds to total distance covered by the team."""
        if team_id in self.stats:
            self.stats[team_id]["total_distance_pixels"] += distance_pixels

    def get_stats(self, team_id):
        """
        Returns calculated statistics for the team.
        """
        if team_id not in self.stats:
            return {}

        team_data = self.stats[team_id]
        
        # Calculate Pass Completion Rate
        total_attempts = team_data["total_passes"] + team_data["interceptions"]
        if total_attempts > 0:
            pass_completion_rate = (team_data["total_passes"] / total_attempts) * 100
        else:
            pass_completion_rate = 0.0

        return {
            "possession_percentage": round(team_data["possession_percentage"], 1),
            "pass_completion_rate": round(pass_completion_rate, 1),
            "total_passes": team_data["total_passes"],
            "total_distance_pixels": round(team_data["total_distance_pixels"], 1)
        }

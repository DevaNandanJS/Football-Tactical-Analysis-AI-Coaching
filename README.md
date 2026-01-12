# ⚽ AI Football Tactical Analysis System

**A professional-grade computer vision pipeline for automated football coaching and scouting insights.**

This system transforms raw match footage into actionable tactical data. By leveraging state-of-the-art Deep Learning (YOLO), Object Tracking (BoTSORT), and geometric transformation algorithms, it digitizes the game into a comprehensive set of statistics and visualizations.

---

## 🚀 Key Features

### 1. Advanced Computer Vision
*   **Object Detection:** Uses **YOLO** models to detect Players, Goalkeepers, Referees, and the Ball with high precision.
*   **Robust Tracking:** Implements **BoTSORT** and Kalman Filters to track unique player identities across frames, handling occlusions and camera motion.
*   **Keypoint Extraction:** Automatically detects field landmarks (corners, penalty spots) to understand the camera's perspective.

### 2. Tactical Logic Engine
*   **Perspective Transformation (Homography):** Maps 2D screen pixels to real-world pitch coordinates (meters), allowing for accurate distance and speed calculations.
*   **Team Classification:** Uses **K-Means Clustering** on player jersey colors to automatically separate teams (e.g., Al Nassr vs. Opponent).
*   **Possession Logic:** a sophisticated proximity-based state machine determines which player controls the ball at any millisecond.
*   **Pass Event Detection:** Automatically identifies passes, interceptions, and possession turnovers.

### 3. Analytics & Visualization Output
The system generates a suite of artifacts for coaches:
*   **Annotated Video:** Overlay including Player IDs, Team colors, Speed (km/h), and a Mini-Map radar.
*   **Movement Heatmaps:** Gaussian-smoothed density maps showing player activity zones.
*   **Pass Networks:** Graph visualizations showing passing connections and tactical shapes.
*   **Match Stats Report:** A generated HTML dashboard with possession %, total passes, and distance covered.
*   **JSON Telemetry:** Full raw data export of every player's position for external analysis.

---

## 🛠️ Installation

1.  **Clone the Project:**
    ```bash
    git clone <repository-url>
    cd Prototype-Football
    ```

2.  **Set up Environment:**
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate (Windows)
    .venv\Scripts\activate
    
    # Activate (Mac/Linux)
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r football_analysis/requirements.txt
    ```

4.  **Model Setup:**
    Ensure your YOLO weights are placed in `football_analysis/models/`:
    *   `yolo-detect.pt` (Object Detection)
    *   `yolo-keypoints.pt` (Field Keypoint Detection)

---

## 💻 Usage

1.  **Configuration:**
    Open `football_analysis/main.py`. You can configure:
    *   **Input Video:** Set `video_source`.
    *   **Team Colors:** Adjust `Club` RGB values to match the jerseys in your video.
    *   **Confidence Thresholds:** Tweak `conf` for detection sensitivity.

2.  **Run the Pipeline:**
    ```bash
    python football_analysis/main.py
    ```

3.  **View Results:**
    Outputs are saved to `football_analysis/output_videos/`:
    *   `*_out.mp4`: The fully annotated video.
    *   `match_stats.html`: The interactive stats report.
    *   `heatmap_*.png`: Tactical heatmaps.
    *   `pass_network.png`: Team passing structure.

---

## 📂 Project Structure

```
football_analysis/
├── analysis/               # Core Analytical Logic
│   ├── pass_event_detector.py    # Logic for detecting passes
│   ├── movement_heatmap_generator.py # Generates heatmaps
│   └── team_stats_manager.py     # Aggregates match statistics
├── annotation/             # Visualization Modules
│   ├── football_video_processor.py # Main orchestration loop
│   └── ... (Annotators for text, map, overlays)
├── models/                 # Neural Network Weights
├── tracker/                # BoTSORT & Kalman Filter implementation
├── utils/                  # Helper functions (Video IO, Bbox math)
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

---

## 📊 Methodology

This project treats football analysis as a multi-stage pipeline:
1.  **Ingest:** Read video frames asynchronously.
2.  **Perceive:** Detect objects and keypoints.
3.  **Contextualize:** Map pixels to meters (Homography) and assign teams (Clustering).
4.  **Reason:** Apply football rules (Possession, Passing) to the spatial data.
5.  **Report:** Render visual and statistical outputs.

---

## 📄 License
[Your License Here]

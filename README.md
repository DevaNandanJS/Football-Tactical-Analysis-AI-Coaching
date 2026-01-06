# Football Analysis

This project performs detailed analysis of football match videos. It uses computer vision techniques to detect, track, and analyze players, the ball, and their movements on the field.

## Features

*   **Object Detection:** Utilizes YOLOv11 to detect and classify objects on the field, including players, goalkeepers, referees, and the football.
*   **Keypoint Detection:** Identifies key points on the football field to compute a homography matrix for perspective transformation.
*   **Player & Ball Tracking:** Employs the `supervision` library to track the detected objects across video frames.
*   **Club Assignment:** Assigns players to their respective clubs based on the primary colors of their jerseys.
*   **Ball Possession:** Determines which player and club is in possession of the ball.
*   **2D Field Projection:** Maps the real-world 3D coordinates of players and the ball onto a 2D top-down view of the football field using the calculated homography.
*   **Speed Estimation:** Calculates and displays the speed of each player in real-time.
*   **Video Annotation:** Generates an output video with annotations, including bounding boxes, tracking IDs, club names, player speeds, and their corresponding positions on the 2D map.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DevaNandanJS/Football-Tactical-Analysis-AI-Coaching-Report.git
    cd football_analysis
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project requires specific PyTorch dependencies. Install the requirements from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models:**
    You will need YOLOv8 models for object detection and keypoint detection. Place them in the `models/` directory.
    - `yolo-detect.pt`: For detecting players, ball, etc.
    - `yolo-keypoints.pt`: For detecting field keypoints.

## Usage

1.  **Configure the Analysis:**
    Open `main.py` and modify the following sections as needed:
    *   **Model Paths:** Update the paths to your YOLO models inside the `ObjectTracker` and `KeypointsTracker` initializations.
    *   **Club Information:** Define the names and jersey colors for the two clubs in the `Club` objects.
    *   **Input/Output:** Set the paths for the input video (`video_source`) and the desired output video (`output_video`) in the `process_video` function call.

2.  **Run the script:**
    Execute the `main.py` script to start the analysis.
    ```bash
    python main.py
    ```
    The processed video with all annotations will be saved to the specified output path.

## Project Structure

```
football_analysis/
├── annotation/         # Modules for annotating video frames.
├── ball_to_player_assignment/ # Logic for assigning ball possession.
├── club_assignment/    # Logic for assigning players to clubs.
├── input_videos/       # Contains input videos and field images.
├── models/             # YOLO models for detection and keypoints.
├── output_videos/      # Directory for output videos and track data.
├── position_mappers/   # Modules for mapping positions to a 2D field.
├── speed_estimation/   # Logic for estimating player speeds.
├── tracking/           # Modules for object and keypoint tracking.
├── utils/              # Utility functions.
├── main.py             # Main script to run the analysis.
└── requirements.txt    # Project dependencies.
```

## Dependencies

The main dependencies are listed in `requirements.txt` and include:
- `torch` & `torchvision`
- `ultralytics` (for YOLO)
- `supervision`
- `opencv-python`
- `pandas`
- `scikit-learn`
- `roboflow`
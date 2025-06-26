# Face Identify: Advanced Real-Time Face Analysis Suite

## Overview

Face Identify is a comprehensive, real-time face analysis and monitoring system. It leverages state-of-the-art computer vision and deep learning libraries to provide:

- Face detection and recognition
- Age, gender, race, and emotion analysis
- Attention and drowsiness monitoring
- Health and fatigue tracking
- Security monitoring (unauthorized access alerts)
- Gesture and voice command controls
- Environmental analysis (light, frame quality)
- Detailed session statistics and dashboard
- Modern Tkinter-based GUI

## Features

- **Face Detection & Recognition**: Detects multiple faces, recognizes known individuals (optional).
- **Demographic Analysis**: Estimates age, gender, race, and emotion for each face.
- **Attention & Drowsiness**: Tracks eye aspect ratio, blink rate, and head pose to monitor attention and fatigue.
- **Health Monitoring**: Fatigue, drowsiness, and mask detection.
- **Security**: Alerts for unauthorized access attempts.
- **Gesture & Voice Control**: Control the app with hand gestures and voice commands.
- **Environment Analysis**: Monitors lighting and frame quality.
- **Statistics Dashboard**: Real-time and session stats, visualized in a modern overlay.
- **Audio Alerts**: Configurable sound alerts for various events.

## Installation

### Prerequisites

- Python 3.8–3.12 recommended
- pip (Python package manager)
- [Optional] CUDA-enabled GPU for deep learning acceleration

### System Packages

- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Windows**: Tkinter is included with most Python installations

### Clone the Repository

```bash
git clone https://github.com/yourusername/face_identify.git
cd face_identify
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Download Pretrained Models

Some models (e.g., for age/gender) are included in `models/`. If missing, download from official sources or DeepFace will auto-download as needed.

## Usage

Run the main application:

```bash
python main.py
```

- The webcam will activate and the GUI will appear.
- Use the on-screen controls to toggle features (face recognition, sound, stats, etc.).
- Press `L` to toggle the statistics dashboard.
- All detections and session logs are saved in `face_detection_output/`.

## Directory Structure

```
face_identify/
├── main.py
├── requirements.txt
├── README.md
├── analyzers/
├── detectors/
├── controls/
├── security/
├── ui/
├── models/
├── face_detection_output/
└── ...
```

## Troubleshooting

### MediaPipe Initialization Error

If you see an error like:

```
RuntimeError: ValidatedGraphConfig Initialization failed.
ConstantSidePacketCalculator: ... Number of output side packets has to be same as number of packets configured in options.
```

- Ensure your `mediapipe` version is compatible with your Python version (try `pip install --upgrade mediapipe`).
- If using a virtual environment, ensure it is activated and all dependencies are installed inside it.
- Try running with a different Python version (3.8–3.10 is most stable for MediaPipe).

### Other Issues

- For `tkinter` errors: Install `python3-tk` via your OS package manager.
- For missing models: Ensure all files in `models/` are present, or let DeepFace auto-download.
- For sound issues: Ensure `pygame` is installed and your system audio is working.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Credits

- [DeepFace](https://github.com/serengil/deepface)
- [MediaPipe](https://github.com/google/mediapipe)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Pillow](https://python-pillow.org/)
- [pygame](https://www.pygame.org/)
- And all other open-source contributors.

---

For questions or contributions, open an issue or pull request on GitHub.

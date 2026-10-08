# PS07: Autonomous Vision and Behaviour Understanding

## Aquatic Distress Behaviour Intelligence

**HackNEX 2026 | NEXUS Club, Karunya**

### 1. What the Project Does

Aquatic Distress Behaviour Intelligence is a computer-vision-based safety system designed to analyse swimming-pool video and identify behaviour that may indicate a swimmer is in distress. The system processes video from a fixed camera and uses object detection, object tracking, behaviour analysis, and temporal reasoning to understand how a person is moving over time rather than simply detecting a person in each individual frame.

The main objective is to identify high-risk aquatic behaviour by observing changes in a swimmer's movement pattern. The system can use the pool scenario to look for behaviour such as normal swimming followed by slowing down, becoming vertical, repeated arm movement, and eventually showing little or no forward progress. It can also identify possible submersion behaviour based on the configured rules.

The system follows a complete video-analysis pipeline:

```text
Input Video
     ↓
Object Detection
     ↓
Object Tracking
     ↓
Feature Extraction
     ↓
Behaviour Analysis
     ↓
Temporal Distress Reasoning
     ↓
Event Generation
     ↓
Severity and Incident Analysis
     ↓
Evidence Generation
     ↓
Report + Highlight Video + Dashboard
```

For the aquatic scenario, the system uses the `pool` scenario preset. This preset tracks people or a custom swimmer model and contains rules for **high-risk aquatic distress** and **possible submersion**. The pool pipeline is intended to support a lifeguard by converting video observations into understandable incidents and evidence rather than simply displaying detection bounding boxes.

The general engine is designed around the idea that meaningful safety incidents require information from multiple frames. Instead of asking only whether a person is visible, the system measures how an object moves, how its behaviour changes, how long a behaviour continues, and whether the observed sequence satisfies a configured rule.

The system produces structured outputs such as `events.json`, `summary.txt`, `report.html`, annotated video, snapshots, plots, a timeline, and a heatmap. These outputs allow an operator or evaluator to understand what the system detected, when it happened, and what evidence caused the event to be reported.

The system also supports incident chains, where multiple related events belonging to the same entity can be grouped into a single incident. This makes the result easier to understand because the system can describe a sequence of behaviours instead of presenting every individual rule trigger as a separate unrelated event.

---

## 2. Technologies, Libraries, and Models Used

The project is implemented primarily in **Python** and uses a combination of computer vision, object detection, object tracking, numerical processing, rule-based reasoning, visualization, and web technologies.

### Core Programming Language

**Python 3.11** is used as the main programming language. Python connects all stages of the system, including video processing, detection, tracking, feature extraction, behaviour analysis, event generation, reporting, evaluation, and the Streamlit interface.

The complete pipeline is coordinated by `run.py`, which connects the different processing modules and executes them in the required order.

### YOLO11n

The project uses **Ultralytics YOLO11n** as the main object detection model.

YOLO11n is a pretrained object detector used to identify objects in video frames. For the general engine, the detector can identify classes such as people, bicycles, cars, motorbikes, buses, trucks, dogs, horses, sheep, and cows depending on the selected scenario.

For the aquatic scenario, the important tracked class is the **person/swimmer**. The pool preset can also be configured to use a custom swimmer detector when required.

The project uses the pretrained model rather than training a new detection model. The `yolo11n.pt` weights are downloaded automatically during the first execution and cached for later runs.

### ByteTrack

**ByteTrack** is used for object tracking.

Object detection identifies objects independently in individual frames, but tracking is required to understand how the same person moves across multiple frames. ByteTrack associates detections between frames and assigns stable IDs to tracked objects.

For example:

```text
Frame 1 → Person #1
Frame 2 → Person #1
Frame 3 → Person #1
Frame 4 → Person #1
```

This allows the system to build a movement history for each detected person and analyse behaviour over time.

### OpenCV

**OpenCV (`opencv-python`)** is used for video processing and computer-vision operations.

It is responsible for tasks such as:

* Reading input videos
* Processing video frames
* Resizing frames
* Writing output videos
* Drawing bounding boxes and tracking information
* Creating annotated frames
* Supporting privacy blur
* Supporting video rendering

The system normally resizes frames to the configured processing width and processes frames according to the configured frame stride.

### NumPy

**NumPy** is used for numerical calculations throughout the system.

It supports calculations involving:

* Object positions
* Movement distance
* Velocity
* Speed
* Body-size normalization
* Distance measurements
* Temporal measurements
* Behaviour thresholds
* Statistical baseline calculations

### PyYAML

**PyYAML** is used to read the project's YAML configuration files.

The main configuration is stored in:

```text
config.yaml
```

Scenario-specific settings are stored in:

```text
scenarios/
```

This allows detection thresholds, behaviour thresholds, scenario wording, and other settings to be changed without modifying the core Python code.

### Matplotlib

**Matplotlib** is used to generate visual evidence and analysis plots.

The system can create plots showing information such as:

* Speed over time
* Distance over time
* Closing speed
* Behaviour thresholds
* Event timelines

These plots are stored along with the other evidence generated for an analysed video.

### Streamlit

**Streamlit** provides the project's browser-based dashboard.

The Streamlit application is started using:

```powershell
streamlit run app.py
```

The dashboard allows the user to select or upload a video, choose a scenario, configure options, run the analysis, and inspect the generated results.

The dashboard presents results through sections such as:

* Summary
* Highlight reel
* Incidents
* Timeline and heatmap
* Full report
* Annotated video
* Downloads

### PyTorch

**PyTorch** is used to run the machine-learning models used by the project. It is installed as part of the Ultralytics model environment.

### FFmpeg

**FFmpeg** is optional and is used for video conversion when available. It helps convert generated videos into H.264 format so that they can be played easily in browsers.

### Rule-Based Behaviour Analysis

The project does not use a separately trained behaviour-classification model. Behaviour decisions are made using **explainable rules and configurable thresholds**.

This makes the reasoning easier to inspect because an event can include the measured values, threshold, confidence components, and evidence supporting the decision.

The repository explicitly states that no model is trained or fine-tuned for the behaviour rules; the behaviour logic is written by hand.

### Main Technical Stack

```text
Python
│
├── YOLO11n
│   └── Object Detection
│
├── ByteTrack
│   └── Object Tracking
│
├── OpenCV
│   └── Video Processing
│
├── NumPy
│   └── Numerical and Feature Calculations
│
├── Rule-Based Behaviour Analysis
│   └── Distress / Event Reasoning
│
├── PyYAML
│   └── Configuration
│
├── Matplotlib
│   └── Evidence Plots
│
└── Streamlit
    └── Web Dashboard
```

---

## 3. How to Install Dependencies

### Requirements

The project is designed to run locally on **Windows, macOS, or Linux** using Python 3.11.

A GPU is not mandatory for running the project. The system can run on a CPU, although video processing is significantly faster when a suitable GPU is available.

The repository contains a `requirements.txt` file containing the Python dependencies required by the project.

### Step 1: Clone the Repository

Clone the project repository:

```powershell
git clone https://github.com/Lokeshraj2006/aquatic-distress-drowning-behavioural-.git
```

Move into the project directory:

```powershell
cd aquatic-distress-drowning-behavioural-
```

### Step 2: Create a Virtual Environment

For Python 3.11:

```powershell
py -3.11 -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

For macOS or Linux:

```bash
source .venv/bin/activate
```

Using a virtual environment keeps the project's Python packages isolated from other Python projects on the system.

### Step 3: Install Dependencies

Install the required packages using:

```powershell
pip install -r requirements.txt
```

The requirements include the libraries used for object detection, tracking, video processing, numerical computation, visualization, configuration, and the Streamlit dashboard.

If a suitable NVIDIA GPU is available, PyTorch can be installed according to the CUDA version of the system before installing the remaining requirements.

### Step 4: Download Sample Data

The project provides a sample-data script:

```powershell
python get_samples.py
```

This downloads the sample video material used for demonstrations and also creates synthetic test clips with known ground-truth labels.

The sample videos include footage used to demonstrate the project's behaviour-analysis pipeline.

### Step 5: Verify the Installation

The project also contains automated tests.

First generate the synthetic clips without downloading the Intel sample videos if required:

```powershell
python get_samples.py --no-intel
```

Then run:

```powershell
pip install pytest
python -B -m pytest -p no:cacheprovider -q tests
```

The tests cover core behaviour logic and parts of the web interface and helper functions.

---

## 4. How to Configure and Run the System

The project uses a central configuration file:

```text
config.yaml
```

Scenario-specific configuration is stored inside:

```text
scenarios/
```

The system loads the main configuration and then applies the selected scenario preset.

### Configuration

Important configuration parameters include:

| Configuration               | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| `video.resize_width`        | Controls the width used for processing frames   |
| `video.frame_stride`        | Controls how frequently frames are processed    |
| `model.conf`                | Minimum object-detection confidence             |
| `model.classes`             | Object classes tracked by the detector          |
| `features.smoothing_s`      | Smoothing duration for movement measurements    |
| `features.min_track_s`      | Minimum duration required for a valid track     |
| `behaviors.*`               | Behaviour detection thresholds                  |
| `events.merge_gap_s`        | Time gap used to merge related event segments   |
| `events.near_miss_ratio`    | Threshold for almost-flagged cases              |
| `events.confidence_weights` | Weights used to calculate event confidence      |
| `chains.max_gap_s`          | Maximum gap between events in an incident chain |
| `privacy.*`                 | Privacy and blur settings                       |
| `output.*`                  | Output and rendering settings                   |

For example, the default configuration processes frames at a configured resize width and frame stride. The behaviour rules then use the processed tracking information to determine whether an event occurred.

### Scenario Selection

The system supports scenario presets through the `--scenario` argument.

For the aquatic project, use:

```powershell
python run.py --video <pool_video>.mp4 --scenario pool
```

The `pool` scenario is configured for aquatic distress behaviour and possible submersion.

The scenario changes the classes being tracked, thresholds, behaviour rules, and wording used in the generated results.

### Command-Line Execution

The basic form of the command is:

```powershell
python run.py --video <VIDEO_PATH> --scenario pool
```

For example:

```powershell
python run.py --video samples/pool_video.mp4 --scenario pool
```

The pipeline then performs detection, tracking, feature extraction, behaviour analysis, event generation, incident processing, and output generation.

### Important Runtime Options

The project provides several command-line options.

```text
--video PATH
```

Specifies the input video.

```text
--scenario NAME
```

Selects a scenario preset such as `pool`.

```text
--zones zones.json
```

Provides a zone definition when the selected rules require a specific area.

```text
--privacy
```

Enables privacy processing for rendered outputs.

```text
--device auto|cpu|0
```

Selects whether YOLO should use automatic device selection, CPU, or a GPU device.

```text
--stride N
```

Controls the number of frames skipped between processed frames.

```text
--max-seconds S
```

Limits analysis to the first specified number of seconds.

```text
--pose
```

Enables the optional pose-verification stage.

```text
--no-video
```

Skips video rendering to make processing faster.

```text
--reuse
```

Reuses compatible cached detections instead of running YOLO again.

### Streamlit Dashboard

The project also provides a web-based interface.

Start it using:

```powershell
streamlit run app.py
```

The dashboard normally opens at:

```text
http://localhost:8501
```

The interface provides a browser-based way to select a video, choose the scenario, configure optional settings, start analysis, and inspect the results.

The Streamlit application does not implement a separate analysis pipeline. It runs `run.py` as a subprocess and displays the files generated by the same command-line pipeline. This keeps the command-line and dashboard results consistent.

### Dashboard Workflow

The user can:

1. Select or upload a video.
2. Select the required scenario.
3. Configure an optional zone.
4. Select analysis options.
5. Start the analysis.
6. Monitor the progress and console output.
7. View generated incidents and events.
8. Open the highlight reel.
9. Inspect the timeline and heatmap.
10. Open the complete HTML report.
11. Watch the annotated video.
12. Download the generated output files.

---

## 5. How to Reproduce the Demonstrated Results

The repository contains precomputed examples and sample inputs so that the demonstrated behaviour of the system can be reproduced without creating a new dataset from scratch.

The repository includes an `examples/` directory containing already-computed results. These results can be opened directly through their `report.html` files to inspect the generated reports before running the complete pipeline.

### Step 1: Install the Project

Follow the installation procedure described above:

```powershell
git clone https://github.com/Lokeshraj2006/aquatic-distress-drowning-behavioural-.git
cd aquatic-distress-drowning-behavioural-
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Download Sample Inputs

Run:

```powershell
python get_samples.py
```

This prepares the sample clips and synthetic demonstration videos.

### Step 3: Run the Main Pipeline

For a general sample video:

```powershell
python run.py --video samples/one-by-one-person-detection.mp4
```

For the aquatic scenario:

```powershell
python run.py --video <pool_video>.mp4 --scenario pool
```

The first execution downloads the YOLO11n weights if they are not already available. The weights are then cached for subsequent runs.

### Step 4: Reproduce the Demonstrated Safety Scenario

The repository includes a synthetic workplace safety clip containing known events. This provides a reproducible example for evaluating the event-detection and reporting pipeline.

Run:

```powershell
python run.py --video samples/synthetic_safety.mp4 --scenario workplace --out outputs/synthetic_safety --reuse
```

The generated output contains incidents such as:

* A person and bicycle near miss
* A person falling
* A speeding bicycle

The repository reports that this synthetic demonstration was evaluated against ground-truth labels and produced three true positives with zero false positives and zero false negatives.

### Step 5: Run Evaluation

The evaluation script compares generated `events.json` results with manually prepared ground-truth labels.

For the complete output directory:

```powershell
python evaluate.py --labels labels.csv --outputs outputs/
```

For one specific result:

```powershell
python evaluate.py --labels labels.csv --events outputs/corridor/events.json --clip corridor.mp4
```

The evaluation compares the predicted behaviour and time interval with the ground-truth label. A tolerance of two seconds is used by default.

The evaluation produces:

* True Positives
* False Positives
* False Negatives
* Precision
* Recall
* F1-score
* Mean start-time error
* Mean end-time error
* False alarms on clips containing no events

The results are printed and saved as:

```text
eval_report.txt
eval_report.json
```

### Step 6: Inspect the Generated Results

After the pipeline finishes, open:

```text
outputs/<video-name>/report.html
```

or, when using a scenario preset:

```text
outputs/<video-name>_<scenario>/report.html
```

The report provides a detailed view of the detected events and incidents.

The main generated files include:

| Output               | Description                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `report.html`        | Complete report containing summaries, incidents, evidence cards, timeline, heatmap, and settings |
| `highlights.mp4`     | Video containing only detected incidents                                                         |
| `summary.txt`        | Short textual summary of the important incidents                                                 |
| `events.json`        | Structured event and incident information                                                        |
| `annotated.mp4`      | Video containing detection boxes, IDs, trails, labels, and timestamps                            |
| `snapshots/e<N>.jpg` | Evidence frame for each event                                                                    |
| `plots/e<N>.png`     | Behaviour and threshold plots                                                                    |
| `heatmap.jpg`        | Visual representation of object movement                                                         |
| `timeline.png`       | Timeline of tracks and detected events                                                           |
| `detections.json`    | Cached YOLO and ByteTrack detections                                                             |

These outputs allow the demonstrated result to be checked both visually and through structured data.

### Reproducing the Web Demonstration

The Streamlit interface can be launched using:

```powershell
streamlit run app.py
```

For the available demonstration results, the dashboard can also open existing folders containing `events.json`. This allows completed results to be inspected without running the detector again.

For a quick demonstration of the project's event-analysis interface, the repository provides a synthetic safety clip that can be selected in the Streamlit interface. Cached detections can be reused so that the demonstration runs much faster than a complete first-time YOLO analysis.

---

## System Workflow

The complete processing workflow can be summarized as follows:

### 1. Video Input

The system receives a video from a fixed camera. The input can be an existing video file or a video recorded using the webcam option.

### 2. Video Preprocessing

OpenCV reads the video and obtains its frame rate. Frames are resized according to `video.resize_width`, and the configured frame stride determines which frames are processed.

### 3. Object Detection

YOLO11n detects objects in the processed frames. For the pool scenario, the main target is the swimmer/person.

### 4. Object Tracking

ByteTrack associates detections between frames and assigns IDs. This creates a continuous track for each detected person.

### 5. Feature Extraction

For every tracked object, the system calculates features such as:

* Position
* Foot point
* Body size
* Aspect ratio
* Speed
* Velocity
* Movement direction
* Track duration

The movement values are normalized using the object's body size, which allows movement to be compared across different distances from the camera.

### 6. Behaviour Analysis

The behaviour engine applies rules to the tracked features. For the broader engine these include behaviours such as running, loitering, zone intrusion, falling, crowding, and near misses.

For the pool scenario, the relevant rules focus on aquatic distress behaviour and possible submersion.

### 7. Temporal Reasoning

The system does not make a decision based on a single frame. Behaviour must satisfy temporal conditions such as duration, movement pattern, and sequence.

For example, the pool scenario describes a high-risk sequence involving:

```text
Normal Swimming
      ↓
Slowing
      ↓
Vertical Position
      ↓
Repeated Arm Motion
      ↓
No Forward Progress
      ↓
Sustained for the Required Duration
      ↓
High-Risk Aquatic Distress Event
```

### 8. Event Generation

Once a behaviour satisfies its rule, the system generates an event containing information such as the entity, behaviour, start time, end time, confidence, severity, and evidence.

### 9. Incident Formation

Related events can be combined into incident chains. This provides a higher-level description of what happened rather than showing every individual rule trigger separately.

### 10. Evidence Generation

For each event, the system can generate:

* Evidence snapshots
* Behaviour plots
* Annotated video
* Timeline
* Heatmap
* Evidence descriptions

### 11. Reporting

The final results are written to structured and human-readable outputs such as:

```text
events.json
summary.txt
report.html
highlights.mp4
annotated.mp4
```

### 12. Dashboard Presentation

The Streamlit interface displays the generated information in a form intended for quick inspection by a user such as a lifeguard or safety operator.

---

## Project Structure

```text
ps07/
│
├── run.py
├── tracker.py
├── features.py
├── behaviors.py
├── events.py
├── chains.py
├── pose_verify.py
├── render.py
├── highlights.py
├── privacy.py
├── report.py
├── zone_picker.py
│
├── app.py
├── ui_helpers.py
├── evaluate.py
├── utils.py
│
├── config.yaml
├── scenarios/
├── requirements.txt
├── bytetrack_custom.yaml
├── DESIGN.md
├── SCOPE.md
├── colab_run.ipynb
├── labels_template.csv
├── zones.example.json
│
├── samples/
├── examples/
└── tests/
```

The major modules are separated according to their responsibilities. For example, `tracker.py` performs detection and tracking, `features.py` creates per-object movement features, `behaviors.py` applies behaviour rules, `events.py` creates and filters events, `chains.py` creates incident chains, and `report.py` generates the final reports.

---

## Summary

Aquatic Distress Behaviour Intelligence combines **YOLO11n object detection, ByteTrack tracking, OpenCV video processing, feature extraction, rule-based temporal behaviour reasoning, evidence generation, and a Streamlit dashboard** into a single video-analysis pipeline.

The system is designed to move beyond simple object detection by analysing the behaviour of tracked swimmers over time. For the aquatic scenario, it focuses on identifying patterns associated with high-risk distress and possible submersion.

The repository provides the required installation instructions, configuration files, scenario presets, command-line execution, Streamlit dashboard, sample inputs, generated outputs, evaluation scripts, and reproducible demonstrations. This makes the project suitable for demonstrating the complete pipeline from video input to behaviour analysis, incident generation, evidence creation, and final reporting.
#   H e x T e c h  
 
# Video Shot Boundary Detection

This repository demonstrates various methods for **Video Shot Boundary Detection**, encompassing both traditional and deep learning techniques. It includes:

1. **Traditional Methods:**
   - Histogram Difference
   - Pixel Difference
2. **Deep Learning Methods:**
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Implementation](#implementation)
5. [Results](#results)
6. [Getting Started](#getting-started)
7. [References](#references)
8. [Visualization Example](#visualization-example)

---

## Overview

Shot boundary detection identifies the transitions between shots in a video. Accurate detection is essential for video editing, content summarization, and scene analysis. This project explores both traditional and deep learning approaches to achieve this.

---

## Dataset

- **Source:** Publicly available video datasets (e.g., TRECVID, YouTube datasets).
- **Annotations:** Transition labels (cut, fade, dissolve, etc.)
- Preprocessed into frames and labeled for training and testing.
- **Access the dataset here:** [Dataset Drive Link](https://drive.google.com/drive/folders/1O7meiXuVUgKlNHGFYmoJ1ZX7IoWCxzVj?usp=drive_link)

---

## Methods

### Traditional Methods

#### 1. Histogram Difference

- **Description:** Compares the histograms of consecutive frames to detect transitions.
- **Implementation:** Measures the histogram similarity using metrics like Bhattacharyya or Chi-Square distance.

#### 2. Pixel Difference

- **Description:** Computes the pixel-wise intensity difference between frames.
- **Implementation:** Calculates the absolute difference and thresholds significant changes.
- 
![Screenshot 2024-12-12 221841](https://github.com/user-attachments/assets/1e66672b-8188-43a4-9c7d-470c17dd6b8e)

**Resources:**

- [Traditional Methods Drive Link](https://drive.google.com/drive/folders/1-cri3JvbEtt6RfzwttKJe12qNEyggPFG?usp=drive_link)
- [Traditional Methods Outputs Drive Link](https://drive.google.com/drive/folders/1RlJ7N2PIF3-YJbeG6KX23Ccbp05By42H?usp=drive_link)

### Deep Learning Methods

#### 3. Convolutional Neural Networks (CNN)

- **Description:** Extracts spatial features from video frames to detect transitions.
- **Architecture:** Includes convolutional layers followed by fully connected layers.

#### 4. Recurrent Neural Networks (RNN)

- **Description:** Models the temporal dependencies in video sequences.
- **Architecture:** Utilizes LSTM or GRU cells for sequential frame analysis.

**Resources:**

- [Deep Learning Methods Drive Link](https://drive.google.com/drive/folders/10RBtSPGyZeeGnhxGMFrAedLSG563Emfk?usp=sharing)
- [Deep Learning Outputs Drive Link](https://drive.google.com/drive/folders/13fW_Bl856KteebiND1REqpJOm1V4phip?usp=drive_link)

---

## Implementation

### Histogram Difference Method

This MATLAB implementation detects shot boundaries by calculating the histogram difference between consecutive video frames.

- **Steps:**

  1. Load the video file and extract the total number of frames.
  2. Compute the grayscale histogram for each frame.
  3. Calculate the absolute difference between histograms of consecutive frames.
  4. Identify major shot boundaries by detecting significant spikes in histogram difference values.

- **Visualization:** The differences are plotted to visualize spikes, which correspond to shot boundaries. Red markers indicate detected transitions.

### Pixel Difference Method

This MATLAB implementation uses pixel intensity differences to detect shot boundaries.

- **Steps:**

  1. Resize each grayscale frame to a smaller matrix size (e.g., 5x5) to simplify computations.
  2. Calculate the sum of absolute differences between pixel intensities of consecutive frames.
  3. Identify shot boundaries by comparing differences against a defined threshold.

- **Additional Features:**

  - Displays each resized frame matrix for debugging.
  - Prints detected shot boundaries with their frame numbers and difference values.

### Deep Learning Methods Implementation

#### 1. CNN-Based Results for `shot_test.mp4`

- **Steps:**
  1. Extract video frames and preprocess them to match the CNN input size.
  2. Train the CNN using labeled frames to classify transitions.
  3. Evaluate performance metrics, including Precision, Recall, and F1-Score.

- **Visualization:**
  Include a graph for `shot_test.mp4` showcasing **Recall**.

#### 2. RNN-Based Results for `11.mp4`

- **Steps:**
  1. Use CNN features as input to the RNN to model temporal dependencies.
  2. Train the RNN to classify transitions and evaluate metrics.
  3. Display performance metrics including a graph showcasing **F1-Score** for `11.mp4`.

### Hardware Used

- **Laptop Specifications:**
  - Processor: Intel i5
  - Graphics: NVIDIA RTX 3050

---

## Results

### Evaluation Metrics

- Precision, Recall, F1-Score
- Accuracy of transition detection (Cut, Fade, Dissolve)

### Comparative Results

| Method               | Precision | Recall | F1-Score |
| -------------------- | --------- | ------ | -------- |
| Histogram Difference | 75%       | 80%    | 77%      |
| Pixel Difference     | 70%       | 75%    | 72%      |
| CNN                  | 85%       | 87%    | 86%      |
| RNN                  | 83%       | 85%    | 84%      |

---

## Getting Started

### Prerequisites

- MATLAB R2021a or later
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-shot-boundary-detection.git

# Navigate to the project directory
cd video-shot-boundary-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Running Traditional Methods

To run the traditional methods:

1. Open the MATLAB scripts (`Histogram Difference` and `Pixel Difference` methods).
2. Replace the `videoPath` variable with the path to your video file.
3. Execute the script to detect and visualize shot boundaries.

**Resources:**

- [Traditional Methods Drive Link](https://drive.google.com/drive/folders/1-cri3JvbEtt6RfzwttKJe12qNEyggPFG?usp=drive_link)
- [Traditional Methods Outputs Drive Link](https://drive.google.com/drive/folders/1RlJ7N2PIF3-YJbeG6KX23Ccbp05By42H?usp=drive_link)

#### Running Deep Learning Methods

1. Preprocess the dataset into labeled frames.
2. Train the CNN or RNN model using the provided architecture.
3. Evaluate the model on the test set.

**Resources:**

- [Deep Learning Methods Drive Link](https://drive.google.com/drive/folders/10RBtSPGyZeeGnhxGMFrAedLSG563Emfk?usp=sharing)
- [Deep Learning Outputs Drive Link](https://drive.google.com/drive/folders/13fW_Bl856KteebiND1REqpJOm1V4phip?usp=drive_link)

---

## References

- Papers and resources on video shot boundary detection.
- Open-source datasets used for experimentation.

---

## Visualization Example

Create a folder named `Results` in the repository to store and visualize detected boundary images. This folder will contain:

1. Processed frames highlighting shot boundaries.
2. Plots showing the spikes in differences (for histogram and pixel methods).

To upload and visualize these images, you can:

- Use platforms like [Imgur](https://imgur.com) or [Google Drive](https://drive.google.com) for public sharing.
- Add these images to a web page or presentation to showcase your findings.


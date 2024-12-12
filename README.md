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
8. [Conclusion and Future Work](#Conclusion and Future Work)

---

## Overview

Shot boundary detection is the process of identifying the transitions between consecutive shots in a video. A 'shot' refers to an uninterrupted sequence of frames captured by a single camera. Transition types include abrupt changes (cuts) and gradual transitions (fades, dissolves). Accurate detection of these transitions is vital for tasks such as video editing, content summarization, video indexing, and scene analysis. This project explores both traditional and deep learning approaches to achieve reliable shot boundary detection. Accurate detection is essential for video editing, content summarization, and scene analysis. This project explores both traditional and deep learning approaches to achieve this.

![Screenshot 2024-12-12 223423](https://github.com/user-attachments/assets/a8637eb3-6602-4588-bc24-cf36b4b1d480)

![Screenshot 2024-12-12 223454](https://github.com/user-attachments/assets/2d92aef9-343a-4070-97d5-c6fb925d79fa)

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
![Screenshot 2024-12-12 222026](https://github.com/user-attachments/assets/07138850-8ab9-46c4-8b5f-df498e9383bf)

#### 2. Pixel Difference

- **Description:** Computes the pixel-wise intensity difference between frames.
- **Implementation:** Calculates the absolute difference and thresholds significant changes.
![Screenshot 2024-12-12 221841](https://github.com/user-attachments/assets/1e66672b-8188-43a4-9c7d-470c17dd6b8e)

**Resources:**

- [Traditional Methods Drive Link](https://drive.google.com/drive/folders/1-cri3JvbEtt6RfzwttKJe12qNEyggPFG?usp=drive_link)
- [Traditional Methods Outputs Drive Link](https://drive.google.com/drive/folders/1RlJ7N2PIF3-YJbeG6KX23Ccbp05By42H?usp=drive_link)

### Deep Learning Methods

#### 3. Convolutional Neural Networks (CNN)

- **Description:** Extracts spatial features from video frames to detect transitions.
- **Architecture:** Includes convolutional layers followed by fully connected layers.
![Screenshot 2024-12-12 222107](https://github.com/user-attachments/assets/141a27e9-3a7f-4e1d-84c9-48274d1f2279)

#### 4. recurrent Neural Networks (RNN)

- **Description:** Models the temporal dependencies in video sequences.
- **Architecture:** Utilizes LSTM or GRU cells for sequential frame analysis.
![Screenshot 2024-12-12 222211](https://github.com/user-attachments/assets/4e068e54-617f-4c01-9e0c-fac63e880a61)

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
![Screenshot 2024-12-12 222340](https://github.com/user-attachments/assets/090e1562-e0d5-480d-b6f1-8a90a0539fc9)

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
  
  1.shot_test.mp4
  
![Screenshot 2024-12-12 222541](https://github.com/user-attachments/assets/9102c429-0681-44ec-929b-891cbf62bf8f)
![Screenshot 2024-12-12 222549](https://github.com/user-attachments/assets/6ea12dbf-5788-4e65-8112-ca79c9533eba)
![Screenshot 2024-12-12 222733](https://github.com/user-attachments/assets/2980b70f-c39e-4e77-be1d-8abaf880a120)

   2.11.mp4
   
![Screenshot 2024-12-12 222755](https://github.com/user-attachments/assets/842a1b03-320d-403f-83f9-b536971329ae)

![Screenshot 2024-12-12 222803](https://github.com/user-attachments/assets/e1d447b1-6834-4b1d-a41b-41c8991591ed)
![Screenshot 2024-12-12 222813](https://github.com/user-attachments/assets/3fa46410-1deb-4a0c-8679-eabc361398cc)

#### 2. RNN-Based Results for `11.mp4`

- **Steps:**
  1. Use CNN features as input to the RNN to model temporal dependencies.
  2. Train the RNN to classify transitions and evaluate metrics.
  3. Display performance metrics including a graph showcasing **F1-Score** for `11.mp4`.
- **Visualization:**
  
  1.shot_test.mp4
![Screenshot 2024-12-12 223007](https://github.com/user-attachments/assets/4b9a6538-898b-4eaf-9cc0-f498350e0532)

![Screenshot 2024-12-12 223013](https://github.com/user-attachments/assets/ab949638-0f9e-4168-9467-2347bba3ed38)

  2.11.mp4

![Screenshot 2024-12-12 223026](https://github.com/user-attachments/assets/afbf1af3-3ba9-4667-bb72-ea0a6cb2150d)
![Screenshot 2024-12-12 223031](https://github.com/user-attachments/assets/94df2abf-5497-4f47-a8d5-6f61108a3d93)


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
git clone https://github.com/prudvi1916/Video_Shot_Boundary_Detection.git

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

## Conclusion and Future Work

### Conclusion

This project effectively demonstrated the application of both traditional and deep learning methods for video shot boundary detection. Traditional methods, such as histogram and pixel differences, provided efficient solutions for detecting abrupt transitions like cuts. On the other hand, deep learning methods (CNNs and RNNs) showcased their superiority in handling complex transitions and leveraging temporal dependencies for higher accuracy.

The results highlighted the strengths and limitations of each approach, paving the way for selecting suitable methods based on specific application requirements, such as computational efficiency or robustness to gradual transitions.

### Future Work

Future efforts can focus on the following areas:

1. **Hybrid Approaches:** Combining traditional and deep learning methods to achieve higher accuracy and efficiency, especially in scenarios with diverse transition types.
2. **Real-Time Processing:** Optimizing deep learning models for real-time applications by reducing computational overhead.
3. **Domain-Specific Applications:** Adapting the models for domain-specific tasks like sports analysis, surveillance, or movie editing.
4. **Enhanced Dataset Diversity:** Expanding the dataset to include more varied and challenging transitions, ensuring robustness across different video genres and resolutions.




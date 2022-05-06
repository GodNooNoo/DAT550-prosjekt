# DAT550-prosjekt
This projects aims to detect deepfakes. It compares the accuracy of different models.

# Requirements
 - Download and install Cmake from https://cmake.org/download/
 - Install requirements.txt

    pip install -r requirements.txt

# Step 1: Download the dataset
We use the dataset from https://www.kaggle.com/competitions/deepfake-detection-challenge/data.
The complete dataset available there is too large, so we use a subset of it (around 10GB).

# Step 2: Preprocess the dataset
Using face_extraction.py, we first split the videos into frames. Second, we extract the faces from the frames.

    python face_extraction.py <path-to-videos>


# Step 3: Train the model

# Step 4: Test the model
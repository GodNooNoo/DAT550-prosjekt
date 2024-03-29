import os
import sys

import cv2 as cv
import cv2
import face_recognition
import pandas as pd
from tqdm import tqdm

PATH_VIDS = ""
PATH_REAL = ""
PATH_FAKE = ""


def extract_faces_optimized():
    """
    This function extracts the faces from the frames in the videos and saves it to the real or fake folder.
    """
    METADATA = pd.read_json(os.path.join(PATH_VIDS, "metadata.json"))

    if not os.path.isdir(PATH_REAL):
        os.mkdir(PATH_REAL)
    if not os.path.isdir(PATH_FAKE):
        os.mkdir(PATH_FAKE)

    videos = [
        os.path.join(PATH_VIDS, vid)
        for vid in os.listdir(PATH_VIDS)
        if vid.endswith(".mp4")
    ]

    for video in tqdm(videos):
        if METADATA[os.path.basename(video)]["label"] == "REAL":
            folder = PATH_REAL
        else:
            folder = PATH_FAKE
        # Create a folder for each video
        folder = os.path.join(folder, os.path.basename(video))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # Extract the frames
        vid = cv.VideoCapture(video)
        frames = []
        while True:
            success, image = vid.read()
            if not success:
                break
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            frames.append(image)
        for idx, frame in enumerate(tqdm(frames)):
            # extract the faces
            faces = face_recognition.face_locations(frame)
            # For each face in the image
            for i, face in enumerate(faces):
                # Get the top, right, bottom, left
                top, right, bottom, left = face
                # Crop the face with padding of 30 pixels
                face = frame[top - 15 : bottom + 15, left - 15 : right + 15]
                if face.size == 0:
                    continue
                cv.imwrite(os.path.join(folder, f"frame{idx}face{i}.jpg"), face)


def remove_empty_folders(path):
    """
    This function removes empty folders.
    """
    folders = list(os.walk(path))[1:]
    for folder in folders:
        if not folder[2]:
            os.rmdir(folder[0])
            print("removed:", folder[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_extraction.py <path_to_videos_folder>")
        sys.exit(1)

    PATH_VIDS = sys.argv[1]
    PATH_REAL = os.path.join(PATH_VIDS, "real")
    PATH_FAKE = os.path.join(PATH_VIDS, "fake")

    extract_faces_optimized()
    remove_empty_folders(PATH_REAL)
    remove_empty_folders(PATH_FAKE)

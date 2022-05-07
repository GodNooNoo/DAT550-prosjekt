import sys
import cv2 as cv
import os
import face_recognition

PATH_VIDS = ''
PATH_FRAMES = ''
PATH_FACES = ''


def video_to_frames():
    """
    This function splits the videos into frames and saves the frames in the frames folder.

    Args:
        path_to_videos_folder (string): Path to the videos folder.
        path_to_frames_folder (string): Path to the frames folder.

    Returns:
        int: Count of videos converted to frames.
    """

    if not os.path.isdir(PATH_FRAMES):
        os.mkdir(PATH_FRAMES)

    videos = [os.path.join(PATH_VIDS, vid) for vid in os.listdir(
        PATH_VIDS) if vid.endswith('.mp4')]
    for video in videos:
        # Create a folder for each video
        frames_folder = os.path.join(
            PATH_FRAMES, os.path.basename(video)[:-4])
        if not os.path.isdir(frames_folder):
            os.mkdir(frames_folder)
        # Extract the frames
        vid = cv.VideoCapture(video)
        count = 0
        while True:
            count += 1
            success, image = vid.read()
            if not success:
                break
            cv.imwrite(os.path.join(frames_folder, f"frame{count}.jpg"), image)
    return len(videos)


def extract_faces():
    """
    This function extracts the faces from the frames in the frames folder and saves the faces in the faces folder.

    Args:
        path_to_frames_folder (string): Path to the frames folder.
        path_to_faces_folder (string): Path to the faces folder.
    """

    if not os.path.isdir(PATH_FRAMES):
        os.mkdir(PATH_FRAMES)

    # Get all the folders in the frames_folder
    folders = [os.path.join(PATH_FRAMES, f) for f in os.listdir(
        PATH_FRAMES) if os.path.isdir(os.path.join(PATH_FRAMES, f))]
    for folder in folders:
        # Create a folder for each video
        faces_folder = os.path.join(
            PATH_FRAMES, os.path.basename(folder))
        if not os.path.isdir(faces_folder):
            os.mkdir(faces_folder)
        # Get all the frames in the folder
        frames = [os.path.join(folder, f)
                  for f in os.listdir(folder) if f.endswith('.jpg')]

        for frame in frames:
            # Get the image
            img = face_recognition.load_image_file(frame)
            # Get the face locations
            face_locations = face_recognition.face_locations(img)
            # For each face in the image
            face_count = 0
            for face_location in face_locations:
                face_count += 1
                # Get the top, right, bottom, left
                top, right, bottom, left = face_location
                # Crop the face
                face = img[top:bottom, left:right]
                # Save the face
                cv.imwrite(os.path.join(faces_folder, os.path.basename(
                    frame)[:-4] + f"_face{face_count}.jpg"), face)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python face_extraction.py <path_to_videos_folder>")
        sys.exit(1)

    PATH_VIDS = sys.argv[1]
    PATH_FRAMES = os.path.join(PATH_VIDS, "frames")
    PATH_FACES = os.path.join(PATH_VIDS, "faces")

    # Extract the frames
    num_vids = video_to_frames()
    print("Extracted frames from {} videos".format(num_vids))

    # Extract the faces
    extract_faces()
    print("Extracted faces")

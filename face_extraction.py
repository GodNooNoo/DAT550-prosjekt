import cv2 as cv
import os
import face_recognition

def video_to_frames(path_to_videos_folder, path_to_frames_folder):
    # Get all the video files in the er
    videos = [os.path.join(path_to_videos_folder, vid) for vid in os.listdir(path_to_videos_folder) if vid.endswith('.mp4')]
    for video in videos:
        # Create a folder for each video
        frames_folder = os.path.join(path_to_frames_folder, os.path.basename(video)[:-4])
        if not os.path.isdir(frames_folder):
            os.mkdir(frames_folder)
        # Extract the frames
        vid = cv.VideoCapture(path_to_videos_folder)
        count = 0
        while True:
            count += 1
            success, image = vid.read()
            if not success:
                break
            cv.imwrite(os.path.join(frames_folder, "frame{:d}.jpg".format(count)), image)
    return count

def extract_faces(path_to_frames_folder, path_to_faces_folder):
    # Get all the folders in the frames_folder
    folders = [os.path.join(path_to_frames_folder, f) for f in os.listdir(path_to_frames_folder) if os.path.isdir(os.path.join(path_to_frames_folder, f))]
    for folder in folders:
        # Create a folder for each video
        faces_folder = os.path.join(path_to_faces_folder, os.path.basename(folder))
        if not os.path.isdir(faces_folder):
            os.mkdir(faces_folder)
        # Get all the frames in the folder
        frames = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        for frame in frames:
            # Get the image
            img = face_recognition.load_image_file(frame)
            # Get the face locations
            face_locations = face_recognition.face_locations(img)
            # For each face in the image
            for face_location in face_locations:
                # Get the top, right, bottom, left
                top, right, bottom, left = face_location
                # Crop the face
                face = img[top:bottom, left:right]
                # Save the face
                cv.imwrite(os.path.join(faces_folder, os.path.basename(frame)), face)


if __name__ == '__main__':
    path_to_videos_folder = './videos'
    path_to_frames_folder = './frames'
    path_to_faces_folder = './faces'
    # Extract the frames
    num_frames = video_to_frames(path_to_videos_folder, path_to_frames_folder)
    print("Extracted {} frames".format(num_frames))
    # Extract the faces
    extract_faces(path_to_frames_folder, path_to_faces_folder)
    print("Extracted faces")


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
        vid = cv.VideoCapture(video)
        count = 0
        while True:
            success, image = vid.read()
            if not success:
                break
            cv.imwrite(os.path.join(frames_folder, "frame{:d}.jpg".format(count)), image)
    return len(videos)

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
        face_count = 0
        for frame in frames:
            # Get the image
            img = face_recognition.load_image_file(frame)
            # Get the face locations
            face_locations = face_recognition.face_locations(img)
            # For each face in the image
            for face_location in face_locations:
                face_count += 1
                # Get the top, right, bottom, left
                top, right, bottom, left = face_location
                # Crop the face
                face = img[top:bottom, left:right]
                # Save the face
                cv.imwrite(os.path.join(faces_folder, "face{:d}.jpg".format(face_count)), face)


if __name__ == '__main__':
    path_to_videos_folder = './sample_set'

    path_to_frames_folder = './sample_frames'
    if not os.path.isdir(path_to_frames_folder):
        os.mkdir(path_to_frames_folder)
    path_to_faces_folder = './sample_faces'
    if not os.path.isdir(path_to_faces_folder):
        os.mkdir(path_to_faces_folder)
    # Extract the frames
    num_vids = video_to_frames(path_to_videos_folder, path_to_frames_folder)
    print("Extracted frames from {} videos".format(num_vids))
    # Extract the faces
    extract_faces(path_to_frames_folder, path_to_faces_folder)
    print("Extracted faces")


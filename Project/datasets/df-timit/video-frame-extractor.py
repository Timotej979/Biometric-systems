import os
import argparse
import subprocess
import cv2
import numpy as np
import dlib
from random import random

class VideoFrameExtractor:
    def __init__(self, video_root, output_root):
        # Data loader parameters
        self.video_root = video_root if video_root is not None else os.path.join(os.getcwd(), 'higher_quality')
        self.output_root = output_root if output_root is not None else os.path.join(os.getcwd(), 'higher_quality_images')
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def extract_frames_and_align_faces(self):
        for root, dirs, files in os.walk(self.video_root):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                    video_path = os.path.abspath(os.path.join(root, file))

                    # Create a directory for each video's frames
                    output_dir = os.path.join(self.output_root, f"{os.path.splitext(file)[0]}-img")
                    os.makedirs(output_dir, exist_ok=True)

                    # Use ffprobe to get the total number of frames
                    ffprobe_command = f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "{video_path}"'
                    num_frames = int(subprocess.check_output(ffprobe_command, shell=True).decode("utf-8").strip())

                    for frame_number in range(1, num_frames + 1):
                        # Use ffmpeg to extract a specific frame
                        ffmpeg_command = f'ffmpeg -i "{video_path}" -vf "select=\'eq(n,{frame_number})\'" -vframes 1 "{output_dir}/frame.png"'
                        subprocess.run(ffmpeg_command, shell=True)

                        # Read the extracted frame
                        frame_path = os.path.join(output_dir, "frame.png")
                        frame = cv2.imread(frame_path)

                        # Detect faces in the frame
                        faces = self.detector(frame)

                        if len(faces) > 0:
                            # Assume there is only one face in the frame
                            face = faces[0]

                            # Get facial landmarks
                            landmarks = self.predictor(frame, face)

                            # Calculate the midpoint between the eyes
                            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
                            mid_point = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                            # Calculate the angle for rotation
                            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])) - 90

                            # Rotate the frame to align the eyes horizontally
                            rotation_matrix = cv2.getRotationMatrix2D(mid_point, angle, 1)
                            aligned_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))

                            # Save the aligned frame
                            aligned_frame_path = os.path.join(output_dir, f"aligned_frame_{frame_number}.png")
                            cv2.imwrite(aligned_frame_path, aligned_frame)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    # Extract frames and align faces from videos
    video_frame_extractor = VideoFrameExtractor(args.video_root, args.output_root)
    video_frame_extractor.extract_frames_and_align_faces()

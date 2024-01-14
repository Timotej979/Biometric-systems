import os
import time
import argparse
import subprocess
import cv2
import numpy as np
import dlib

class VideoFrameExtractor:
    def __init__(self, video_root, output_root):
        # Data loader parameters
        self.video_root = video_root if video_root is not None else os.path.join(os.getcwd(), 'higher_quality')
        self.output_root = output_root if output_root is not None else os.path.join(os.getcwd(), 'higher_quality_images')
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictors/shape_predictor_68_face_landmarks.dat")
        # Set the number of frames to extract from each video
        self.num_frames = 1

    def extract_frames_and_align_faces(self):
        for root, dirs, files in os.walk(self.video_root):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                    video_path = os.path.abspath(os.path.join(root, file))

                    # Create a directory for each video's frames
                    output_dir = os.path.join(self.output_root, os.path.basename(root))
                    os.makedirs(output_dir, exist_ok=True)

                    # Use ffprobe to get the total number of frames
                    ffprobe_command = f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "{video_path}"'
                    num_frames = int(subprocess.check_output(ffprobe_command, shell=True).decode("utf-8").strip())

                    # Select frames to extract
                    random_frames = np.random.randint(1, num_frames, size=self.num_frames)

                    for frame_number in random_frames:
                        # Use ffmpeg to extract a specific frame
                        ffmpeg_command = f'ffmpeg -y -i "{video_path}" -vf "select=\'eq(n,{frame_number})\'" -vframes 1 "{output_dir}/frame.png"'
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

                            # Calculate the angle for rotation
                            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
                            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])) - 90

                            # Rotate the frame to align the eyes horizontally
                            rotated_frame = cv2.warpAffine(frame, cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), angle, 1),
                                                          (frame.shape[1], frame.shape[0]))

                            # Rotate the frame by 90 degrees conunter-clockwise
                            rotated_frame = cv2.rotate(rotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            # Crop the rotated frame to remove black borders
                            non_zero_pixels = cv2.findNonZero(cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY))
                            x, y, w, h = cv2.boundingRect(non_zero_pixels)
                            cropped_frame = rotated_frame[y:y + h, x:x + w]

                            # Save the aligned and cropped frame
                            aligned_frame_path = os.path.join(output_dir, f"{frame_number}.png")
                            cv2.imwrite(aligned_frame_path, cropped_frame)
                        
                        # Remove the extracted frame
                        os.remove(frame_path)
                        time.sleep(0.2)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=False, help="The root directory of the videos")
    parser.add_argument("--output_root", type=str, required=False, help="The root directory of the output images")
    args = parser.parse_args()

    # Extract frames, align faces, and crop images from videos
    video_frame_extractor = VideoFrameExtractor(args.video_root, args.output_root)
    video_frame_extractor.extract_frames_and_align_faces()

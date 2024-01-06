#!/usr/bin/env python3
import os, argparse
import cv2


class Viola_Jones_Detector():
    """
        Viola-Jones face detector class using OpenCV
    """

    def __init__(self, args):
        # Initialize input arguments into self.values
        #   Detector arguments:
        self.cascade = args.cascade
        self.scaleFactor = args.scaleFactor
        self.minNeighbors = args.minNeighbors
        self.minSize = (args.minSizeX, args.minSizeY)
        self.maxSize = (args.maxSizeX, args.maxSizeY)
        #   Execution arguments:
        self.display = args.display
        self.displayTime = args.displayTime

    def initialize_classifier(self):
        # Initialize the classifier
        self.classifier = cv2.CascadeClassifier("./cascades/" + self.cascade)

    def get_faces(self, image_path, **kwargs):    
        # Get detected faces from the classifier
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, _, weights = self.classifier.detectMultiScale3(gray, **kwargs, outputRejectLevels=True)

        results = [(x, y, w, h, score) for (x, y, w, h), score in zip(detections, weights)]

        if self.display:
            self.display_image(img, results)

        return results    

    def display_image(self, img, results):
        # Display the image with the detection boxes and scores
        for x, y, w, h, score in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(int(self.displayTime * 1000))
        cv2.destroyAllWindows()

    def output_detections(self, file, detections, image_path):
        # Write the detections to the output file
        file.write(image_path.rstrip(".jpg") + '\n')
        file.write(str(len(detections)) + '\n')
        for x, y, w, h, score in detections:
            file.write(f"{x} {y} {w} {h} {score:.2f}\n")

    def process_images(self, image_file, output_file, images_dir):
        # Construct the detector settings
        settings = {
            "scaleFactor": self.scaleFactor,
            "minNeighbors": self.minNeighbors,
            "minSize": self.minSize,
            "maxSize": self.maxSize
        }

        with open(output_file, "w") as out:
            with open(image_file, "r") as file:

                # Read the lines from the image file
                lines = file.readlines()
                
                # Process every other line
                idx = 0
                while idx < len(lines):
                    # Get the image path and name
                    image_path = lines[idx].strip()
                    
                    # Add the .jpg extension if it is missing
                    if not image_path.endswith(".jpg"):
                        image_path += ".jpg"
                    # Get the full image path
                    full_image_path = os.path.join(images_dir, image_path)
                    
                    # Check if the next line is a number
                    if idx + 1 < len(lines) and lines[idx+1].strip().isnumeric():
                        # Get the detections
                        detections = self.get_faces(full_image_path, **settings)
                        self.output_detections(out, detections, image_path)
                        # Skip the next line
                        idx += 1
                    # Skip the next line
                    idx += 1




if __name__ == "__main__":

    # Get command line arguments
    parser = argparse.ArgumentParser(description='Viola-Jones face detector')

    # Input arguments: 
    #   Detector arguments:
    #       - scale factor (default 1.05)
    #       - min neighbors (default 6)
    #       - min size (default 20x20)
    #       - max size (default 250x250)
    #   Execution arguments:
    #       - display (default False)
    #       - display time (default 0.5s)
    #       - output (default output.txt)
    #       - foldNum (default 1)

    # Detector arguments
    parser.add_argument('--cascade', 
                        type=str,
                        default='haarcascade_frontalface_default.xml',
                        help='HAAR cascade file, you can change the cascade file to try different detectors. Possible options for face detections include the following:\n - haarcascade_frontalface_alt.xml\n - haarcascade_frontalface_alt2.xml\n - haarcascade_frontalface_alt_tree.xml\n - haarcascade_profileface.xml\n - ')
    parser.add_argument('--scaleFactor',
                        type=float,
                        default=1.05,
                        help='scale factor used for the detector')
    parser.add_argument('--minNeighbors',
                        type=int,
                        default=6,
                        help='min neighbors used for the detector')
    parser.add_argument('--minSizeX',
                        type=int,
                        default=20,
                        help='min size of the sliding window in x direction')
    parser.add_argument('--minSizeY',
                        type=int, default=20,
                        help='min size of the sliding window in y direction')
    parser.add_argument('--maxSizeX',
                        type=int,
                        default=250,
                        help='max size of the sliding window in x direction')
    parser.add_argument('--maxSizeY',
                        type=int,
                        default=250,
                        help='max size of the sliding window in y direction')
    
    # Execution arguments
    parser.add_argument('--display',
                        type=bool,
                        default=False,
                        help='displays all images with detections that are above the threshold')
    parser.add_argument('--displayTime',
                        type=float,
                        default=0.5,
                        help='time in s to display each image, only used if display is set to True')
    parser.add_argument('--foldNum',
                        type=int,
                        default=1,
                        help='fold number to process')

    # Parse the arguments
    args = parser.parse_args()

    # Get the current directory. Adjust this if your base directory is different.
    current_dir = os.getcwd()

    # Set folds, images and output directory
    folds_dir = os.path.join(current_dir, "FDDB-folds")
    images_dir = os.path.join(current_dir, "originalPics")
    output_dir = os.path.join(current_dir, "detections")

    # Get the images to process and set the output files
    image_file = os.path.join(folds_dir, f"FDDB-fold-{args.foldNum:02}-ellipseList.txt")
    output_file = os.path.join(output_dir, f"fold-{args.foldNum:02}-out.txt")

    # Create the detector with the given arguments
    detector = Viola_Jones_Detector(args)

    # Initialize the classifier
    detector.initialize_classifier()

    # Process the images
    detector.process_images(image_file, output_file, images_dir)




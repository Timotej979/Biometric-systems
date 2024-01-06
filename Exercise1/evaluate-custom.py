import os, sys
import matplotlib.pyplot as plt


class Viola_Jones_Evaluation():

    def __init__(self, args):
        # Initialize input arguments into self.values
        self.foldsDir = args["foldsDir"]
        self.outputDir = args["outputDir"]
        self.imageFile = args["imageFile"]
        self.outputFile = args["outputFile"]
        self.confidenceScores = args["confidenceScores"]
        # Initialize the expected detections and the detected faces
        self.expectedDetections = {}
        self.detectedFaces = {}
        # Initialize the evaluation metrics
        self.missedDetections = {}
        self.exactDetections = {}
        self.falseDetections = {}

    def evaluate_custom(self):
        # Open the image file and read the expected detections
        with open(self.imageFile, "r") as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                image_name = lines[i].strip()
                i += 1
                detection_value = int(lines[i].strip())
                i += 1
                # Create the expected detections dictionary step by step
                self.expectedDetections[image_name] = detection_value

        # Open the output file and read the detected faces
        with open(self.outputFile, "r") as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                image_name = lines[i].strip()
                i += 1
                num_detections = int(lines[i].strip())
                i += 1
                detections = []
                for _ in range(num_detections):
                    detection_line = lines[i].strip().split()
                    i += 1
                    # Extract the relevant information from the detection line
                    x, y, w, h, confidence = map(float, detection_line[:])
                    detections.append({"x": x, "y": y, "w": w, "h": h, "confidence": confidence})
                self.detectedFaces[image_name + "-detections"] = detections
                self.detectedFaces[image_name] = num_detections

        # Compare the expected detections with the detected faces
        for image_name in self.expectedDetections.keys():
            # Check if the image name is in the detected faces
            if image_name in self.detectedFaces:

                # Check wether the number of expectedDetections is greater or equal or less than the detected faces
                if self.expectedDetections[image_name] > self.detectedFaces[image_name]:
                    # If the number of expectedDetections is greater than the detected faces, then the difference is the number of missedDetections
                    self.missedDetections[image_name] = self.expectedDetections[image_name] - self.detectedFaces[image_name]

                elif self.expectedDetections[image_name] < self.detectedFaces[image_name]:
                    self.falseDetections[image_name] = self.detectedFaces[image_name] - self.expectedDetections[image_name]

                else:
                    self.exactDetections[image_name] = self.expectedDetections[image_name]
            else:
                print(f"Detection mismatch for {image_name}")
                sys.exit(1)

    def graph_evaluated(self):
        # Check if the confidence scores are not empty
        if self.confidenceScores:
            plt.figure(figsize=(12, 10))  # Increase the height to display the graphs vertically
            plt.subplot(2, 1, 1)  # Create two vertical subplots

            # Initialize lists to store thresholds, corresponding counts, and the expected number
            thresholds = []
            actual_counts = []

            # Iterate over different confidence score thresholds
            for threshold in range(70, 100, 2):  # Adjust the range and step size as needed
                threshold /= 100  # Convert to a percentage
                count = sum(1 for score in self.confidenceScores if score >= threshold)
                thresholds.append(threshold)
                actual_counts.append(count)

            # Plot the actual counts against thresholds
            plt.plot(thresholds, actual_counts, label='Actual')
            plt.xlabel('Confidence Score Threshold')
            plt.ylabel('Number of Detections')
            plt.title('Actual Detections vs Confidence Score Threshold')
            plt.grid(axis='y')
            plt.legend()

            plt.subplot(2, 1, 2)  # Second subplot

            # Create the bar graph for missed and false detections per image
            categories = list(self.expectedDetections.keys())  # Use image names as categories
            missed_counts = [self.missedDetections.get(image_name, 0) for image_name in categories]
            false_counts = [self.falseDetections.get(image_name, 0) for image_name in categories]

            x = range(len(categories))
            width = 0.35

            # Adjust the width to create space between categories
            plt.bar([i + width for i in x], missed_counts, width, align='edge', label='Missed Detections', color='r')
            plt.bar([i + 2 * width for i in x], false_counts, width, align='edge', label='False Detections', color='b')

            plt.xlabel('Image Numbers')
            plt.ylabel('Number of Detections')
            plt.title('Detection Categories per Image')
            plt.xticks([i + width for i in x], range(1, len(categories) + 1))
            plt.grid(axis='y')
            plt.legend()

            plt.tight_layout()  # Adjust subplot spacing

            plt.show()



if __name__ == '__main__':

    # Get current directory
    current_dir = os.getcwd()

    # Set folds, images and output directory
    folds_dir = os.path.join(current_dir, "CUSTOM-folds")
    output_dir = os.path.join(current_dir, "results-custom")

    # Get the images to process and set the output files
    image_file = os.path.join(folds_dir, "FLICKR-fold.txt")
    output_file = os.path.join(output_dir, "flickr-fold-out.txt")

    # Set the confidence scores and sort them in ascending order
    confidence_scores = [0.85, 0.92, 0.75, 0.88, 0.67, 0.95, 0.72, 0.90, 0.79]
    confidence_scores.sort()

    # Create args
    args = {
        "foldsDir": folds_dir,
        "outputDir": output_dir,
        "imageFile": image_file,
        "outputFile": output_file,
        "confidenceScores": confidence_scores
    }

    # Create the detector with the given arguments
    detector = Viola_Jones_Evaluation(args)

    # Initialize the classifier
    detector.evaluate_custom()

    # Process the images
    detector.graph_evaluated()
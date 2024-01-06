import os, subprocess, multiprocessing
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class OpenBR():
    """
    Class to run OpenBR comparison on LFW dataset.
    Methods available:
        - run_openbr_comparison: Compare two faces using OpenBR.
        - process_pairs: Process the pairs file and save the results.
        - generate_roc_curve: Generate ROC curve.
    """

    def __init__(self, br_path, lfw_dataset_path, pairs_file_path, output_file_path, display_log):
        """Initialize OpenBR object."""
        # Initialize class variables
        self.br_path = br_path
        self.lfw_dataset_path = lfw_dataset_path
        self.pairs_file_path = pairs_file_path
        self.output_file_path = output_file_path
        self.display_log = display_log
        # Initialize local class variables
        self.img1_path = None
        self.img2_path = None
        self.labels = []
        self.scores = []

    ################################################################################################################################################################
    # Standard OpenBR comparison
    def process_pairs_standard(self):
        """Process the pairs file and save the results."""
        with open(self.pairs_file_path, 'r') as file, open(self.output_file_path + '/similarity_scores.txt', 'w') as out_file:
            next(file)  # Skip the first line (number of pairs)
            for line in file:
                tokens = line.strip().split()
                if len(tokens) == 3:  # Same person
                    person, img_num1, img_num2 = tokens
                    self.img1_path = os.path.join(self.lfw_dataset_path, person, "{}_{}.jpg".format(person, img_num1.zfill(4)))
                    self.img2_path = os.path.join(self.lfw_dataset_path, person, "{}_{}.jpg".format(person, img_num2.zfill(4)))
                else:  # Different people
                    person1, img_num1, person2, img_num2 = tokens
                    self.img1_path = os.path.join(self.lfw_dataset_path, person1, "{}_{}.jpg".format(person1, img_num1.zfill(4)))
                    self.img2_path = os.path.join(self.lfw_dataset_path, person2, "{}_{}.jpg".format(person2, img_num2.zfill(4)))
                
                if self.display_log:
                    print("--------------------------------------------------")
                    print("Comparing {} and {}".format(self.img1_path, self.img2_path))
                # Run OpenBR comparison
                similarity_score = self.run_openbr_comparison_standard()
                out_file.write("{}: {}\n".format(line.strip(), similarity_score.decode('utf-8')))

                if self.display_log:
                    print("Similarity score: {}".format(similarity_score.decode('utf-8')))
                    print("--------------------------------------------------")

    def run_openbr_comparison_standard(self):
        """Compare two faces using OpenBR."""
        command = "br -algorithm FaceRecognition -compare {} {}".format(self.img1_path, self.img2_path)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        return output.strip()

    ################################################################################################################################################################
    # Parallel OpenBR comparison
    def process_pairs_parallel(self):
        # Create a multiprocessing Queue to store the results
        output_queue = multiprocessing.Queue()

        with open(self.pairs_file_path, 'r') as file:
            next(file)  # Skip the first line (number of pairs)

            # Create a list to hold the processes
            processes = []

            for line in file:
                tokens = line.strip().split()

                if len(tokens) == 3:  # Same person
                    person, img_num1, img_num2 = tokens
                    img1_path = os.path.join(self.lfw_dataset_path, person, "{}_{}.jpg".format(person, img_num1.zfill(4)))
                    img2_path = os.path.join(self.lfw_dataset_path, person, "{}_{}.jpg".format(person, img_num2.zfill(4)))
                else:  # Different people
                    person1, img_num1, person2, img_num2 = tokens
                    img1_path = os.path.join(self.lfw_dataset_path, person1, "{}_{}.jpg".format(person1, img_num1.zfill(4)))
                    img2_path = os.path.join(self.lfw_dataset_path, person2, "{}_{}.jpg".format(person2, img_num2.zfill(4)))

                if self.display_log:
                    print("--------------------------------------------------")
                    print("Comparing {} and {}".format(img1_path, img2_path))

                # Start a new process for each comparison
                process = multiprocessing.Process(target=self.run_openbr_comparison_parallel, args=(img1_path, img2_path, output_queue, line))
                processes.append(process)
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()

            # Get results from the queue and write to the output file
            with open(self.output_file_path + '/similarity_scores.txt', 'w') as out_file:
                while not output_queue.empty():
                    similarity_score, line = output_queue.get()
                    out_file.write("{}: {}\n".format(line.strip(), similarity_score.decode('utf-8')))

                    if self.display_log:
                        print("Similarity score: {}".format(similarity_score.decode('utf-8')))
                        print("--------------------------------------------------")

    def run_openbr_comparison_parallel(self, img1_path, img2_path, output_queue, line):
        """Compare two faces using OpenBR."""
        command = "br -algorithm FaceRecognition -compare {} {}".format(img1_path, img2_path)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        output_queue.put((output.strip(), line))

    ################################################################################################################################################################
    # ROC curve
    def generate_roc_curve(self): 
        """Generate ROC curve."""
        with open(self.output_file_path + '/similarity_scores.txt', 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            self.scores.append(parts[4] if len(parts) == 5 else parts[3])
            # Assign 1 for genuine pairs, 0 for impostor pairs
            self.labels.append(0 if len(parts) == 5 else 1)

        # Calculate ROC curve
        self.scores = np.where(np.isneginf(np.array(self.scores).astype(float)), -4e38, np.array(self.scores).astype(float))
        fpr, tpr, thresholds = roc_curve(self.labels, self.scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Calculate Equal Error Rate (EER)
        eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
        eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

        # Plot EER and the value of its cordinates
        plt.plot(eer, 1 - eer, marker='o', markersize=6, color="red")
        plt.annotate("(%.4f, %.4f)" % (eer, 1 - eer), xy=(eer, 1 - eer), xytext=(eer - 0.2, 1 - eer + 0.1), color="red")

        # Print EER
        print("Equal Error Rate (EER): %.4f" % eer)

        # Save ROC curve
        plt.savefig(self.output_file_path + '/roc_curve.png')


if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description='Run OpenBR comparison on LFW dataset.')
    # Add file path arguments
    parser.add_argument('--br_path', type=str, default='br', help='Path to the br executable.')
    parser.add_argument('--lfw_dataset_path', type=str, default='lfw-dataset', help='Path to the LFW dataset.')
    parser.add_argument('--pairs_file_path', type=str, default='lfwa-pairs/pairsDevTest.txt', help='Path to the pairsDevTest.txt file.')
    parser.add_argument('--output_file_path', type=str, default='output-openbr', help='Path to the output file.')
    # Add optional arguments
    parser.add_argument('--display_log', action='store_true', help='Display OpenBR progress.')
    parser.add_argument('--parallel', action='store_true', help='Run OpenBR in parallel.')
    # Parse arguments
    args = parser.parse_args()

    # Create OpenBR object
    openbr = OpenBR(args.br_path, args.lfw_dataset_path, args.pairs_file_path, args.output_file_path, args.display_log)

    # Check if parallel
    if args.parallel:
        openbr.process_pairs_parallel()
    else:
        openbr.process_pairs_standard()

    # Generate ROC curve
    openbr.generate_roc_curve()
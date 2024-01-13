import os, sys, csv, time, datetime, argparse, subprocess
import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class SBIControler:
    def __init__(self, test_root, weights_bool, worker_num):
        # Data loader parameters
        self.weights_path = os.path.join('/app/detector/weights', 'FFraw.tar' if weights_bool else 'FFc23.tar')
        self.test_root = test_root if test_root is not None else '/datasets/face-forensics/test'
        self.output_dir = '/output/'
        # Initialize the number of workers
        self.worker_num = worker_num

    def get_image_paths(self):
        image_paths = []
        labels = []

        # Traverse the directory recursively
        for root, dirs, files in os.walk(self.test_root):
            for file in files:
                # Check if the file has a picture extension
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    # Construct the absolute path and add it to the list
                    image_path = os.path.abspath(os.path.join(root, file))
                    image_paths.append(image_path)

                    # Check if the absolute path contains word 'deepfake'
                    if 'deepfake' in image_path:
                        labels.append(1)
                    elif 'real' in image_path:
                        labels.append(0)
                    else:
                        print("Error: The image path {} does not contain word deepfake or real".format(image_path))
                        sys.exit(1)
    
        return image_paths[1:30], labels[1:30]

    def run_inference_single(self, image_path):
        # Construct the command and execute it
        command = 'python3 /app/detector/src/inference/inference_image.py -w {} -i {}'.format(self.weights_path, image_path)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Get the output and parse it to get the detection score
        output, error = process.communicate()
        output = output.decode("utf-8")
        output = output.split('\n')
        print(output)
        output = output[1].strip('fakeness: ')
        print(output)
        return float(output)

    def run_inference(self):
        # Get the image paths
        image_paths, labels = self.get_image_paths()

        print("Begin testing...")
        start_time = time.time()

        # Create a directory to save the output
        dataset_name = self.test_root.split(os.sep)[-1]
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.weights_path.split(os.sep)[-1].split('.')[0]
        output_time_dir = f'output_test_{current_datetime}_{dataset_name}_{model_name}'
        os.makedirs(self.output_dir + output_time_dir, exist_ok=True)

        out_string = f"Testing. Using {self.test_root} for testing and using {self.weights_path} for model weights."
        log_filename = f'log_{dataset_name}_{current_datetime}.txt'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        # Run inference on each image using multiple processes and get the detection scores
        scores = process_map(self.run_inference_single, image_paths, max_workers=self.worker_num, chunksize=1)

        print(labels)
        print(scores)

        # Print total testing time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total testing time: {elapsed_time}")

        # Get ROC curve and AUC
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        roc_auc = metrics.auc(fpr, tpr)
        # Save the ROC curve to output directory
        plt.figure(figsize=(15, 10))
        plt.plot(fpr, tpr, c="dodgerblue")
        plt.title("ROC curve", fontsize=18)
        plt.xlabel("FPR", fontsize=18)
        plt.ylabel("TPR", fontsize=18)
        filename = f'ROC_{dataset_name}_{current_datetime}.svg'
        plt.savefig( os.path.join(self.output_dir + output_time_dir, filename))
        print(f"AUC: {roc_auc}")

        ## Calculate EER, treshold@EER, APCER@EER and BPCER@EER
        # APCER - proportion of deepfakes incorrectly classified as real images - false negative rate
        # BPCER - proportion of real images incorrectly classified as deepfakes - false positive rate
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        fpr_at_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        fnr_at_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        # Store the labels and scores in a csv file
        with open(os.path.join(self.output_dir + output_time_dir, 'labels.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(labels)
        with open(os.path.join(self.output_dir + output_time_dir, 'scores.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(scores)

        # Log the output
        out_string = f'EER: {eer:.4f}, EER treshold: {eer_threshold:.4f}, BPCER(FPR)@EER: {fpr_at_eer:.4f}, APCER(FNR)@EER: {fnr_at_eer:.4f}'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        # Calculate percission, recall and F1 score
        precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, scores, pos_label=0)
        pr_auc = metrics.auc(recall, precision)

        # Calculate F1 at threshold
        f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
        f1_max_idx = np.argmax(f1_scores)
        f1_max_threshold = thresholds_pr[f1_max_idx]

        # Log the output
        out_string = f'PR AUC: {pr_auc:.4f}, F1 score: {f1_scores[f1_max_idx]:.4f}, F1 threshold: {f1_max_threshold:.4f}'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        # Plot Precision-Recall curve
        plt.figure(figsize=(15, 10))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AUC = {:.2f})'.format(pr_auc))
        plt.scatter(recall[f1_max_idx], precision[f1_max_idx], marker='o', color='red', label='F1 point ({:.4f}, {:.4f})'.format(recall[f1_max_idx], precision[f1_max_idx]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        filename = f'PR_{dataset_name}_{current_datetime}.svg'
        plt.savefig(os.path.join(self.output_dir + output_time_dir, filename))

        # Generate classification report
        y_pred = (np.array(scores) > f1_max_threshold).astype(int)
        target_names = ['Deepfake', 'Real']
        classification_rep = metrics.classification_report(labels, y_pred, target_names=target_names)
        print(classification_rep)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(classification_rep)
            file.write(os.linesep)

        return

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a test set.')
    parser.add_argument('--test_root', type=str, default=None, help='The root directory of the test set.')
    parser.add_argument('--weights_bool', type=bool, default=True, help='The weights to use for inference.')
    parser.add_argument('--worker_num', type=int, default=1, help='The number of workers to use for inference.')
    args = parser.parse_args()

    # Create the SBIControler object
    sbi_controler = SBIControler(args.test_root, args.weights_bool, args.worker_num)

    # Run inference
    sbi_controler.run_inference()
        
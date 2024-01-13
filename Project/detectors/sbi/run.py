import os, sys, subprocess, argparse
import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
from sklearn import metrics


class SBIControler:
    def __init__(self, test_root, weights_bool, worker_num):
        if weights_bool:
            self.weights_path = '/app/detector/weights/FFraw.tar'
        else:
            self.weights_path = '/app/detector/weights/FFc23.tar'

        if test_root is None:
            self.test_root = '/datasets/face-forensics/test'
        else:
            self.test_root = test_root

        self.worker_num = worker_num

    def get_image_paths(self):
        image_paths = []

        # Traverse the directory recursively
        for root, dirs, files in os.walk(self.test_root):
            for file in files:
                # Check if the file has a picture extension
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    # Construct the absolute path and add it to the list
                    image_path = os.path.abspath(os.path.join(root, file))
                    image_paths.append(image_path)
        
        return image_paths

    def run_inference_single(self, image_path):
        # This function is used for process_map
        command = 'python3 /app/detector/src/inference/inference_image.py -w {} -i {}'.format(self.weights_path, image_path)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        output = output.decode("utf-8")
        output = output.split('\n')
        output = int(output[1].strip('fakeness: '))
        return output

    def run_inference(self):
        print("Running inference on test set {} with weights {}...".format(self.test_root, self.weights_path))

        # Get the image paths
        image_paths = self.get_image_paths()

        # Use process_map with tqdm to parallelize and show progress
        output_list = process_map(self.run_inference_single, image_paths, max_workers=self.worker_num)

        # Now you have a list of outputs for each image if needed
        print("All processes finished.")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a test set.')
    parser.add_argument('--test_root', type=str, default=None, help='The root directory of the test set.')
    parser.add_argument('--weights_bool', type=bool, default=True, help='The weights to use for inference.')
    parser.add_argument('--woker_num', type=int, default=4, help='The number of workers to use for inference.')
    args = parser.parse_args()

    # Create the SBIControler object
    sbi_controler = SBIControler(args.test_root, args.weights_bool, args.woker_num)

    # Run inference
    sbi_controler.run_inference()
        
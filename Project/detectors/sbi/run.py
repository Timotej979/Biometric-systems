import os, sys, subprocess, argparse
import matplotlib
from sklearn import metrics


class SBIControler:
    def __init__(self, test_root, weights_bool):
        if self.weights_bool:
            self.weights_path = 'weights/FFraw.tar'
        else:
            self.weights_path = 'weights/FFc23.tar'

        if test_root is None:
            self.test_root = '/datasets/face-forensics/test'
        else:
            self.test_root = test_root

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

    def run_inference(self):
        print('Running inference on test set {} with weights {}...', self.test_root, self.weights_path)

        # Get the image paths
        image_paths = self.get_image_paths()

        # Run inference on each image
        output_list = []
        for image_path in image_paths:
            print('Running inference on image {}...', image_path)
            # Run the command
            subprocess.run(['CUDA_VISIBLE_DEVICES=* python3 src/inference/inference_image.py -w {} -i {}'.format(self.weights_path, image_path)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            
            # Check if error occurred
            if error:
                print(error)
                sys.exit(1)
            else:
                print(output)
                output_list.append(output)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a test set.')
    parser.add_argument('--test_root', type=str, default=None, help='The root directory of the test set.')
    parser.add_argument('--weights_bool', type=bool, default=True, help='The weights to use for inference.')
    args = parser.parse_args()

    # Create the SBIControler object
    sbi_controler = SBIControler(args.test_root, args.weights_bool)

    # Print the image paths
    image_paths = sbi_controler.get_image_paths()
    print(image_paths)
        
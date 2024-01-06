import os, argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, pairwise


class GridSVC:
    """
    GridSVC class for training and testing a Support Vector Classifier (SVC) with Grid Search.
    Methods available:
        - read_pairs: Read pairs from file.
        - normalize_features: Normalize features.
        - crop_image: Crop the image to the bounding box.
        - detect_and_crop_face_haar: Detect and crop face using Haar Cascade.
        - process_pairs: Process pairs sequentially.
        - train_model: Train the model.
        - test_model: Test the model.
        - plot_roc_curve: Plot ROC curve.
    """

    def __init__(self, dataset_dir, train_pairs_file, test_pairs_file, output_file_path, face_detection_type, substract_hog_features, svc_kernel, display_log):
        # Initialize paths
        self.dataset_dir = dataset_dir
        self.train_pairs_file = train_pairs_file
        self.test_pairs_file = test_pairs_file
        self.output_file_path = output_file_path
        # Initialize parameters
        self.display_log = display_log
        self.face_detection_type = face_detection_type
        self.substract_hog_features = substract_hog_features
        self.svc_kernel = svc_kernel

        # SVM model
        self.svm_model = None
        # Bounding box for cropping face
        self.bounding_box = (70, 60, 180, 200)  
        self.haar_detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    ########################################################################################################################################################
    # Read pairs from file
    def read_pairs(self, file_path):
        # Display log
        if self.display_log:
            print("Reading pairs from {}...".format(file_path))

        # Read pairs from file
        pairs = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line (number of pairs)
            for line in file:
                tokens = line.strip().split()
                pairs.append(tokens)
        return pairs

    # Normalize features
    def normalize_features(self, features):
        # Display log
        if self.display_log:
            print("Normalizing features...")

        # Check if features array is empty
        if features.size == 0:
            print("Error: Empty features array.")
            return features

        # Normalize features
        min_val = np.min(features)
        max_val = np.max(features)
        normalized = (features - min_val) / (max_val - min_val)
        return normalized

    ########################################################################################################################################################
    # Crop the image to the bounding box
    def crop_image(self, image):
        # Display log
        if self.display_log:
            print("Cropping image...")

        # Crop the image
        x, y, width, height = self.bounding_box
        cropped_image = image[y:y + height, x:x + width]
        return cropped_image

    # Detect and crop face using Haar Cascade
    def detect_and_crop_face_haar(self, image):
        # Display log
        if self.display_log:
            print("Detecting and cropping face...")

        # Detect faces in the image
        faces = self.haar_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            # Crop the first detected face
            x, y, w, h = faces[0]
            cropped_face = image[y:y + h, x:x + w]

            return cropped_face
        else:
            # Return the original image if no face is found
            return image

    ########################################################################################################################################################
    # Process pairs sequentially
    def process_pairs(self, pairs):
        # Display log
        if self.display_log:
            print("Processing pairs...")

        # Process pairs
        paired_hog_features = []
        labels = []
        for pair in pairs:
            hog_features_pair = []
            try:
                if len(pair) == 3:  # Same person
                    person, img_num1, img_num2 = pair
                    person1, person2 = person, person
                else:  # Different people
                    person1, img_num1, person2, img_num2 = pair

                for img_num, person in zip([img_num1, img_num2], [person1, person2]):
                    img_path = os.path.join(self.dataset_dir, person, "{}_{}.jpg".format(person, img_num.zfill(4)))
                    image = cv2.imread(img_path)

                    if image is None:
                        # Skip if the image cannot be read
                        print("Skipped: %s" % img_path)
                        continue

                    # Check if the image is already in grayscale
                    if len(image.shape) > 2 and image.shape[2] == 3:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = image  # The image is already in grayscale

                    # Detect and crop face
                    if self.face_detection_type == 'crop':
                        cropped_image = self.crop_image(gray_image)
                    elif self.face_detection_type == 'haar':
                        cropped_image = self.detect_and_crop_face_haar(gray_image)
                    else:
                        print("Error: Invalid face detection type.")
                        return

                    # Display log
                    if self.display_log:
                        # Display the cropped image
                        print("Calculating HOG features for %s..." % img_path)

                    # Extract HOG features from the cropped face
                    hog_features = hog(cropped_image, pixels_per_cell=(20, 20), cells_per_block=(2, 2), visualize=False)

                    # Resize HOG features to a consistent length (e.g., 1800) and flatten
                    hog_features = np.resize(hog_features, 1800).flatten()
                    
                    # Append the HOG features of the image
                    hog_features_pair.append(hog_features)

                # Compute the difference between HOG features of the two images in the pair
                if self.substract_hog_features:
                    hog_features_pair = abs(hog_features_pair[0] - hog_features_pair[1])

                # Append the HOG features of the pair
                paired_hog_features.append(np.hstack(hog_features_pair))
                label = 1 if len(pair) == 3 else 0
                labels.append(label)

            except Exception as e:
                # Print the error and skip the current pair
                print("Error processing pair %s: %s" % (pair, e))
                continue

        return np.array(paired_hog_features), labels

    ########################################################################################################################################################
    ########################################################################################################################################################
    # Train the model
    def train_model(self):
        # Display log
        if self.display_log:
            print("Training model...")

        # Process training pairs
        train_pairs = self.read_pairs(self.train_pairs_file)
        train_hog_features_array, train_labels = self.process_pairs(train_pairs)

        print("Before normalization - Train features shape:", train_hog_features_array.shape)

        # Normalize features
        train_hog_features_array = self.normalize_features(train_hog_features_array)

        # GRID SEARCH FOR OBTAINING BEST PARAMETERS
        # Define parameter grid for SVM
        if self.svc_kernel == 'linear':
            param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                          'kernel': ['linear']}

        elif self.svc_kernel == 'poly':
            param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                          'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                          'degree': [2, 3, 4],
                          'kernel': ['poly']}

        elif self.svc_kernel == 'rbf':
            param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                          'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                          'kernel': ['rbf']}

        elif self.svc_kernel == 'sigmoid':
            param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                          'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                          'kernel': ['sigmoid']}

        else:
            print("Error: Invalid SVC kernel.")
            return

        # Create and fit the GridSearchCV parameter grid using parallel processing (all cores)
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
        grid.fit(train_hog_features_array, train_labels)

        # Print the best parameters
        print("Best Parameters Found: ", grid.best_params_)

        # Train SVM model with best parameters
        self.svm_model = grid.best_estimator_

    ########################################################################################################################################################
    # Test the model
    # Test the model
    def test_model(self, title):
        # Display log
        if self.display_log:
            print("Testing model...")

        # Check if the model is trained
        if self.svm_model is None:
            print("Error: Model not trained. Call train_model() first.")
            return

        # Process test pairs
        test_pairs = self.read_pairs(self.test_pairs_file)
        hog_features_array_test, test_labels = self.process_pairs(test_pairs)
        hog_features_array_test = self.normalize_features(hog_features_array_test)

        # Predict on test data
        predicted_labels = self.svm_model.predict(hog_features_array_test)

        # Create output directory
        if not os.path.exists(os.path.join(self.output_file_path, title)):
            os.makedirs(os.path.join(self.output_file_path, title))

        # Create file handlers for good and bad recognitions
        good_recognition_file = open(os.path.join(self.output_file_path, title, 'good-recognition-examples.txt'), 'w')
        bad_recognition_file = open(os.path.join(self.output_file_path, title, 'bad-recognition-examples.txt'), 'w')

        # Iterate through test pairs and write paths to appropriate files
        for pair, label in zip(test_pairs, predicted_labels):
            try:
                img_path1 = os.path.join(self.dataset_dir, pair[0], "{}_{}.jpg".format(pair[0], pair[1].zfill(4)))
                img_path2 = os.path.join(self.dataset_dir, pair[2], "{}_{}.jpg".format(pair[2], pair[3].zfill(4)))

                if label == 1:  # Correct recognition
                    good_recognition_file.write("%s\t%s\n" % (img_path1, img_path2))
                else:  # Incorrect recognition
                    bad_recognition_file.write("%s\t%s\n" % (img_path1, img_path2))

            except IndexError:
                continue

        # Close the file handlers
        good_recognition_file.close()
        bad_recognition_file.close()

        # Calculate ROC curve for test set
        distances = self.svm_model.decision_function(hog_features_array_test)
        fpr, tpr, thresholds = roc_curve(test_labels, distances)
        roc_auc = auc(fpr, tpr)

        # Evaluate performance
        print(classification_report(test_labels, predicted_labels))

        return fpr, tpr, thresholds, roc_auc


    ########################################################################################################################################################
    # Plot ROC curve
    def plot_roc_curve(self, fpr, tpr, thresholds, auc_value, title):
        """Plot ROC curve and display Equal Error Rate (EER)."""
        # Display log
        if self.display_log:
            print("Plotting ROC curve...")

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %.2f)' % auc_value)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')

        # Calculate Equal Error Rate (EER)
        eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
        eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

        # Display EER as a point on the graph
        plt.plot(eer, 1 - eer, marker='o', markersize=6, color="red")
        plt.annotate("(%.4f, %.4f)" % (eer, 1 - eer), xy=(eer, 1 - eer), xytext=(eer - 0.2, 1 - eer + 0.1), color="red")
        
        # Print EER
        print("Equal Error Rate (EER): %.4f" % eer)

        # Save plot
        plt.savefig(os.path.join(self.output_file_path, title + '.png'))

########################################################################################################################################################
########################################################################################################################################################
if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description='Run LibLinear SVM on LFWA dataset.')
    # Add file path arguments
    parser.add_argument('--lfwa_dataset_path', type=str, default='lfwa-dataset', help='Path to the LFWA dataset.')
    parser.add_argument('--pairs_train_file', type=str, default='lfwa-pairs/pairsDevTrain.txt', help='Path to the pairsDevTrain.txt file.')
    parser.add_argument('--pairs_test_file', type=str, default='lfwa-pairs/pairsDevTest.txt', help='Path to the pairsDevTest.txt file.')
    parser.add_argument('--output_file_path', type=str, default='output-svm', help='Path to the output file.')
    # Add optional arguments
    parser.add_argument('--display_log', action='store_true', default=False, help='Display log.')
    parser.add_argument('--face_detection_type', type=str, default='haar', help='Face detector type (crop, haar).')
    parser.add_argument('--substract_hog_features', action='store_true', default=False, help='Substract HOG features.')
    parser.add_argument('--svc_kernel', type=str, default='rbf', help='SVC kernel (linear, poly, rbf, sigmoid, precomputed).')
    # Parse arguments
    args = parser.parse_args()

    # Create the GridSVC object
    grid_svc = GridSVC(args.lfwa_dataset_path,
                       args.pairs_train_file,
                       args.pairs_test_file,
                       args.output_file_path,
                       args.face_detection_type,
                       args.substract_hog_features,
                       args.svc_kernel, args.display_log)

    # Train the model
    grid_svc.train_model()

    # Test the model
    if args.substract_hog_features:
        fpr, tpr, thresholds, roc_auc = grid_svc.test_model('ROC-SVC-' + args.face_detection_type + '-' + args.svc_kernel + '-substractHOG')
    else:
        fpr, tpr, thresholds, roc_auc = grid_svc.test_model('ROC-SVC-' + args.face_detection_type + '-' + args.svc_kernel)

    # Plot ROC curve
    if args.substract_hog_features:
        grid_svc.plot_roc_curve(fpr, tpr, thresholds, roc_auc, 'ROC-SVC-' + args.face_detection_type + '-' + args.svc_kernel + '-substractHOG')
    else:
        grid_svc.plot_roc_curve(fpr, tpr, thresholds, roc_auc, 'ROC-SVC-' + args.face_detection_type + '-' + args.svc_kernel)

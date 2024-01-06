#!/usr/bin/env python3
import cv2
import os
from tqdm import tqdm

class Detector:
    def __init__(self):
        cascade_path = "./cascades/" + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)

    def get_faces(self, image_path, display=False, **kwargs):
        img = self._read_image(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections, _, weights = self.classifier.detectMultiScale3(gray, **kwargs, outputRejectLevels=True)

        results = [(x, y, w, h, score) for (x, y, w, h), score in zip(detections, weights)]
        if display:
            self._display_image(img, results)

        return results

    def _read_image(self, img_path):
        return cv2.imread(img_path)

    def _display_image(self, img, results):
        for x, y, w, h, score in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def output_detections(file, detections, image_path):
    file.write(image_path.rstrip(".jpg") + '\n')
    file.write(str(len(detections)) + '\n')
    for x, y, w, h, score in detections:
        file.write(f"{x} {y} {w} {h} {score:.2f}\n")

def process_images(image_files, output_files, detector, **kwargs):
    for img_file, out_file in tqdm(zip(image_files, output_files)):
        with open(out_file, "w") as out:
            with open(img_file, "r") as file:
                lines = file.readlines()
                idx = 0
                while idx < len(lines):
                    image_path = lines[idx].strip()
                    if not image_path.endswith(".jpg"):
                        image_path += ".jpg"
                    full_image_path = os.path.join(os.getcwd() + "/originalPics", image_path)
                    if idx + 1 < len(lines) and lines[idx+1].strip().isnumeric():
                        detections = detector.get_faces(full_image_path, **kwargs)
                        output_detections(out, detections, image_path)
                        idx += 1
                    idx += 1

if __name__ == "__main__":
    detector = Detector()

    base_dir = os.getcwd()
    file_range = range(1, 11)
    fold_dir = os.path.join(base_dir, "FDDB-folds")
    image_files = [os.path.join(fold_dir, f"FDDB-fold-{i:02}-ellipseList.txt") for i in file_range]
    default_out_files = [os.path.join(base_dir + "/detections-default", f"fold-{i:02}-out.txt") for i in file_range]
    accurate_out_files = [os.path.join(base_dir + "/detections", f"fold-{i:02}-out.txt") for i in file_range]

    default_settings = {}
    accurate_settings = {
        'scaleFactor': 1.05,
        'minNeighbors': 6,
        'minSize': (20, 20),
        'maxSize': (250, 250)
    }

    #process_images(image_files, default_out_files, detector, **default_settings)
    process_images(image_files, accurate_out_files, detector, **accurate_settings)
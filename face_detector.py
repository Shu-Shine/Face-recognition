import cv2
from mtcnn import MTCNN
import numpy as np

np.object = object

# The FaceDetector class provides methods for detection, tracking, and alignment of faces. # Preprocessing algorithms
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=60, tm_threshold=0.5, aligned_image_size=224):  #tm_window_size=20,tm_threshold=0.7
        # Prepare face alignment.
        self.detector = MTCNN()  # for detecting faces

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

    # Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size  # extension size beyond the bounding box


    # Track a face in a new image using template matching.
    def track_face(self, image):
        # Use the first detection as a reference
        if self.reference is None:
            self.reference = self.detect_face(image)
            if self.reference is None:
                return None

        # Convert the frames to grayscale
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reference = self.crop_face(self.reference["image"], self.reference["rect"])
        gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        # Define the search window, under the assumption of small motions
        x, y, width, height = self.reference["rect"]  # reference bounding box
        search_window = gray_frame[max(0, y - self.tm_window_size):min(y + height - 1 + self.tm_window_size, image.shape[0]-1),\
                        max(0, x - self.tm_window_size):min(x + width - 1 + self.tm_window_size, image.shape[1]-1)]

        # Perform template matching
        result = cv2.matchTemplate(search_window, gray_reference, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # similarity and location (top-left)
        # print('max_val', max_val)

        # Template matching is sensitive to extreme head pose variations
        if max_val < self.tm_threshold:
            # Re-initialize the tracker using MTCNN   # much slower
            new_detection = self.detect_face(image)
            # Update the reference detection
            if new_detection is not None:
                self.reference = new_detection

        # Calculate the face position based on the maximum similarity
        x_offset = max_loc[0] + x - self.tm_window_size
        y_offset = max_loc[1] + y - self.tm_window_size
        new_position = (x_offset, y_offset, width, height)

        # Align the detected face, before extracting features from the face
        aligned = self.align_face(image, new_position)

        return {"rect": new_position, "image": image, "aligned": aligned, "response": max_val}

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)  # MTCNN, dict_keys(['box', 'confidence', 'keypoints'])
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]  # (x, y, width, height) top-left

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size. 224×224 pixel
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))  # scaling

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]  # face area


import cv2
import face_recognition
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_faces_dir='data/known_faces'):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, known_faces_dir):
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                name = os.path.splitext(filename)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

    def recognize_faces(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            names.append(name)
        return names

    def get_face_description(self, frame):
        names = self.recognize_faces(frame)
        if not names:
            return "No faces recognized."
        return f"Recognized: {', '.join(names)}."
from multiprocessing import Process
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from win10toast import ToastNotifier
import datetime
import time
import math

blinkList=[]
distanceList=[]

class BlinkCalculator:
    # constants
    # right eye indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # Left eye indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    CLOSED_EYES_FRAME = 3
    # variables
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    colors = [(76, 168, 240), (255, 0, 255), (255, 255, 0)]
    # instantiation face detection solution
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)

    @staticmethod
    def draw_bbox(img, bbox, color, l=30, t=5, rt=1):
        """draw bounding box around user(s) face
        Args:
            img (numpy ndarray): video frame
            bbox (tuple): bounding box data (x,y,width, height)
            color (tuple): color in BGR
            l (int, optional): corners lines length. Defaults to 30.
            t (int, optional): corners lines thickness. Defaults to 5.
            rt (int, optional): bounding box thickness. Defaults to 1.
        """
        # draw bbox
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, color, rt)
        # top left
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        # top right
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        # bottom left
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        # bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

    @staticmethod
    def draw_dist_between_eyes(img, center_left, center_right, color, distance_value):
        """draw a line between user's eyes and annotate the distance (pixel) between them
        Args:
            img (numpy ndarray): video frame
            center_left (tuple): left eye landmark (x,y)
            center_right (tuple): right eye landmark (x,y)
            color (tuple): color in BGR
            distance_value ([type]): distance between eyes (pixel)
        """
        # mark eyes
        cv2.circle(img, center_left, 1, color, thickness=8)
        cv2.circle(img, center_right, 1, color, thickness=8)

        # line between eyes
        cv2.line(img, center_left, center_right, color, 3)

        # add distance value
        cv2.putText(img, f'{int(distance_value)}',
                    (center_left[0], center_left[1] -
                     10), cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)

    def run_config(self):
        """it is used to for the initial configuration of the system where the user needs to measure few distances in cm corresponding to different distances in pixel
        """

        # webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:
                    # get bbox data
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin * iw), int(bboxc.ymin *
                                                     ih), int(bboxc.width * iw), int(bboxc.height * ih)
                    bbox_list.append(bbox)

                    # get the eyes landmark
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x * iw), int(left_eye.y * ih)),
                                      (int(right_eye.x * iw), int(right_eye.y * ih))])

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):
                # calculate distance between left and right eye
                dist_between_eyes = np.sqrt(
                    (eye[0][1] - eye[1][1]) ** 2 + (eye[0][0] - eye[1][0]) ** 2)

                # draw bbox
                BlinkCalculator.draw_bbox(image, bbox, self.colors[0])

                # draw distace between eyes
                BlinkCalculator.draw_dist_between_eyes(
                    image, eye[0], eye[1], self.colors[0], dist_between_eyes)

            cv2.imshow('webcam', image)
            if cv2.waitKey(5) & 0xFF == ord('k'):
                break
        cap.release()

    def landmark_detection(img, results, draw=False):
        img_height, img_width = img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks
        return mesh_coord

    # Euclidean distance
    def euclidean_distance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    # Blinking Ratio
    def blink_ratio(img, landmarks, right_indices, left_indices):
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rh_distance = BlinkCalculator.euclidean_distance(rh_right, rh_left)
        rv_distance = BlinkCalculator.euclidean_distance(rv_top, rv_bottom)

        lv_distance = BlinkCalculator.euclidean_distance(lv_top, lv_bottom)
        lh_distance = BlinkCalculator.euclidean_distance(lh_right, lh_left)

        re_ratio = rh_distance / rv_distance
        le_ratio = lh_distance / lv_distance

        return (re_ratio + le_ratio) / 2

    def calculateBlinks(self,distance_pixel, distance_cm):
        print("calculateBlinks")
        datetime_start_blink = datetime.datetime.now()
        datetime_start = datetime.datetime.now()
        n = ToastNotifier()
        coff = np.polyfit(distance_pixel, distance_cm, 2)
        while self.cap.isOpened():
            success, image = self.cap.read()

            start = time.time()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False

            # Get the result
            results = self.face_mesh.process(image)
            results_distance = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            # To improve performance
            image.flags.writeable = True

            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                print("results")
                mesh_coordinates = BlinkCalculator.landmark_detection(image, results, False)
                ratio = BlinkCalculator.blink_ratio(image, mesh_coordinates, self.RIGHT_EYE, self.LEFT_EYE)
                cv2.putText(image, "Ratio: " + str(np.round(ratio, 2)), (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                            # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    #l  # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    # See where the user's head tilting
                    if y < -10:
                        text = "Looking Left"
                    elif y > 10:
                        text = "Looking Right"
                    elif x < -10:
                        text = "Looking Down"
                    elif x > 10:
                        text = "Looking Up"
                    else:
                        text = "Forward"
                        if ratio > 3.6:
                            self.CEF_COUNTER += 1
                        else:
                            if self.CEF_COUNTER > self.CLOSED_EYES_FRAME:
                                self.TOTAL_BLINKS += 1
                                self.CEF_COUNTER = 0

                    cv2.putText(
                        image, "Total Blinks: " + str(self.TOTAL_BLINKS), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2
                    )

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                     dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                    # Add the text on the image
                    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                    cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

                if ((datetime.datetime.now() - datetime_start_blink).total_seconds() / 60) >= 1:
                    if self.TOTAL_BLINKS > 2:
                        n.show_toast("TRINETRA", "you are blinking too much!! Take a break and relax your eyes")
                    blinkList.append(self.TOTAL_BLINKS)
                    self.TOTAL_BLINKS = 0
                    datetime_start_blink=datetime.datetime.now()

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
            if results_distance.detections:
                print("results_distance")
                for detection in results_distance.detections:
                    # get bbox data
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin * iw), int(bboxc.ymin *
                                                     ih), int(bboxc.width * iw), int(bboxc.height * ih)
                    bbox_list.append(bbox)

                    # get the eyes landmark
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x * iw), int(left_eye.y * ih)),
                                      (int(right_eye.x * iw), int(right_eye.y * ih))])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            distance_collection = []

            for bbox, eye in zip(bbox_list, eyes_list):

                # calculate distance between left and right eye
                dist_between_eyes = np.sqrt(
                    (eye[0][1] - eye[1][1]) ** 2 + (eye[0][0] - eye[1][0]) ** 2)

                # calculate distance in cm
                a, b, c = coff
                distance_cm = a * dist_between_eyes ** 2 + b * dist_between_eyes + c
                distance_collection.append(distance_cm)
                if ((datetime.datetime.now() - datetime_start).total_seconds() / 60) >= 1:
                    print("inside the main logic")
                    average = sum(distance_collection) / len(distance_collection)
                    if average <= 51:
                        # draw bbox
                        BlinkCalculator.draw_bbox(image, bbox, self.colors[1])
                        # add distance in cm

                        n.show_toast("TRINETRA",
                                     f'{int(distance_cm)} cm - you are too close to your computer screen. Please maintain some distance',
                                     duration=10)
                    datetime_start = datetime.datetime.now()

            cv2.imshow('Head Pose Estimation', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            
        self.cap.release()



if __name__ == '__main__':
    # # step-1: initial configuration
    # eye_screen_distance = DistanceCalculator()
    # eye_screen_distance.run_config()

    # step-2: get eye to screen distance
    distance_df = pd.read_csv('distance_xy.csv')
    eye_screen_distance = BlinkCalculator()
    eye_screen_distance.calculateBlinks(distance_df['distance_pixel'], distance_df['distance_cm'])

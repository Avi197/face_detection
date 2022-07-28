import cv2
import numpy as np

import face_mesh_point
import face_mesh_custom
from utils import euclidean_dist

right_eye = face_mesh_point.right_eye
left_eye = face_mesh_point.left_eye
outer_lips = face_mesh_point.outer_lips

face_mesh = face_mesh_custom.FaceMeshCustom(max_num_faces=2, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)


def detect_img(image):
    # Convert the color space from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance
    image.flags.writeable = False
    # Get the result
    results = face_mesh.process(image)
    # To improve performance
    image.flags.writeable = True
    return results


def validate_face(image, results, eye_check=False):
    img_h, img_w, img_c = image.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            jaw = {}
            min_x = 0
            max_x = 0
            min_y = 0
            max_y = 0
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if min_x == 0:
                    min_x = x
                if min_y == 0:
                    min_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if eye_check:
                    if idx in right_eye.keys():
                        right_eye[idx] = (x, y)
                    if idx in left_eye.keys():
                        left_eye[idx] = (x, y)
                # if idx in [61, 291]:
                if idx in outer_lips.keys():
                    outer_lips[idx] = (x, y)
                if idx in [132, 361]:
                    jaw[idx] = (x, y)
                if idx in [33, 263, 61, 291, 199]:
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
            box = [min_x, min_y, max_x, max_y]
            box[0] = int(box[0] - abs(box[2] - box[0]) * 10 / 100)
            box[1] = int(box[1] - abs(box[3] - box[1]) * 10 / 100)
            box[2] = int(box[2] + abs(box[2] - box[0]) * 10 / 100)
            box[3] = int(box[3] + abs(box[3] - box[1]) * 10 / 100)

            direction = detect_direction(face_2d, face_3d, img_h, img_w)
        return box, direction, (right_eye, left_eye), outer_lips, jaw
    return None, None, None, None, None


def validate_box_size(img, box, threshold=0.4):
    if all([b > 0 for b in box]) and box[0] < img.shape[0]:
        if img.shape[0] < img.shape[1]:
            return abs((box[1] - box[3]) / img.shape[0]) > threshold
        return abs((box[0] - box[2]) / img.shape[1]) > threshold
    return False


def validate_lmk(box, img_h, img_w, percent=95):
    y_0 = [x < 0 for x in [box[1], box[3]]]
    y_h = [x > img_h * percent / 100 for x in [box[1], box[3]]]
    x_0 = [y < 0 for y in [box[0], box[2]]]
    x_w = [y > img_w / 2 * percent / 100 for y in [box[0], box[2]]]
    print(box)
    if any(x_0) or any(x_w) or any(y_0) or any(y_h):
        return False
    return True


def validate_blur(frontal_face, threshold=11):
    gray = cv2.cvtColor(frontal_face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 300), interpolation=cv2.INTER_AREA)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_value > threshold


def validate_blink(eye, threshold=0.10):
    # Finding Distance Right Eye
    hR = euclidean_dist(eye[0][160], eye[0][144]) + euclidean_dist(eye[0][157], eye[0][154])
    wR = euclidean_dist(eye[0][33], eye[0][133]) * 2
    earRight = hR / wR

    # Finding Distance Left Eye
    hL = euclidean_dist(eye[1][384], eye[1][381]) + euclidean_dist(eye[1][387], eye[1][373])
    wL = euclidean_dist(eye[1][263], eye[1][362]) * 2
    earLeft = hL / wL

    ear = (earLeft + earRight) / 2
    if ear < threshold:
        return True
    return False


def draw_closed_loop(img, draw_dict):
    for idx, l in enumerate(list(draw_dict.values())):
        if idx != len(draw_dict) - 1:
            img = cv2.line(img, list(draw_dict.values())[idx], list(draw_dict.values())[idx + 1], (0, 255, 0), 2)
        else:
            img = cv2.line(img, list(draw_dict.values())[idx], list(draw_dict.values())[0], (0, 255, 0), 2)
    return img


# TODO: change global dict to something else
def check_smile(img, results, status, count_frame, stop_frame=20, ratio=0.45):
    # results = detect_img(img)
    if count_frame == stop_frame:
        status = 'Smile detected'
    else:
        box, _, _, lips, jaw = validate_face(img, results, eye_check=False)

        if box is not None:
            lips_length = euclidean_dist(lips[61], lips[291])
            jaw_length = euclidean_dist(jaw[132], jaw[361])
            img = draw_closed_loop(img, lips)

            if lips_length / jaw_length > ratio:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(img, f'Keep smiling for a few sec', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)
                count_frame += 1
            else:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                cv2.putText(img, f'Smile please', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                if count_frame > 0:
                    count_frame -= 4
        elif box is None:
            cv2.putText(img, 'Face not found', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            if count_frame > 0:
                count_frame -= 4
    return img, status, count_frame


def check_profile(img, results, status, count_frame, challenge):
    # results = detect_img(img)
    if count_frame == 50:
        status = f'{challenge} success'
    else:
        box, direction, _, _, _ = validate_face(img, results)
        if box is not None:
            if direction == challenge:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(img, f'Keep your face {challenge} for a few sec', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)
                count_frame += 1
            else:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                cv2.putText(img, f'Keep your face {challenge}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                if count_frame > 0:
                    count_frame -= 4
        elif box is None:
            cv2.putText(img, 'Face not found', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            if count_frame > 0:
                count_frame -= 4
    return img, status, count_frame


def check_eye_blink(img, results, status, blink, stop_frame=100):
    # results = detect_img(img)
    if blink == stop_frame:
        status = f'Blink success'
    else:
        box, _, eye, _, _ = validate_face(img, results, eye_check=True)
        if box is not None:
            img = draw_closed_loop(img, eye[0])
            img = draw_closed_loop(img, eye[1])

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            if len(eye[0]) == 16 and len(eye[1]) == 16:
                if validate_blink(eye, threshold=0.19):
                    blink += 1
            cv2.putText(img, f'Blink count: {blink}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        elif box is None:
            cv2.putText(img, 'Face not found', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return img, status, blink


def detect_direction(face_2d, face_3d, img_h, img_w):
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
    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360

    # See where the user's head tilting
    if y < -8:
        return "Left"
    elif y > 8:
        return "Right"
    elif x < -4:
        return "Down"
    elif x > 14:
        return "Up"
    else:
        return "Forward"

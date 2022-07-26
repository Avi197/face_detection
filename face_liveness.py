import face_mesh_custom
import face_mesh_point
import numpy as np
import cv2

from utils import euclidean_dist

right_eye = face_mesh_point.right_eye
left_eye = face_mesh_point.left_eye
outer_lips = face_mesh_point.outer_lips

mp_face_mesh = face_mesh_custom
face_mesh = mp_face_mesh.FaceMeshCustom(max_num_faces=1, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)


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


def validate_face(image, results, eye_check=False):
    img_h, img_w, img_c = image.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            # lips = {}
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


def draw_func(img, dic):
    for idx, l in enumerate(list(dic[0].values())):
        if idx != len(dic[0]) - 1:
            cv2.line(img, list(dic[0].values())[idx], list(dic[0].values())[idx + 1], (0, 255, 0), 2)
        else:
            cv2.line(img, list(dic[0].values())[idx], list(dic[0].values())[0], (0, 255, 0), 2)
    return img


# TODO: change global dict to something else
def check_smile(img, status, count_frame, stop_frame=20, ratio=0.45):
    results = detect_img(img)
    if count_frame == stop_frame:
        status = 'Smile detected'
    else:
        box, _, _, lips, jaw = validate_face(img, results, eye_check=False)

        if box is not None:
            lips_length = euclidean_dist(lips[61], lips[291])
            jaw_length = euclidean_dist(jaw[132], jaw[361])
            for idx, l in enumerate(list(lips.values())):
                if idx != len(lips) - 1:
                    cv2.line(img, list(lips.values())[idx], list(lips.values())[idx + 1], (0, 255, 0), 2)
                else:
                    cv2.line(img, list(lips.values())[idx], list(lips.values())[0], (0, 255, 0), 2)

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


def check_profile(img, status, count_frame, challenge):
    results = detect_img(img)
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


def check_eye_blink(img, status, blink, stop_frame=100):
    results = detect_img(img)
    if blink == stop_frame:
        status = f'Blink success'
    else:
        box, _, eye, _, _ = validate_face(img, results, eye_check=True)
        if box is not None:
            for idx, l in enumerate(list(eye[0].values())):
                if idx != len(eye[0]) - 1:
                    cv2.line(img, list(eye[0].values())[idx], list(eye[0].values())[idx + 1], (0, 255, 0), 2)
                else:
                    cv2.line(img, list(eye[0].values())[idx], list(eye[0].values())[0], (0, 255, 0), 2)
            for idx, l in enumerate(eye[1].values()):
                if idx != len(eye[1]) - 1:
                    cv2.line(img, list(eye[1].values())[idx], list(eye[1].values())[idx + 1], (0, 255, 0), 2)
                else:
                    cv2.line(img, list(eye[1].values())[idx], list(eye[1].values())[0], (0, 255, 0), 2)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            if len(eye[0]) == 16 and len(eye[1]) == 16:
                if validate_blink(eye, threshold=0.19):
                    blink += 1
            cv2.putText(img, f'Blink count: {blink}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        elif box is None:
            cv2.putText(img, 'Face not found', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return img, status, blink


def webcam(cap):
    front_count_frame = 0
    left_count_frame = 0
    right_count_frame = 0
    _front_count_frame = 0
    blink = 0
    smile = 0
    count = 0
    status = 'ready'
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            front_count_frame = 0
            left_count_frame = 0
            right_count_frame = 0
            blink = 0
            smile = 0
            status = 'start'
            count = 0
        if status == 'success':
            cv2.putText(img, 'You are a real person', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        if status == 'ready':
            cv2.putText(img, f'Press R to check liveness', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        if status == 'check_fail':
            cv2.putText(img, 'Check fail, press R to restart', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        if status == 'start':
            if count < 50:
                img = cv2.putText(img, 'Follow the action', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                count += 1
            else:
                img, status, left_count_frame = check_profile(img, status, left_count_frame, 'Left')

        if status == 'Left success':
            img, status, front_count_frame = check_profile(img, status, front_count_frame, 'Forward')
        if status == 'Forward success':
            img, status, right_count_frame = check_profile(img, status, right_count_frame, 'Right')
        if status == 'Right success':
            img, status, blink = check_eye_blink(img, status, blink)
        if status == 'Blink success':
            img, status, smile = check_smile(img, status, smile, stop_frame=60, ratio=0.44)
        if status == 'Smile detected':
            status = 'success'
        cv2.imshow('Head Pose Estimation', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    question = [1, 2, 3, 4]
    cap = cv2.VideoCapture(0)

    # webcam_temp(cap)
    webcam(cap)

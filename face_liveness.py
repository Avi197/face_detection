import cv2

from validate import check_profile, check_eye_blink, check_smile, detect_img


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
        results = detect_img(img)

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
                img, status, left_count_frame = check_profile(img, results, status, left_count_frame, 'Left')

        if status == 'Left success':
            img, status, front_count_frame = check_profile(img, results, status, front_count_frame, 'Forward')
        if status == 'Forward success':
            img, status, right_count_frame = check_profile(img, results, status, right_count_frame, 'Right')
        if status == 'Right success':
            img, status, blink = check_eye_blink(img, results, status, blink)
        if status == 'Blink success':
            img, status, smile = check_smile(img, results, status, smile, stop_frame=60, ratio=0.44)
        if status == 'Smile detected':
            status = 'success'
        cv2.imshow('Head Pose Estimation', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    webcam(cap)

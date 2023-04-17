import cv2

def webcam_capture(writepath):
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    idx = 0
    while True:
        ret, frame = cap.read()
        scale_factor = 1
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 27: # 27 is the ASCII code for the ESC key
            break
        elif c == ord('r'):
            #write frame to disk
            print(f"recording... idx:{idx}")
            cv2.imwrite(fr'{writepath}/{idx}.png', frame)
            idx += 1

    cap.release()
    cv2.destroyAllWindows()
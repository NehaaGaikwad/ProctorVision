import cv2

from camera import Camera


camera = Camera()

while True:

    frame = camera.read()

    if frame is None:
        print("Camera frame could not be read.")
        break

    fps = camera.calculate_fps()

    height, width = frame.shape[:2]

    # FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Camera status
    cv2.putText(
        frame,
        "Camera: ACTIVE",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Resolution
    cv2.putText(
        frame,
        f"Resolution: {width}x{height}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("ProctorVision AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
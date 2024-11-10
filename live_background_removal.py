import cv2
from rembg import remove
import numpy as np

# OpenCV Video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to bytes for rembg (rembg works with byte data)
    _, buffer = cv2.imencode('.png', frame)
    frame_bytes = buffer.tobytes()

    # Use rembg to remove the background
    output_bytes = remove(frame_bytes)

    # Convert the output bytes back to an image using numpy
    nparr = np.frombuffer(output_bytes, np.uint8) # type: ignore
    result_frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Check if result_frame has an alpha channel (transparency)
    if result_frame.shape[2] == 4:
        # Split the channels: BGRA (Blue, Green, Red, Alpha)
        bgr = result_frame[:, :, :3]  # Get the BGR part of the image
        alpha = result_frame[:, :, 3]  # Get the alpha channel

        # Create a mask for the transparent background
        mask = alpha == 0

        # You can replace the background with any color (e.g., white)
        bgr[mask] = (255, 255, 255)

        # Show the result
        cv2.imshow("Live Background Removal", bgr)

    else:
        # If no alpha channel is present, display the result without background removal
        cv2.imshow("Live Background Removal", result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

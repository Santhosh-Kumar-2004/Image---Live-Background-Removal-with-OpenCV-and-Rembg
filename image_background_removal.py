import cv2
from rembg import remove
import numpy as np

def remove_background(input_path, output_path):
    # Read the image with OpenCV
    image = cv2.imread(input_path)

    # Encode the image to bytes (rembg works with byte data)
    _, buffer = cv2.imencode('.png', image)
    image_bytes = buffer.tobytes()

    # Remove the background
    output_bytes = remove(image_bytes)

    # Convert the output bytes back to a NumPy array
    output_array = np.frombuffer(output_bytes, np.uint8) # type: ignore
    output_image = cv2.imdecode(output_array, cv2.IMREAD_UNCHANGED)

    # Check if the output image has an alpha channel
    if output_image.shape[2] == 4:
        # Split the channels: BGRA
        bgr = output_image[:, :, :3]  # Get the BGR part of the image
        alpha = output_image[:, :, 3]  # Get the alpha channel

        # Create a mask for transparency
        mask = alpha == 0

        # Set the background color (white) for transparent areas
        bgr[mask] = (255, 255, 255)

        # Save the output image
        cv2.imwrite(output_path, bgr)
        print(f"Background removed. Output saved to {output_path}")

    else:
        # If no alpha channel is present, save the result directly
        cv2.imwrite(output_path, output_image)
        print(f"Background removed. Output saved to {output_path}")

# Example usage
input_image_path = './known_faces/Santhosh.jpg'  # Path to your input image
output_image_path = './known_faces/Santhosh_no_bg.png'  # Path to save the output image
remove_background(input_image_path, output_image_path)

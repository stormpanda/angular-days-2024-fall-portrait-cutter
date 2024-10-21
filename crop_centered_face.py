import cv2
import dlib
import numpy as np
import sys
from rembg import remove
from PIL import Image

def crop_and_remove_background(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the image
    faces = detector(image, 1)

    if len(faces) == 0:
        print("No faces detected.")
        return

    # Get the bounding box of the first detected face
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Calculate the center of the bounding box
    center_x, center_y = x + w // 2, y + h // 2

    # Adjust the cropping area to include head and shoulders
    crop_width = w * 2
    crop_height = h * 2

    # Calculate the cropping box coordinates
    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    end_x = min(center_x + crop_width // 2, image.shape[1])
    end_y = min(center_y + crop_height // 2, image.shape[0])

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Convert the cropped image from BGR to RGB
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format for rembg
    cropped_image_pil = Image.fromarray(cropped_image_rgb)

    # Remove the background using rembg
    result_pil = remove(cropped_image_pil)

    # Convert the result back to OpenCV format
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    # Create a 3-channel image with anthracite grey background
    anthracite_grey = [41, 41, 41]  # RGB values for anthracite grey
    result_with_bg = np.full_like(result_bgr, anthracite_grey)
    mask = result_rgb[:, :, 3] > 0  # Use the alpha channel as mask
    result_with_bg[mask] = result_bgr[mask][:, :3]

    # Save the resulting image
    cv2.imwrite(output_path, result_with_bg)
    print(f"Cropped image with anthracite grey background saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_and_remove_background.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        crop_and_remove_background(input_image_path, output_image_path)
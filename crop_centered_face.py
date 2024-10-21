import cv2
import dlib
import numpy as np
import sys
import os
from rembg import remove
from PIL import Image

def crop_and_remove_background(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to load image {image_path}.")
        return
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    # Detect faces in the image
    faces = detector(image, 1)
    if len(faces) == 0:
        print(f"No faces detected in {image_path}.")
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
    result_rgba = np.array(result_pil)
    # Separate the alpha channel
    alpha_channel = result_rgba[:, :, 3]
    rgb_channels = result_rgba[:, :, :3]
    # Create a 3-channel image with anthracite grey background
    anthracite_grey = (41, 41, 41)  # RGB values for anthracite grey
    result_with_bg = np.full_like(rgb_channels, anthracite_grey)
    # Blend the foreground with the background using the alpha channel
    alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
    result_with_bg = (rgb_channels * alpha_factor + result_with_bg * (1 - alpha_factor)).astype(np.uint8)
    # Resize the image to fit within a 300x300 pixel box while maintaining aspect ratio
    result_with_bg_pil = Image.fromarray(result_with_bg)
    result_with_bg_pil.thumbnail((300, 300), Image.LANCZOS)
    # Create a new 300x300 pixel image with anthracite grey background
    final_image = Image.new("RGB", (300, 300), anthracite_grey)
    final_image.paste(result_with_bg_pil, ((300 - result_with_bg_pil.width) // 2, (300 - result_with_bg_pil.height) // 2))
    # Convert the final image back to OpenCV format
    final_image_cv = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
    # Save the resulting image
    cv2.imwrite(output_path, final_image_cv)
    print(f"Cropped image with anthracite grey background saved to {output_path}")

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)
        crop_and_remove_background(input_image_path, output_image_path)
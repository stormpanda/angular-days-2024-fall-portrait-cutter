import cv2
import dlib
import sys

def crop_head_and_shoulders(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
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
    # Increase the height and width of the cropping area
    crop_width = w * 2
    crop_height = h * 2

    # Calculate the cropping box coordinates
    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    end_x = min(center_x + crop_width // 2, image.shape[1])
    end_y = min(center_y + crop_height // 2, image.shape[0])

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_head_and_shoulders.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        crop_head_and_shoulders(input_image_path, output_image_path)
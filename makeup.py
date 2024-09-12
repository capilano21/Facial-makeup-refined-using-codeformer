import argparse
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to detect facial landmarks
def detect_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
    return None

# Function to normalize landmarks
def normalize_landmarks(landmarks, height, width):
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks.landmark]

# Function to blend images using soft light
def blend_soft_light(base, blend):
    base = base.astype(np.float32) / 255.0
    blend = blend.astype(np.float32) / 255.0
    result = np.clip((1 - 2 * blend) * (base ** 2) + 2 * base * blend, 0, 1) * 255.0
    return result.astype(np.uint8)

# Function to apply lipstick
def apply_lipstick(image, lip_mask, color_bgr):
    lipstick_layer = np.zeros_like(image, dtype=np.uint8)
    lipstick_layer[lip_mask > 0] = color_bgr
    result_image = image.copy()
    result_image[lip_mask > 0] = blend_soft_light(image[lip_mask > 0], lipstick_layer[lip_mask > 0])
    return result_image

# Function to apply eyeshadow with Gaussian blur
def apply_eye_shadow(image, landmarks, color, radius=21):
    left_eye_indices = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
    right_eye_indices = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]
    
    mask = np.zeros_like(image)
    left_eye = np.array([landmarks[i] for i in left_eye_indices], np.int32)
    right_eye = np.array([landmarks[i] for i in right_eye_indices], np.int32)

    cv2.fillPoly(mask, [left_eye], color)
    cv2.fillPoly(mask, [right_eye], color)
    
    mask = cv2.GaussianBlur(mask, (radius, radius), 60)
    output = cv2.addWeighted(image, 1.0, mask, 1.5, 0.0)
    
    return output

# Function to apply blush
def apply_blush(image, landmarks, cheek_indices, color, radius=41):
    mask = np.zeros_like(image)
    for idx in cheek_indices:
        cv2.circle(mask, landmarks[idx], radius, color, -1)
    
    mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    output = cv2.addWeighted(image, 1.0, mask, 0.15, 0.0)
    
    return output

# Main function to apply makeup
def apply_makeup(image_path, apply_lipstick_flag=True, apply_eyeshadow_flag=True, apply_blush_flag=True, 
                 lipstick_color_name='wine', eyeshadow_color=(70, 130, 180), blush_color=(255, 105, 180), save=False):
    # Load image
    image = cv2.imread(image_path)
    
    # Lipstick color options
    lipstick_colors = {
        'nude': (189, 154, 123), 'red': (0, 0, 255), 'dark_red': (0, 0, 139),
        'peach': (147, 112, 219), 'plum': (142, 69, 133), 'coral': (255, 127, 80),
        'wine': (70, 0, 50), 'brown': (101, 67, 33), 'fuchsia': (255, 0, 255), 'green': (0, 230, 20)
    }
    
    # Detect facial landmarks
    landmarks = detect_landmarks(image)
    if not landmarks:
        raise RuntimeError("No face landmarks detected.")
    
    # Normalize landmarks
    height, width = image.shape[:2]
    normalized_landmarks = normalize_landmarks(landmarks, height, width)
    
    # Create lip mask
    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 310, 270, 269, 267, 0, 37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    lip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(lip_mask, [np.array([normalized_landmarks[i] for i in lip_indices], np.int32)], 255)
    
    # Apply lipstick if the flag is set
    if apply_lipstick_flag:
        lipstick_color = lipstick_colors.get(lipstick_color_name, lipstick_colors['wine'])
        image = apply_lipstick(image, lip_mask, lipstick_color)
    
    # Apply eyeshadow if the flag is set
    if apply_eyeshadow_flag:
        image = apply_eye_shadow(image, normalized_landmarks, eyeshadow_color)
    
    # Apply blush if the flag is set (on cheekbones)
    if apply_blush_flag:
        left_cheek_indices = [50]  # Landmark near the left cheekbone
        right_cheek_indices = [280]  # Landmark near the right cheekbone
        image = apply_blush(image, normalized_landmarks, left_cheek_indices + right_cheek_indices, blush_color)
    
    # Convert BGR to RGB for displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the final result
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    if save:
        plt.savefig("makeup_result.jpg")
    plt.show()

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply makeup to an image.")
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument("--lipstick", action="store_true", help="Apply lipstick.")
    parser.add_argument("--eyeshadow", action="store_true", help="Apply eyeshadow.")
    parser.add_argument("--blush", action="store_true", help="Apply blush.")
    parser.add_argument("--lipstick_color", type=str, default="wine", help="Lipstick color name.")
    parser.add_argument("--eyeshadow_color", type=str, default="(70, 130, 180)", help="Eyeshadow color in BGR format.")
    parser.add_argument("--blush_color", type=str, default="(255, 105, 180)", help="Blush color in BGR format.")
    parser.add_argument("--save", action="store_true", help="Save the final image with makeup applied.")

    args = parser.parse_args()

    apply_makeup(args.image, apply_lipstick_flag=args.lipstick, apply_eyeshadow_flag=args.eyeshadow, 
                 apply_blush_flag=args.blush, lipstick_color_name=args.lipstick_color, 
                 eyeshadow_color=eval(args.eyeshadow_color), blush_color=eval(args.blush_color), save=args.save)

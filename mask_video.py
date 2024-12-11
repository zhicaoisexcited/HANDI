import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Pool, cpu_count

def create_hand_mask_with_union(mediapipe_image, detection_result, frame_width, frame_height):
    hand_landmarks_list = detection_result.hand_landmarks
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    all_hand_points = []

    for hand_landmarks in hand_landmarks_list:
        hand_coordinates = np.array([
            (int(landmark.x * frame_width), int(landmark.y * frame_height))
            for landmark in hand_landmarks
        ], dtype=np.int32)
        all_hand_points.extend(hand_coordinates)
        cv2.fillPoly(mask, [hand_coordinates], 255)

    all_hand_points = np.array(all_hand_points)
    if len(all_hand_points) > 0:
        hull = cv2.convexHull(all_hand_points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask

def process_video(filename, input_folder, output_folder_union, model_path):
    # Only process video files
    if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return

    video_name_without_extension = os.path.splitext(filename)[0]
    union_output_path = os.path.join(output_folder_union, f"{video_name_without_extension}.png")

    # Skip processing if the union mask image already exists
    if os.path.exists(union_output_path):
        print(f"Skipping already processed video file: {filename}")
        return

    video_path = os.path.join(input_folder, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final_mask_union = np.zeros((frame_height, frame_width), dtype=np.uint8)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2, 
        min_hand_detection_confidence=0.1, 
        min_hand_presence_confidence=0.1, 
        min_tracking_confidence=0.1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        frame_np = mp_image.numpy_view()

        mask_union = create_hand_mask_with_union(frame_np, detection_result, frame_width, frame_height)
        final_mask_union = cv2.bitwise_or(final_mask_union, mask_union)

    cap.release()
    detector.close()

    try:
        cv2.imwrite(union_output_path, final_mask_union)
        print(f"Processed {filename}, union mask saved to {union_output_path}")
    except Exception as e:
        print(f"Failed to save union mask: {union_output_path}, Error: {e}")

def main():
    input_folder = '/home/zhicao/ego4d/train/target'
    output_folder_union = '/scratch1/home/zhicao/ego4d/mask_video'
    model_path = 'hand_landmarker.task'
    
    os.makedirs(output_folder_union, exist_ok=True)

    filenames = os.listdir(input_folder)
    args = [(filename, input_folder, output_folder_union, model_path) for filename in filenames]

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_video, args)

if __name__ == "__main__":
    main()

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn.functional as F
import numpy as np
import time
import cv2

def decode_latents(vae, latents):
    start_time = time.time()
    latents = latents.to(vae.device)
    latents = 1 / vae.config.scaling_factor * latents
    batch_size, channels, height, width = latents.shape
    latents = latents.view(batch_size, channels, height, width)
    with torch.no_grad():
        image = vae.decode(latents).sample
    # print(f"Decode latents: {time.time() - start_time:.4f} seconds")
    return image.float()

def hand_pose_loss(latent_video1, latent_video2, vae, device='cuda'):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    mse_loss = 0.0
    vae.to(device)
    vae.eval()
    
    frames = latent_video1.shape[2]

    for i in range(0, frames, 4):
        frame_start = time.time()

        # 将 latents 移动到 GPU
        latent_frame1 = latent_video1[:, :, i, :, :].to(device)
        latent_frame2 = latent_video2[:, :, i, :, :].to(device)
        
        # 解码 latents
        decoded_frames1 = decode_latents(vae, latent_frame1)
        decoded_frames2 = decode_latents(vae, latent_frame2)
        
        print("latent:", latent_video1.shape)
        print("decoded_frames1:", decoded_frames1.shape)
        # 转换为 NumPy 数组
        start_np = time.time()
        frame1 = decoded_frames1.detach().cpu().numpy()
        frame2 = decoded_frames2.detach().cpu().numpy()
        # print(f"Convert to numpy: {time.time() - start_np:.4f} seconds")

        for batch_idx in range(frame1.shape[0]):
            batch_start = time.time()

            img1 = frame1[batch_idx]
            img2 = frame2[batch_idx]

            img1_uint8 = (img1 * 255).astype(np.uint8)
            img2_uint8 = (img2 * 255).astype(np.uint8)

            # i1: (256, 256, 3)
            i1 = np.transpose(img1_uint8, (1, 2, 0))
            i2 = np.transpose(img2_uint8, (1, 2, 0))
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)

            # MediaPipe 检测
            detection_start = time.time()
            image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=i1)
            image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=i2)
            # print(f"Batch {batch_idx} preparation: {time.time() - batch_start:.4f} seconds")

            detection_result1 = detector.detect(image1)
            detection_result2 = detector.detect(image2)
            # print(f"Detection time: {time.time() - detection_start:.4f} seconds")

            # 计算 MSE 损失
            loss_start = time.time()
            hand_landmarks_list1 = detection_result1.hand_landmarks
            hand_landmarks_list2 = detection_result2.hand_landmarks

            for hand_landmarks1, hand_landmarks2 in zip(hand_landmarks_list1, hand_landmarks_list2):
                landmarks_array1 = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks1], dtype=np.float32)
                landmarks_array2 = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks2], dtype=np.float32)
                mse_loss += np.mean((landmarks_array1 - landmarks_array2) ** 2)
        #     print(f"Loss calculation: {time.time() - loss_start:.4f} seconds")

        # print(f"Frame {i} total time: {time.time() - frame_start:.4f} seconds")

    return mse_loss

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn.functional as F
import numpy as np
import time
import cv2

def decode_latents(vae, latents):
    latents = latents.to(vae.device)
    latents = 1 / vae.config.scaling_factor * latents
    batch_size, channels, height, width = latents.shape
    latents = latents.view(batch_size, channels, height, width)
    with torch.no_grad():
        image = vae.decode(latents).sample
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

    mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
    vae.to(device)
    vae.eval()
    
    frames = latent_video1.shape[2]

    for i in range(0, frames, 4):
        latent_frame1 = latent_video1[:, :, i, :, :].to(device)
        latent_frame2 = latent_video2[:, :, i, :, :].to(device)

        decoded_frames1 = decode_latents(vae, latent_frame1)
        decoded_frames2 = decode_latents(vae, latent_frame2)
        
        print("latent:", latent_video1.shape)
        print("decoded_frames1:", decoded_frames1.shape)

        frame1 = (decoded_frames1 * 255).to(torch.uint8)
        frame2 = (decoded_frames2 * 255).to(torch.uint8)

        for batch_idx in range(frame1.shape[0]):
            img1 = frame1[batch_idx]
            img2 = frame2[batch_idx]

            # Convert to CPU for mediapipe processing
            img1_np = img1.permute(1, 2, 0).cpu().numpy()
            img2_np = img2.permute(1, 2, 0).cpu().numpy()
            
            img1_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)
            img2_np = cv2.cvtColor(img2_np, cv2.COLOR_BGR2RGB)

            image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=img1_np)
            image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=img2_np)

            detection_result1 = detector.detect(image1)
            detection_result2 = detector.detect(image2)

            hand_landmarks_list1 = detection_result1.hand_landmarks
            hand_landmarks_list2 = detection_result2.hand_landmarks

            for hand_landmarks1, hand_landmarks2 in zip(hand_landmarks_list1, hand_landmarks_list2):
                # Convert landmarks to PyTorch tensors
                landmarks1 = torch.tensor([[lm.x, lm.y, lm.z] for lm in hand_landmarks1], 
                    dtype=torch.float32, device=device, requires_grad=True)
                landmarks2 = torch.tensor([[lm.x, lm.y, lm.z] for lm in hand_landmarks2], 
                    dtype=torch.float32, device=device, requires_grad=True)
                
                # Calculate MSE loss using PyTorch
                mse_loss = mse_loss + F.mse_loss(landmarks1, landmarks2)

    return mse_loss

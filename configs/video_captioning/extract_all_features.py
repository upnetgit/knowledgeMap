import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_resnet(weights_path):
    model = models.resnet152(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # 去掉最后的 avgpool 和 fc，保留到 conv5 输出特征图
    model = nn.Sequential(*list(model.children())[:-2])
    model = model.to(device).eval()
    return model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features_from_video(video_path, model, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    if len(frames) < max_frames:
        # 重复最后一帧补齐
        last = frames[-1]
        frames.extend([last] * (max_frames - len(frames)))
    # 批量预处理
    batch = torch.stack([transform(f) for f in frames]).to(device)
    with torch.no_grad():
        feat = model(batch)          # [max_frames, 2048, 7, 7]
        feat = feat.mean(dim=[2,3])  # 全局平均池化 -> [max_frames, 2048]
    return feat.cpu().numpy()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_resnet(args.resnet_weight)
    video_files = [f for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    for video_file in tqdm(video_files, desc="Extracting features"):
        video_path = os.path.join(args.video_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        out_path = os.path.join(args.output_dir, f"{base_name}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Skipping existing {out_path}")
            continue
        try:
            features = extract_features_from_video(video_path, model, max_frames=args.max_frames)
            np.save(out_path, features)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Folder containing videos')
    parser.add_argument('--output_dir', type=str, default='./data/features', help='Output folder for .npy files')
    parser.add_argument('--resnet_weight', type=str, required=True, help='Path to resnet152-394f9c45.pth')
    parser.add_argument('--max_frames', type=int, default=50, help='Number of frames to extract')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features')
    args = parser.parse_args()
    main(args)
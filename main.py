import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from IPython.display import display, Image as IPyImage

%matplotlib inline

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def sample_video_clips(video_path, clip_length=8, frame_stride=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    pbar = tqdm(total=total_frames, desc="Reading frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(Image.fromarray(frame))
        pbar.update(1)
    cap.release()
    pbar.close()

    clips = []
    for i in range(0, len(frame_list) - clip_length + 1, frame_stride):
        clip = frame_list[i:i + clip_length]
        clips.append(clip)
    return clips

def preprocess_clips(clips):
    clip_tensors = []
    for clip in clips:
        inputs = processor(images=clip, return_tensors="pt", padding=True)
        clip_tensor = inputs['pixel_values']  # Shape: (clip_length, 3, H, W)
        clip_tensors.append(clip_tensor)
    clip_tensors = torch.stack(clip_tensors)  # Shape: (num_clips, clip_length, 3, H, W)
    return clip_tensors

def get_clip_embeddings(clip_tensors):
    num_clips, clip_length, C, H, W = clip_tensors.shape
    clip_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(num_clips), desc="Extracting clip embeddings"):
            clip = clip_tensors[i].to(device)
            frame_embeddings = model.get_image_features(pixel_values=clip)
            frame_embeddings /= frame_embeddings.norm(p=2, dim=-1, keepdim=True)
            clip_embedding = frame_embeddings.mean(dim=0)
            clip_embeddings.append(clip_embedding.cpu())
    clip_embeddings = torch.stack(clip_embeddings)
    return clip_embeddings

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu()

def find_matching_clips(text, clip_embeddings, threshold=0.3):
    text_embedding = get_text_embedding(text)
    similarities = torch.matmul(clip_embeddings, text_embedding.T).squeeze()
    matching_indices = (similarities >= threshold).nonzero(as_tuple=True)[0]
    return matching_indices, similarities

def display_matching_clips(clips, matching_indices, similarities, top_k=3, output_gif='output.gif'):
    if len(matching_indices) == 0:
        print("No matching clips found.")
        return

    sorted_indices = matching_indices[similarities[matching_indices].argsort(descending=True)]
    top_indices = sorted_indices[:top_k]

    gif_frames = []
    for idx in top_indices:
        clip = clips[idx]
        gif_frames.extend([np.array(frame) for frame in clip])

    imageio.mimsave(output_gif, gif_frames, fps=5)
    print(f"GIF saved as {output_gif}")

    display(IPyImage(filename=output_gif))

def search_video_with_text(video_path, text_query, clip_length=8, frame_stride=2, threshold=0.3):
    clips = sample_video_clips(video_path, clip_length=clip_length, frame_stride=frame_stride)
    print(f"Number of clips sampled: {len(clips)}")

    print("Extracting clip embeddings...")
    clip_tensors = preprocess_clips(clips)
    clip_embeddings = get_clip_embeddings(clip_tensors)

    print(f"Searching for '{text_query}' in video...")
    matching_indices, similarities = find_matching_clips(text_query, clip_embeddings, threshold=threshold)

    display_matching_clips(clips, matching_indices, similarities)

if __name__ == "__main__":

    video_path = "content/IMG_3181.mp4"

    text_query = "golf follow through"

    # Saw slightly better performance with a higher clip length and lower threshold. Will test more and update the Readme
    search_video_with_text(video_path, text_query, clip_length=16, frame_stride=2, threshold=0.10)

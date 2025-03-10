# preprocess.py
import os
import torch
import numpy as np
import torchaudio
from decord import VideoReader
from tqdm.autonotebook import tqdm
from PIL import Image
import torchvision.transforms as T
from data_utils import load_chat  # so we can reference your existing function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_fn = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_frames_decord(video_path, start_sec, end_sec, fps=1):
    try:
        vr = VideoReader(video_path)
    except:
        raise IOError(f"Could not open video {video_path}")

    video_fps = vr.get_avg_fps()
    if video_fps <= 0:
        video_fps = 30.0

    start_frame = int(start_sec * video_fps)
    end_frame   = int(end_sec * video_fps)
    end_frame   = min(end_frame, len(vr) - 1)

    if start_frame > end_frame:
        return torch.empty(0)

    total_duration = end_sec - start_sec
    num_frames = int(total_duration * fps)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=np.int32)
    indices = np.clip(indices, 0, len(vr) - 1)

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = T.functional.resize(frames_tensor, (224, 224))
    frames_tensor = T.functional.normalize(
        frames_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return frames_tensor

def extract_audio_snippet(start_sec, end_sec, waveform, mel_transform, TARGET_SR):
    start_sample = int(start_sec * TARGET_SR)
    end_sample   = int(end_sec * TARGET_SR)
    end_sample   = min(end_sample, waveform.shape[1])

    snippet = waveform[:, start_sample:end_sample]

    mel_spec = mel_transform(snippet)
    mel_spec = torch.log(mel_spec + 1e-9)
    # If stereo, average
    if mel_spec.shape[0] > 1:
        return mel_spec.mean(dim=0)
    else:
        return mel_spec.squeeze(0)

def precompute_features(video_path, audio_path, chat_file, output_dir):
    """
    Precompute frames & audio spectrogram for each chat message.
    """
    os.makedirs(output_dir, exist_ok=True)

    chat_entries = load_chat(chat_file)
    waveform, original_sr = torchaudio.load(audio_path)
    
    TARGET_SR = 16000
    if original_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(original_sr, TARGET_SR)
        waveform = resampler(waveform)

    # Optionally move to GPU if you prefer
    waveform = waveform.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
    ).to(device)

    for idx, (time_s, _, _) in enumerate(tqdm(chat_entries, desc="Data preparation")):
        start_sec = max(0, time_s - 10)
        end_sec   = time_s

        # Video
        frames = extract_frames_decord(video_path, start_sec, end_sec, fps=1)
        torch.save(frames, os.path.join(output_dir, f"vid_{idx}.pt"))

        # Audio
        audio_snippet = extract_audio_snippet(start_sec, end_sec, waveform, mel_transform, TARGET_SR)
        torch.save(audio_snippet, os.path.join(output_dir, f"aud_{idx}.pt"))

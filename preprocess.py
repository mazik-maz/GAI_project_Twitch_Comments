import os
import math
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from PIL import Image

from decord import VideoReader

import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

import torchvggish
from torchvggish import vggish_input

from data_utils import load_chat, tokenize_comment  # your local utilities

torchaudio.set_audio_backend("soundfile")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 1) Create (or load) models and transforms outside so we don't re-init them
################################################################################
# ResNet for video frames
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC layer
resnet.eval().to(device)

# VGGish for audio
vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish_model.eval().to(device)

# Image transform for ResNet
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

################################################################################
# 2) Helper function: chunkify the entire video in one pass
################################################################################
def chunkify_video(
    video_path: str,
    snippet_duration: float = 10.0,
    fps_for_sampling: float = 1.0
):
    """
    Loads the entire video once via decord. Splits it into 10-second chunks,
    extracts frames for each chunk at a certain sampling fps (fps_for_sampling),
    and returns a list of averaged ResNet embeddings, one per chunk.
    """
    try:
        vr = VideoReader(video_path)
    except:
        raise IOError(f"Could not open video {video_path}")

    native_fps = vr.get_avg_fps()
    if native_fps <= 0:
        native_fps = 30.0

    # Total duration in seconds (approx)
    total_frames = len(vr)
    total_duration_sec = total_frames / native_fps

    # Number of full 10-second chunks (the last one might be shorter)
    num_chunks = math.ceil(total_duration_sec / snippet_duration)

    # For each chunk, gather frames, compute average embedding
    video_embeddings = []  # will hold (num_chunks) tensors of shape (2048,)

    for chunk_idx in tqdm(range(num_chunks)):
        start_sec = chunk_idx * snippet_duration
        end_sec   = min((chunk_idx + 1) * snippet_duration, total_duration_sec)

        start_frame = int(round(start_sec * native_fps))
        end_frame   = int(round(end_sec * native_fps))
        end_frame   = min(end_frame, total_frames - 1)

        if start_frame > end_frame:
            # Could happen if snippet is out of range
            video_embeddings.append(torch.zeros(2048))
            continue

        # Choose how many frames we want to sample in this chunk
        chunk_duration = end_sec - start_sec
        num_sampled_frames = int(chunk_duration * fps_for_sampling)
        if num_sampled_frames < 1:
            num_sampled_frames = 1

        # Indices for frames in this chunk
        indices = np.linspace(start_frame, end_frame, num_sampled_frames, dtype=np.int32)
        indices = np.clip(indices, 0, total_frames - 1)

        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)

        # Extract embeddings for each frame, then average
        emb_list = []
        with torch.no_grad():
            for frame in frames:
                img = Image.fromarray(frame, mode='RGB')
                img_t = img_transform(img).unsqueeze(0).to(device)
                feat = resnet(img_t)         # (1, 2048, 1, 1)
                feat = feat.squeeze()        # (2048,)
                emb_list.append(feat)

        if len(emb_list) == 0:
            video_embeddings.append(torch.zeros(2048))
        else:
            # Average the frame embeddings
            emb_tensor = torch.stack(emb_list, dim=0)  # (N, 2048)
            snippet_emb = emb_tensor.mean(dim=0)       # (2048,)
            video_embeddings.append(snippet_emb.cpu())

    return video_embeddings

################################################################################
# 3) Helper function: chunkify the entire audio in one pass
################################################################################
def chunkify_audio(
    audio_path: str,
    snippet_duration: float = 10.0,
    sample_rate: int = 16000
):
    """
    Loads the entire audio once, or uses torchaudio to read in
    chunks of 10 seconds, then passes each chunk to VGGish, returning
    a list of average audio embeddings (128-dim).
    """
    info = torchaudio.info(audio_path)
    total_frames = info.num_frames
    sr = info.sample_rate
    # total duration in seconds
    total_duration_sec = total_frames / float(sr)

    # Number of chunks
    num_chunks = math.ceil(total_duration_sec / snippet_duration)

    audio_embeddings = []

    # We can do chunk-based reading from file (to avoid reading a huge wave in memory)
    for chunk_idx in tqdm(range(num_chunks)):
        start_sec = chunk_idx * snippet_duration
        end_sec   = min((chunk_idx + 1) * snippet_duration, total_duration_sec)

        start_sample = int(round(start_sec * sr))
        end_sample   = int(round(end_sec * sr))
        num_frames   = end_sample - start_sample

        if num_frames <= 0:
            # empty chunk
            audio_embeddings.append(torch.zeros(128))
            continue

        # Load just the chunk
        wf_snip, sr_snip = torchaudio.load(
            audio_path,
            frame_offset=start_sample,
            num_frames=num_frames
        )  # wf_snip shape: (channels, samples)

        # Convert to mono if needed
        wf_snip = wf_snip.mean(dim=0).numpy()

        # Pass through VGGish
        with torch.no_grad():
            embed = vggish_model(wf_snip, fs=sr_snip)  # shape might be (1,128) or multiple chunks
        if embed.dim() > 1:
            # Average across sub-chunks if VGGish splits it
            embed = embed.mean(dim=0)  # => (128,)
        audio_embeddings.append(embed.cpu())

    return audio_embeddings

################################################################################
# 4) Main function: precompute all 10s chunk embeddings, then join them
################################################################################
def preprocess_files(
    video_path,
    audio_path,
    chat_file,
    output_dir,
    snippet_duration=10.0,
    sample_rate=16000,
    fps_for_sampling=1.0,
    word2idx=None
):
    """
    1) Precompute video embeddings in fixed 10s chunks
    2) Precompute audio embeddings in fixed 10s chunks
    3) For each chat entry, pick the chunk that corresponds
       to the chat timestamp floor-divided by snippet_duration
       (or clip to the last if out of range), then save to disk
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Chunkify the video and audio
    print("Chunkifying the video...")
    video_embeddings = chunkify_video(
        video_path=video_path,
        snippet_duration=snippet_duration,
        fps_for_sampling=fps_for_sampling
    )
    print(f"Video produced {len(video_embeddings)} chunks.")

    print("Chunkifying the audio...")
    audio_embeddings = chunkify_audio(
        audio_path=audio_path,
        snippet_duration=snippet_duration,
        sample_rate=sample_rate
    )
    print(f"Audio produced {len(audio_embeddings)} chunks.")

    # 2) Load chat lines
    chat_entries = load_chat(chat_file)

    # 3) For each chat line, find the appropriate chunk index and assemble
    print("Processing chat lines...")
    num_video_chunks = len(video_embeddings)
    num_audio_chunks = len(audio_embeddings)
    max_chunks = max(num_video_chunks, num_audio_chunks)

    for idx, (time_s, speaker, comment) in enumerate(tqdm(chat_entries)):
        # snippet index for time_s
        chunk_idx = int(time_s // snippet_duration)

        # if out of range, clamp to the last chunk
        if chunk_idx >= max_chunks:
            chunk_idx = max_chunks - 1
            if chunk_idx < 0:
                # no valid chunks at all => store zeros or skip
                vid_emb = torch.zeros(2048)
                aud_emb = torch.zeros(128)
        # Retrieve the chunk embeddings (default to zeros if the index is out of range)
        if chunk_idx < len(video_embeddings) and chunk_idx >= 0:
            vid_emb = video_embeddings[chunk_idx]
        else:
            vid_emb = torch.zeros(2048)

        if chunk_idx < len(audio_embeddings) and chunk_idx >= 0:
            aud_emb = audio_embeddings[chunk_idx]
        else:
            aud_emb = torch.zeros(128)

        # 4) Tokenize text (if you have a word2idx)
        if word2idx:
            text_tokens = tokenize_comment(comment, word2idx)
            text_t = torch.LongTensor(text_tokens)
        else:
            text_t = comment  # or store raw text

        # 5) Save sample
        sample_dict = {
            'video': vid_emb,     # (2048,)
            'audio': aud_emb,     # (128,)
            'text': text_t,
            'time': time_s,
            'speaker': speaker
        }
        outpath = os.path.join(output_dir, f"out_{idx}.pt")
        torch.save(sample_dict, outpath)

    print(f"Done. Processed {len(chat_entries)} chat lines.")

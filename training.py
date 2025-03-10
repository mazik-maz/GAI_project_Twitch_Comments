import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from decord import VideoReader
from PIL import Image
from tqdm.autonotebook import tqdm
import numpy as np
import cv2
import torchaudio

##########################################
# 1. TRAINING LOOP
##########################################

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, device='cuda'):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}')
    for batch in dataloader:
        # Move data to GPU
        videos = batch['video']  # (B, T, 3, 224, 224)
        audios = batch['audio']  # (B, n_mels, A_time)
        texts  = batch['text']   # (B, seq_len)

        B, T, C, H, W = videos.shape
        # Flatten frames => (B, T, 3*224*224)
        video_flat = videos.view(B, T, -1).to(device)

        # For audio, we want shape (B, A_time, n_mels) => transpose
        # (B, n_mels, A_time) => (B, A_time, n_mels)
        audios_t = audios.transpose(1, 2).to(device)

        optimizer.zero_grad()
        logits = model(video_flat, audios_t, texts).to(device)
        # logits => (B, seq_len, vocab_size)

        B_, seq_len, vocab_sz = logits.shape
        loss = criterion(logits.view(B_*seq_len, vocab_sz).to(device), texts.view(-1).to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix_str(f'Loss: {loss.item()}')
        pbar.update()
        # print(batch_idx, len(dataloader), loss.item(), end=' ')

        # OPTIONAL: Slow down the loop so nvidia-smi can catch GPU usage
        # time.sleep(0.1)

    return total_loss / len(dataloader)

def run_training_example():
    """
    Illustrates the entire flow.
    Assumes we have:
      - 'stream.mp4', 'stream.wav', 'chat.txt'
      - chat.txt includes lines like "[0:00:10] StreamElements: ..."

    If your data is large or not present, you'll need to adapt or create smaller test data.
    """
    # ========== BUILD VOCAB =============
    chat_path = "/kaggle/input/dorozeadata/chat1.txt"  # adapt path as needed
    special_toks = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    word2idx, idx2word = build_vocabulary(
        chat_file=chat_path,
        min_freq=1,
        max_size=5000,
        special_tokens=special_toks
    )
    vocab_size = len(word2idx)
    print("Vocab size:", vocab_size)

    # ========== CREATE DATASET ==========
    # Run this once to precompute features
    if not os.path.isdir("/kaggle/working/cached_features"):
        precompute_features(
            video_path="/kaggle/input/dorozeadata/video1.mp4",
            audio_path="/kaggle/input/dorozeadata/audio.wav",
            chat_file="/kaggle/input/dorozeadata/chat1.txt",
            output_dir="/kaggle/working/cached_features"
        )
    
    # Then create dataset from cache
    dataset = TwitchCommentDataset(
        cache_dir="/kaggle/working/cached_features",
        chat_file="/kaggle/input/dorozeadata/chat1.txt",
        word2idx=word2idx
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Adjust for your GPU
        shuffle=True,
        collate_fn=my_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset length:", len(dataset))

    # ========== INIT MODEL ==========
    video_feature_dim = 3*224*224
    audio_feature_dim = 64
    model = MultiModalLSTM(
        vocab_size=vocab_size,
        video_feature_dim=video_feature_dim,
        audio_feature_dim=audio_feature_dim,
        hidden_dim=512
    ).to(device)

    print("Model on device:", next(model.parameters()).device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # If <PAD> = index 0, you can ignore it in CrossEntropy
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # ========== TRAIN LOOP ==========
    num_epochs = epoch
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, epoch+1, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        MODEL_PATH = "/kaggle/working/my_multimodal_model.pth"
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model weights saved to {MODEL_PATH}")

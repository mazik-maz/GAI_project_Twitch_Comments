# train.py
import torch
import time
from tqdm.autonotebook import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, device='cuda'):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}')
    for batch in dataloader:
        videos = batch['video']  # (B, T, 3, 224, 224)
        audios = batch['audio']  # (B, n_mels, A_time)
        texts  = batch['text']   # (B, seq_len)

        B, T, C, H, W = videos.shape
        # Flatten frames => (B, T, 3*224*224)
        video_flat = videos.view(B, T, -1).to(device)
        # (B, n_mels, A_time) -> (B, A_time, n_mels)
        audios_t   = audios.transpose(1, 2).to(device)

        optimizer.zero_grad()
        logits = model(video_flat, audios_t, texts.to(device))  # (B, seq_len, vocab_size)
        
        B_, seq_len, vocab_sz = logits.shape
        loss = criterion(logits.view(B_*seq_len, vocab_sz), texts.view(-1).to(device))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix_str(f'Loss: {loss.item():.4f}')
        pbar.update()
        
    pbar.close()
    return total_loss / len(dataloader)

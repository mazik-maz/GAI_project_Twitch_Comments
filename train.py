# train.py
import torch
import time
import numpy as np
from tqdm.autonotebook import tqdm

def weighted_sum_rows(logits, end_idx, alpha=1.0):
    """
    Compute a weighted sum of `logits[0]` through `logits[end_idx]`,
    where rows closer to `end_idx` get more weight.

    Args:
        logits (torch.Tensor): 2D tensor of shape (batch_size, vocab_size).
        end_idx (int): Last row index to include in the sum (0..end_idx).
        alpha (float): Controls how quickly weight decays with distance.
                       Larger alpha => more penalty for being far from end_idx.

    Returns:
        torch.Tensor: A single 1D tensor of shape (vocab_size) with the weighted sum.
    """
    device = logits.device
    # Indices 0..end_idx
    indices = torch.arange(end_idx + 1, device=device, dtype=torch.float32)

    # Distance from end_idx for each index
    dist = end_idx - indices  # or use torch.abs(end_idx - indices) if you prefer absolute distance

    # Exponential decay weighting
    weights = torch.exp(-alpha * dist)

    # Multiply each row by its corresponding weight
    weighted_rows = logits[:end_idx+1] * weights.unsqueeze(-1)  # (end_idx+1, vocab_size)

    # Normalize by sum of weights (scalar)
    return weighted_rows.sum(dim=0) / weights.sum()


def transform_logits_across_batch(logits, alpha=1.0):
    """
    Transform `logits` (shape = (batch_size, vocab_size)) so that
    output[i] is the weighted sum of rows [0..i], giving higher weight
    to rows closer to i.

    Args:
        logits (torch.Tensor): 2D tensor of shape (batch_size, vocab_size).
        alpha (float): Weight-decay factor in the exponential weight formula.

    Returns:
        torch.Tensor: 2D tensor of the same shape, transformed row by row.
    """
    batch_size, vocab_size = logits.shape
    results = []

    for i in range(batch_size):
        row_weighted_sum = weighted_sum_rows(logits, i, alpha=alpha)
        results.append(row_weighted_sum)

    return torch.stack(results, dim=0)

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, device='cuda'):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}')
    for batch in pbar:
        video_feats = batch['video'].to(device)  # (B, T, 3, 224, 224)
        audio_feats = batch['audio'].to(device)  # (B, n_mels, A_time)
        text_tokens = batch['text'].to(device)   # (B, seq_len)

        optimizer.zero_grad()
        logits = model(video_feats, audio_feats, text_tokens)  # (B, seq_len, vocab_size)

        curr_loss = 0
        for sent, logit in zip(text_tokens, logits):
            curr_loss += criterion(transform_logits_across_batch(logit[:-1]), sent[1:])
        
        curr_loss.backward()
        optimizer.step()

        total_loss += curr_loss.item()
    return total_loss / len(dataloader)

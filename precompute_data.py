import os
import re
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from decord import VideoReader
from tqdm.autonotebook import tqdm
import numpy as np
import torchaudio

##########################################
# 1. PARSE CHAT FILE & BUILD VOCAB
##########################################

def parse_time_to_seconds(time_string):
    """Convert 'HH:MM:SS' or 'MM:SS' to integer total seconds."""
    parts = time_string.split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        raise ValueError(f"Unexpected time format: {time_string}")
    return hours * 3600 + minutes * 60 + seconds

def parse_chat_line(line):
    """
    Lines look like: [0:00:10] StreamElements: dorozea is now live! ...
    Returns (timestamp_s, speaker, comment).
    """
    match = re.match(r'^\[(.*?)\]\s*(.*?):\s*(.*)$', line.strip())
    if not match:
        return None
    time_str = match.group(1)
    speaker = match.group(2)
    comment = match.group(3)
    time_s = parse_time_to_seconds(time_str)
    return (time_s, speaker, comment)

def load_chat(chat_file):
    """
    Reads chat file and returns a list of (timestamp_s, speaker, comment).
    """
    entries = []
    with open(chat_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_chat_line(line)
            if parsed:
                entries.append(parsed)
    return entries

def build_vocabulary(chat_file, min_freq=1, max_size=None, special_tokens=None):
    """
    Build a word->index vocabulary from the chat data.
    """
    chat_entries = load_chat(chat_file)
    counter = Counter()
    for (_, _, comment) in chat_entries:
        tokens = comment.split()
        for tok in tokens:
            counter[tok.lower()] += 1

    sorted_by_freq = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    filtered = [(word, freq) for (word, freq) in sorted_by_freq if freq >= min_freq]

    if max_size is not None and len(filtered) > max_size:
        filtered = filtered[:max_size]

    vocab_tokens = [w for (w, _) in filtered]

    if special_tokens is None:
        special_tokens = []
    vocab_tokens = special_tokens + vocab_tokens

    word2idx = {}
    idx2word = []
    for i, w in enumerate(vocab_tokens):
        word2idx[w] = i
        idx2word.append(w)

    return word2idx, idx2word

def tokenize_comment(comment, word2idx, unk_token='<UNK>', eos_token='<EOS>'):
    tokens = comment.split()
    indices = []
    indices.append(word2idx['<SOS>'])
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in word2idx:
            indices.append(word2idx[tok_lower])
        else:
            indices.append(word2idx[unk_token])
    
    # Append <EOS> if it exists in your vocab
    if eos_token in word2idx:
        indices.append(word2idx[eos_token])
    
    return indices


##########################################
# 2. VIDEO & AUDIO EXTRACTION
##########################################

transform_fn = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_frames_decord(video_path, start_sec, end_sec, fps=1):
    """
    Extract frames using decord from 'start_sec' to 'end_sec' at 'fps' frames per second.
    Returns a (T, 3, 64, 64) tensor (matches original processing).
    """
    try:
        # ctx = decord.gpu(0) if torch.cuda.is_available() else decord.cpu(0)
        # vr = VideoReader(video_path, ctx=ctx)
        vr = VideoReader(video_path)
    except:
        raise IOError(f"Could not open video {video_path}")

    # Get video metadata
    video_fps = vr.get_avg_fps()
    if video_fps <= 0:
        video_fps = 30.0  # Fallback to standard FPS

    # Calculate frame indices
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    end_frame = min(end_frame, len(vr) - 1)  # Ensure we don't exceed video length

    if start_frame > end_frame:
        return torch.empty(0)

    total_duration = end_sec - start_sec
    num_frames = int(total_duration * fps)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=np.int32)
    indices = np.clip(indices, 0, len(vr)-1)

    # Batch read and process frames
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) in RGB
    
    frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = T.functional.resize(frames_tensor, (224, 224))  # Batch resize
    frames_tensor = T.functional.normalize(
        frames_tensor, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    return frames_tensor

def extract_audio_snippet(start_sec, end_sec, waveform, mel_transform, TARGET_SR):
    """Process audio 4-10x faster using pre-loaded data."""
    # Calculate samples directly on GPU
    start_sample = int(start_sec * TARGET_SR)
    end_sample = int(end_sec * TARGET_SR)
    end_sample = min(end_sample, waveform.shape[1])

    # Extract snippet (GPU tensor, no copy)
    snippet = waveform[:, start_sample:end_sample]
    
    # Process mel spectrogram on GPU
    mel_spec = mel_transform(snippet)
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Convert to CPU only if needed (keep on GPU for training)
    return mel_spec.mean(dim=0) if mel_spec.shape[0] > 1 else mel_spec.squeeze(0)

def precompute_features(video_path, audio_path, chat_file, output_dir):
    """Run this once before training to cache features."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load chat data and video/audio files once
    chat_entries = load_chat(chat_file)
    waveform, original_sr = torchaudio.load(audio_path)  # Load once, reuse

    # 2. Pre-resample if needed (do this once)
    TARGET_SR = 16000
    if original_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(original_sr, TARGET_SR)
        waveform = resampler(waveform)
        
    # 3. Move to GPU and cache
    waveform = waveform.to(device)
    
    # 4. Pre-initialize Mel transform on GPU
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_mels=64,
        n_fft=1024,          # Reduced from default 2048 for speed
        hop_length=256,       # Increased from default 512 for speed
    ).to(device)
    
    for idx, (time_s, _, _) in enumerate(tqdm(chat_entries, desc="Data preparation")):
        # Video processing
        start_sec = max(0, time_s - 10)
        end_sec = time_s
        frames = extract_frames_decord(video_path, start_sec, end_sec, fps=1)
        torch.save(frames, os.path.join(output_dir, f"vid_{idx}.pt"))
        
        # Audio processing
        audio_snippet = extract_audio_snippet(start_sec, end_sec, waveform, mel_transform, TARGET_SR)
        torch.save(audio_snippet, os.path.join(output_dir, f"aud_{idx}.pt"))


##########################################
# 3. DATASET & DATALOADER
##########################################

class TwitchCommentDataset(Dataset):
    def __init__(self, cache_dir, chat_file, word2idx):
        self.cache_dir = cache_dir
        self.chat_entries = load_chat(chat_file)
        self.word2idx = word2idx

    def __getitem__(self, idx):
        # Load precomputed features
        video_frames = torch.load(os.path.join(self.cache_dir, f"vid_{idx}.pt"))
        audio_mel = torch.load(os.path.join(self.cache_dir, f"aud_{idx}.pt"))
        
        # Text processing remains the same
        _, _, comment = self.chat_entries[idx]
        text_tokens = tokenize_comment(comment, self.word2idx)
        
        return {
            'video': video_frames,
            'audio': audio_mel,
            'text': torch.LongTensor(text_tokens),
            'time': self.chat_entries[idx][0],
            'speaker': self.chat_entries[idx][1]
        }

    
    def __len__(self):
        return len(self.chat_entries)

def my_collate_fn(batch):
    """
    Pad frames (time dim), pad audio (time dim), and pad text (token dim).
    """
    max_vid_frames = max(x['video'].shape[0] for x in batch)
    max_aud_time   = max(x['audio'].shape[1] for x in batch)
    max_text_len   = max(x['text'].shape[0] for x in batch)

    videos = []
    audios = []
    texts  = []
    times  = []
    speakers = []

    for item in batch:
        vid = item['video']   # (Tv, 3, 224, 224)
        aud = item['audio']   # (n_mels, Ta)
        txt = item['text']    # (Tt,)

        # pad video
        pad_vid_frames = max_vid_frames - vid.shape[0]
        if pad_vid_frames > 0:
            pad_shape = (pad_vid_frames, 3, 224, 224)
            vid_pad = torch.zeros(pad_shape, dtype=vid.dtype)
            vid_padded = torch.cat([vid, vid_pad], dim=0)
        else:
            vid_padded = vid

        # pad audio
        pad_aud_time = max_aud_time - aud.shape[1]
        if pad_aud_time > 0:
            pad_shape = (aud.shape[0], pad_aud_time)
            aud_pad = torch.zeros(pad_shape, dtype=aud.dtype)
            aud_padded = torch.cat([aud, aud_pad], dim=1)
        else:
            aud_padded = aud

        # pad text
        pad_text_len = max_text_len - txt.shape[0]
        if pad_text_len > 0:
            pad_text = torch.zeros((pad_text_len,), dtype=txt.dtype)
            txt_padded = torch.cat([txt, pad_text], dim=0)
        else:
            txt_padded = txt

        videos.append(vid_padded.unsqueeze(0))
        audios.append(aud_padded.unsqueeze(0))
        texts.append(txt_padded.unsqueeze(0))

        times.append(item['time'])
        speakers.append(item['speaker'])

    videos = torch.cat(videos, dim=0)  # (B, max_vid_frames, 3, 224, 224)
    audios = torch.cat(audios, dim=0)  # (B, n_mels, max_aud_time)
    texts  = torch.cat(texts, dim=0)   # (B, max_text_len)

    return {
        'video': videos,
        'audio': audios,
        'text': texts,
        'time': times,
        'speaker': speakers
    }

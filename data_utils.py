# data_utils.py
import os
import re
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from collections import Counter

########################
# Chat & Vocab
########################

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
    Returns (time_s, speaker, comment).
    """
    match = re.match(r'^\[(.*?)\]\s*(.*?):\s*(.*)$', line.strip())
    if not match:
        return None
    time_str = match.group(1)
    speaker  = match.group(2)
    comment  = match.group(3)
    time_s   = parse_time_to_seconds(time_str)
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
    """
    Adds <SOS> at the start, splits the comment, and appends <EOS> at the end.
    """
    tokens = comment.split()
    indices = []
    # Insert <SOS> if it is in the vocab
    if '<SOS>' in word2idx:
        indices.append(word2idx['<SOS>'])
    
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in word2idx:
            indices.append(word2idx[tok_lower])
        else:
            indices.append(word2idx[unk_token])
    
    # Append <EOS> if it exists
    if eos_token in word2idx:
        indices.append(word2idx[eos_token])
    
    return indices

########################
# Dataset & Collate
########################

class TwitchCommentDataset(Dataset):
    def __init__(self, cache_dir, chat_file, word2idx):
        """
        cache_dir: path where .pt files for 'vid_{i}.pt' and 'aud_{i}.pt' are stored
        chat_file: the .txt with chat lines
        word2idx: vocab dictionary
        """
        self.cache_dir = cache_dir
        self.chat_entries = load_chat(chat_file)
        self.word2idx = word2idx

    def __len__(self):
        return len(self.chat_entries)

    def __getitem__(self, idx):
        video_path = os.path.join(self.cache_dir, f"vid_{idx}.pt")
        audio_path = os.path.join(self.cache_dir, f"aud_{idx}.pt")

        video_frames = torch.load(video_path)  # shape: (Tv, 3, 224, 224)
        audio_mel = torch.load(audio_path)     # shape: (n_mels, time)

        _, _, comment = self.chat_entries[idx]
        text_tokens = tokenize_comment(comment, self.word2idx)

        return {
            'video': video_frames,
            'audio': audio_mel,
            'text': torch.LongTensor(text_tokens),
            'time': self.chat_entries[idx][0],
            'speaker': self.chat_entries[idx][1]
        }

def my_collate_fn(batch):
    """
    Pad frames (video), pad audio, and pad text.
    """
    max_vid_frames = max(x['video'].shape[0] for x in batch)
    max_aud_time   = max(x['audio'].shape[1] for x in batch)
    max_text_len   = max(x['text'].shape[0] for x in batch)

    videos, audios, texts = [], [], []
    times, speakers = [], []

    for item in batch:
        vid = item['video']
        aud = item['audio']
        txt = item['text']

        # Pad video
        pad_vid_frames = max_vid_frames - vid.shape[0]
        if pad_vid_frames > 0:
            pad_shape = (pad_vid_frames, 3, 224, 224)
            vid_pad = torch.zeros(pad_shape, dtype=vid.dtype)
            vid = torch.cat([vid, vid_pad], dim=0)

        # Pad audio
        pad_aud_time = max_aud_time - aud.shape[1]
        if pad_aud_time > 0:
            pad_shape = (aud.shape[0], pad_aud_time)
            aud_pad = torch.zeros(pad_shape, dtype=aud.dtype)
            aud = torch.cat([aud, aud_pad], dim=1)

        # Pad text
        pad_text_len = max_text_len - txt.shape[0]
        if pad_text_len > 0:
            txt_pad = torch.zeros((pad_text_len,), dtype=txt.dtype)
            txt = torch.cat([txt, txt_pad], dim=0)

        videos.append(vid.unsqueeze(0))
        audios.append(aud.unsqueeze(0))
        texts.append(txt.unsqueeze(0))
        times.append(item['time'])
        speakers.append(item['speaker'])

    videos = torch.cat(videos, dim=0)  # (B, max_vid, 3, 224, 224)
    audios = torch.cat(audios, dim=0)  # (B, n_mels, max_aud_time)
    texts  = torch.cat(texts, dim=0)   # (B, max_text_len)

    return {
        'video': videos,
        'audio': audios,
        'text': texts,
        'time': times,
        'speaker': speakers
    }

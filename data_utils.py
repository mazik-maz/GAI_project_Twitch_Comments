# data_utils.py
import os
import re
import torch
from torch.utils.data import Dataset
from collections import Counter

########################
# Chat & Vocab
########################

def parse_time_to_seconds(time_string):
    """Convert 'HH:MM:SS' or 'MM:SS' to integer total seconds."""
    parts = time_string.split(':')
    parts = [int(p.split(',')[0]) for p in parts]
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
    Lines look like: [0:00:10] StreamElements: ...
    Returns (time_s, speaker, comment).
    """
    match = re.match(r'^\[(.*?)\]\s+<(.*?)>\s+(.*)$', line.strip())
    if not match:
        return None
    time_str = match.group(1)
    speaker  = match.group(2)
    comment  = match.group(3)
    time_s   = parse_time_to_seconds(time_str)
    return (time_s, speaker, comment)

def load_chat(chat_file):
    """
    Reads chat file (e.g. .irc) and returns a list of (timestamp_s, speaker, comment).
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
    """
    Expects that 'preprocess_files(...)' saved 
    each snippet to: output_dir/out_{idx}.pt 
    with keys:
      'video'  -> shape either (dim,) or (T, dim)
      'audio'  -> shape either (dim,) or (A_time, dim)
      'text'   -> shape (seq_len,)
      'time'   -> float (timestamp)
      'speaker'-> str
    This dataset will load them all, tokenizing text
    if not already done (or you can store tokens directly).
    """
    def __init__(self, cache_dir, chat_file, word2idx):
        """
        cache_dir: directory containing 'out_{i}.pt'
        chat_file: your .irc with chat lines
        word2idx:  vocab dictionary
        """
        self.cache_dir = cache_dir
        self.chat_entries = load_chat(chat_file)
        self.word2idx = word2idx

    def __len__(self):
        return len(self.chat_entries)

    def __getitem__(self, idx):
        """
        We'll load 'out_{idx}.pt' which is the file created
        by the new preprocess pipeline. That file has:
           { 'video':..., 'audio':..., 'text':..., 'time':..., 'speaker':... }
        """
        out_path = os.path.join(self.cache_dir, f"out_{idx}.pt")
        sample_dict = torch.load(out_path)

        video_emb = sample_dict['video']  # e.g. shape (2048,) or (T, 2048)
        audio_emb = sample_dict['audio']  # e.g. shape (128,) or (A_time,128)
        text_tokens = sample_dict['text'] # e.g. shape (seq_len,)

        # If you stored raw text instead of tokens, you could re-tokenize here:
        # text_tokens = tokenize_comment(sample_dict['text'], self.word2idx)

        return {
            'video': video_emb,
            'audio': audio_emb,
            'text': text_tokens,
            'time': sample_dict['time'],
            'speaker': sample_dict['speaker']
        }


def my_collate_fn(batch):
    """
    This handles variable-length video embeddings, audio embeddings, and text tokens.
    We'll pad along the time dimension for video/audio if they're sequences.
    If they're single embeddings (dim,) we treat them as (1, dim).
    """
    # Determine the max length of video embeddings (if they're sequences)
    # or we consider them (1, dim) if shape is (dim,).
    max_vid_len = 0
    max_aud_len = 0
    max_text_len = 0

    # We collect the dimension
    # e.g. if 'video' is shape (T, 2048) or (1, 2048)
    for item in batch:
        vid_shape = item['video'].shape
        if len(vid_shape) == 1:
            # (dim,) => treat as length=1
            vid_len = 1
        else:
            vid_len = vid_shape[0]
        if vid_len > max_vid_len:
            max_vid_len = vid_len
        
        aud_shape = item['audio'].shape
        if len(aud_shape) == 1:
            aud_len = 1
        else:
            aud_len = aud_shape[0]
        if aud_len > max_aud_len:
            max_aud_len = aud_len
        
        txt_len = item['text'].shape[0]
        if txt_len > max_text_len:
            max_text_len = txt_len

    # Prepare storage
    # We'll assume 'video' has shape (B, max_vid_len, feature_dim)
    # We'll assume 'audio' has shape (B, max_aud_len, feature_dim)
    # We'll assume 'text' has shape (B, max_text_len)
    video_list = []
    audio_list = []
    text_list  = []
    times      = []
    speakers   = []

    for item in batch:
        vid = item['video']
        aud = item['audio']
        txt = item['text']

        # Ensure video is 2D: (vid_len, dim)
        if len(vid.shape) == 1:
            # (dim,) => reshape to (1, dim)
            vid = vid.unsqueeze(0)
        vid_len, vid_dim = vid.shape
        pad_vid_len = max_vid_len - vid_len
        if pad_vid_len > 0:
            pad_shape = (pad_vid_len, vid_dim)
            vid_pad = torch.zeros(pad_shape, dtype=vid.dtype)
            vid = torch.cat([vid, vid_pad], dim=0)

        # Ensure audio is 2D: (aud_len, dim)
        if len(aud.shape) == 1:
            aud = aud.unsqueeze(0)
        aud_len, aud_dim = aud.shape
        pad_aud_len = max_aud_len - aud_len
        if pad_aud_len > 0:
            pad_shape = (pad_aud_len, aud_dim)
            aud_pad = torch.zeros(pad_shape, dtype=aud.dtype)
            aud = torch.cat([aud, aud_pad], dim=0)

        # Pad text
        txt_len = txt.shape[0]
        pad_txt_len = max_text_len - txt_len
        if pad_txt_len > 0:
            pad_txt = torch.zeros((pad_txt_len,), dtype=txt.dtype)
            txt = torch.cat([txt, pad_txt], dim=0)

        video_list.append(vid.unsqueeze(0))
        audio_list.append(aud.unsqueeze(0))
        text_list.append(txt.unsqueeze(0))

        times.append(item['time'])
        speakers.append(item['speaker'])

    # stack
    videos = torch.cat(video_list, dim=0)  # (B, max_vid_len, vid_dim)
    audios = torch.cat(audio_list, dim=0)  # (B, max_aud_len, aud_dim)
    texts  = torch.cat(text_list, dim=0)   # (B, max_text_len)

    return {
        'video': videos,    # (B, max_vid_len, vid_dim)
        'audio': audios,    # (B, max_aud_len, aud_dim)
        'text': texts,      # (B, max_text_len)
        'time': times,
        'speaker': speakers
    }

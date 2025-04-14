# Multimodal Transformer for Twitch Chat Generation

This repository contains an end-to-end pipeline for:
1. **Downloading** Twitch VODs and chat logs.
2. **Preprocessing** videos and audio to extract embeddings with ResNet (for video) and VGGish (for audio).
3. **Building a vocabulary** from Twitch chat logs.
4. **Creating a PyTorch Dataset** that pairs the extracted embeddings with chat tokens.
5. **Training** a multimodal Transformer model to predict the chat text from video+audio inputs.
6. **Performing inference** with the trained model (e.g., generating new comments token-by-token).

Below you will find a walkthrough of the repository structure, installation requirements, usage examples, and additional details on how each script works.

---

## Repository Structure

```
.
├── data_utils.py
├── inference.py
├── model.py
├── preprocess.py
├── train.py
├── download_pipeline.ipynb
├── test_transformer_model.ipynb
├── requirements.txt (?) <- (You can create one listing your Python dependencies)
└── README.md
```

### 1. `data_utils.py`
- **Core functionalities** for:
  - Parsing Twitch chat logs.
  - Building a word-level vocabulary from chat lines.
  - Tokenizing text comments with special tokens (`<SOS>`, `<EOS>`, etc.).
  - Defining a custom PyTorch `Dataset` (`TwitchCommentDataset`) that loads precomputed `.pt` files (video embeddings, audio embeddings, tokenized text).
  - Providing a custom collate function (`my_collate_fn`) to pad variable-length inputs for batches.

### 2. `inference.py`
- **Inference functions** for the trained multimodal model.
- Includes a function (`generate_comment`) that:
  1. Encodes video/audio features via the model’s encoders.
  2. Decodes the text tokens step-by-step, applying a causal mask.
  3. Supports top-k sampling and temperature-based sampling.

### 3. `model.py`
- **Definition of the multimodal Transformer model**.
- Consists of:
  - `VideoEncoder` and `AudioEncoder` classes (each a TransformerEncoder).
  - A `CommentDecoder` class (a TransformerDecoder with embedding layers for text tokens).
  - A `MultiModal` wrapper that:
    - Projects video and audio features to a common embedding dimension (`d_model`).
    - Encodes them separately.
    - Averages (pools) each encoder's output.
    - Concatenates them as the “memory” for the text decoder.
    - Finally decodes text tokens, returning logits over the vocabulary.

### 4. `preprocess.py`
- **Preprocessing script** for:
  1. Chunkifying the entire video into fixed-duration segments (e.g., 10-second chunks).
     - Sampling frames within each chunk at a specified FPS, extracting ResNet features, and averaging them into a single 2048-d vector.
  2. Chunkifying the entire audio similarly, extracting 128-d VGGish embeddings.
  3. Parsing chat lines to map each chat message to the appropriate chunk (based on timestamp).
  4. Storing everything in `.pt` files, one per chat line.

### 5. `train.py`
- **Training utilities** containing:
  - A custom training loop (`train_one_epoch`) that:
    - Retrieves the `video`, `audio`, and `text` data from a batch.
    - Passes them through the model.
    - Uses a custom loss function (can be standard CrossEntropy or a weighted transformation of logits).
    - Performs backpropagation and updates model parameters.

### 6. `download_pipeline.ipynb`
- **Notebook** demonstrating how to:
  - Use `twitch-dl` and `tdh-tcd` to **download** the last N VODs and chat logs from a Twitch streamer.
  - Extract the streamer’s VOD IDs and do any needed format conversions.
- This is a utility pipeline to obtain your data (`.mkv` files for video and `.irc` or JSON files for chat).

### 7. `test_transformer_model.ipynb`
- **Example notebook** walking through the entire workflow:
  1. Installing dependencies.
  2. Building vocabulary from an `.irc` chat file.
  3. Running `preprocess_files` to create `.pt` embeddings for each chat line.
  4. Creating a PyTorch `DataLoader`.
  5. Instantiating and training the `MultiModal` Transformer model.
  6. Saving/loading model weights.
  7. Running inference on a sample to generate text tokens.

---

## Getting Started

### 1. Environment Setup

1. **Clone** the repository:
   ```bash
   git clone https://github.com/username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create** and **activate** a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # on Linux/Mac
   venv\Scripts\activate          # on Windows
   ```

3. **Install** dependencies:
   ```bash
   pip install --upgrade pip
   pip install torch torchaudio torchvision tqdm decord soundfile resampy
   # or if there's a requirements.txt:
   # pip install -r requirements.txt
   ```
   Additional packages:
   - `twitch-dl` and `tdh-tcd` for downloading Twitch VODs/chat.  
   - `ffmpeg` installed on your system (if you need to convert `.mkv` -> `.wav` or extract frames, etc.).

### 2. Downloading Data

- If you **already have** `.mkv` and `.irc` files, skip this step.
- To **download from Twitch**:
  1. Rename or adapt [`download_pipeline.ipynb`](./download_pipeline.ipynb).
  2. Set `streamer_name`, `num_streams_to_download`, `client_id`, etc.
  3. Run cells to fetch the VOD IDs, then download the videos and chat logs.

You should end up with files like:
```
2424877187.mkv
2424877187.irc
```
(Matching IDs for video + chat.)

### 3. Preprocessing

- Use the script/notebook that calls [`preprocess.py`](./preprocess.py).  
- You must specify:
  - A **video file** (e.g., `2424877187.mkv`)
  - An **audio file** (e.g., `2424877187.wav`) if you separate audio from the video. If the `.mkv` has audio, you might extract `.wav` via `ffmpeg`.
  - A **chat file** (e.g., `2424877187.irc`)
  - An **output directory** to store `.pt` embeddings
  - A **vocabulary** dictionary (`word2idx`) if you want to store tokenized text

Example call:
```python
from preprocess import preprocess_files
from data_utils import build_vocabulary

# Build vocabulary from a chat file
special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
word2idx, idx2word = build_vocabulary(
    chat_file="data/2424877187.irc",
    min_freq=1,
    max_size=5000,
    special_tokens=special_tokens
)

# Preprocess and save .pt files
preprocess_files(
    video_path="data/2424877187.mkv",
    audio_path="data/2424877187.wav",
    chat_file="data/2424877187.irc",
    output_dir="precomputed_data",
    snippet_duration=10.0,   # in seconds
    sample_rate=16000,       # for audio
    fps_for_sampling=1.0,    # frames/sec for video
    word2idx=word2idx
)
```
This produces:
```
precomputed_data/
   out_0.pt
   out_1.pt
   ...
   out_N.pt
```
Each `.pt` file contains the keys: `{ "video", "audio", "text", "time", "speaker" }`.

### 4. Training

- **Create** a dataset and dataloader from the `.pt` files and chat lines:
  ```python
  from data_utils import TwitchCommentDataset, my_collate_fn
  from torch.utils.data import DataLoader

  dataset = TwitchCommentDataset(
      cache_dir="precomputed_data",
      chat_file="data/2424877187.irc",
      word2idx=word2idx
  )

  dataloader = DataLoader(
      dataset,
      batch_size=4,
      shuffle=True,
      collate_fn=my_collate_fn
  )
  ```
- **Instantiate** the model in [`model.py`](./model.py):
  ```python
  from model import MultiModal
  import torch
  import torch.nn as nn
  import torch.optim as optim

  vocab_size = len(word2idx)
  model = MultiModal(
      vocab_size=vocab_size,
      d_model=512,
      video_feature_dim=2048,
      audio_feature_dim=128,
      nhead=8,
      num_encoder_layers=4,
      num_decoder_layers=4,
      dim_feedforward=2048,
      dropout=0.1
  ).to(device)
  ```
- **Train** with [`train.py`](./train.py):
  ```python
  from train import train_one_epoch

  criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  num_epochs = 10

  for epoch in range(1, num_epochs + 1):
      avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, epoch, device='cuda')
      print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

  # Save
  torch.save(model.state_dict(), "multimodal_transformer.pth")
  ```

### 5. Inference

- **Load** the trained model and run inference using [`inference.py`](./inference.py):
  ```python
  from inference import generate_comment

  # Reload model
  model = MultiModal(
      vocab_size=vocab_size,
      d_model=512,
      video_feature_dim=2048,
      audio_feature_dim=128,
      ...
  ).to(device)
  model.load_state_dict(torch.load("multimodal_transformer.pth"))

  # Suppose we have some sample video/audio embedding
  sample_data = dataset[500]  # random index
  video_emb = sample_data["video"]
  audio_emb = sample_data["audio"]

  start_tok = word2idx["<SOS>"]
  end_tok   = word2idx["<EOS>"]

  gen_tokens = generate_comment(
      model=model,
      video_tensor=video_emb,
      audio_tensor=audio_emb,
      start_token_idx=start_tok,
      end_token_idx=end_tok,
      max_len=20,
      device='cuda',
      temperature=1.0,
      top_k=5
  )

  # Decode tokens
  idx2word_map = {v: k for k, v in word2idx.items()}
  decoded = [idx2word_map.get(t, "<UNK>") for t in gen_tokens]
  print(" ".join(decoded))
  ```

---

## Tips and Notes

1. **Audio Extraction**: If your video `.mkv` still contains audio, you can either:
   - Pass the same file to the video chunking function or
   - Extract a `.wav` first (`ffmpeg -i input.mkv -vn -acodec pcm_s16le output.wav`) and then pass `output.wav` to the audio chunking.

2. **Chat Format**: The `.irc` format is expected to have lines like:
   ```
   [0:00:10] <username> This is a comment
   [0:00:15] <anotherUser> Another comment
   ```
   Adjust the regex in `data_utils.py` if your chat format differs.

3. **Vocabulary Size**: Adjust `min_freq` or `max_size` in `build_vocabulary()` to control the total number of tokens. Words not in the vocabulary map to `<UNK>`.

4. **Performance**: For large VODs or numerous lines, consider distributing the workload. The chunk-based approach is designed to avoid re-encoding the entire video/audio for each line.

5. **Advanced**:
   - The current approach does a **global average pool** over time for both video and audio. You could keep the full sequence and let the decoder cross-attend to it, which might increase performance but also memory usage.
   - In training, we have a custom weighting function in `train_one_epoch` that can be replaced with a more standard approach (e.g., `CrossEntropyLoss` on each time step). Adjust as needed.

---

## Contributing

Feel free to open issues or pull requests if you find bugs, have questions, or want to add features (e.g., different encoding strategies, better text decoding logic, etc.).

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [twitch-dl](https://github.com/ihabunek/twitch-dl) for convenient video downloads.
- [tdh-tcd](https://github.com/HclX/tdh-tcd) for chat log download.
- [ResNet50](https://pytorch.org/vision/stable/models.html#id12) and [TorchVGGish](https://github.com/harritaylor/torchvggish) for feature extraction.

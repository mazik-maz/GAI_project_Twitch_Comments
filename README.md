# Twitch Multimodal LSTM

This repository demonstrates a multimodal LSTM model that processes video (frames) + audio (mel-spectrograms) to generate Twitch chat comments.

## Model Operation

Our multimodal LSTM model processes video frames and audio spectrograms to generate textual comments. Specifically, we use two specialized LSTM encoders—one for video and one for audio. First, we flatten each batch of video frames (after extracting and resizing them) into a shape suitable for an LSTM input, then pass them through the VideoEncoder. Meanwhile, the raw audio waveform is transformed into a Mel-Spectrogram and fed into the AudioEncoder, which also uses an LSTM to capture temporal features from the spectrogram frames. Each encoder outputs a hidden state representing the learned context of its respective modality. These two hidden states are concatenated and passed through a fusion layer to form the initial hidden and cell states of the CommentDecoder. In turn, the decoder—another LSTM—generates output tokens one step at a time, ultimately producing a chat comment. This decoder can be run with various decoding strategies (greedy or sampling) to yield natural text responses. The presence of <SOS> and <EOS> tokens helps the model learn clear start and end boundaries for each generated comment, improving coherence and stopping behavior.

## File Structure

- **data_utils.py**  
  Contains logic for parsing chat files, building vocabulary, and defining the `TwitchCommentDataset`.

- **preprocess.py**  
  Extracts frames from MP4 and audio WAV data, produces `.pt` files with precomputed features.

- **model.py**  
  Defines the LSTM-based architecture for VideoEncoder, AudioEncoder, and CommentDecoder (combined in MultiModalLSTM).

- **train.py**  
  Defines the training loop (`train_one_epoch`).

- **inference.py**  
  Provides a `generate_comment` function with top-k sampling + temperature-based decoding.

- **notebook_demo.ipynb**  
  A Jupyter notebook showing how to use all these modules together. It demonstrates building the vocab, preprocessing data, training or loading a model, and generating comments.

- **my_multimodal_model.pth**  
  Saved model weights from a trained run. You can load these in the notebook or at inference time.

## Usage

1. **Install Requirements**  
   - `pip install torch torchvision torchaudio decord tqdm`

2. **Precompute Features**  
   - Adjust the paths in `notebook_demo.ipynb` or call `precompute_features(...)` from your own script.

3. **Train**  
   - Run `notebook_demo.ipynb`, set `train_model=True`, or call your own training loop in `train.py`.

4. **Inference**  
   - Use `inference.py` to load the model and run `generate_comment`.

## Changes Made

- We added `<SOS>` at the start and `<EOS>` at the end of each comment.  
- We introduced top-k sampling with temperature in `generate_comment`.  
- We reorganized code into separate modules for clarity and easier maintenance.

## Links to data

- Model trained for 100 epochs: https://www.kaggle.com/datasets/mazikil/twitch-model-4
- Precomputed video and audio data: https://www.kaggle.com/datasets/mazikil/precompute-data-gai
- Data of 28 minutes stream (video, audio, chat): https://www.kaggle.com/datasets/mazikil/dorozeadata


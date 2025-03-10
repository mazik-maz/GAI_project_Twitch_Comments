# model.py
import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=1):
        super(VideoEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, (h, c) = self.lstm(x)
        return (h, c)

class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=1):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, A_time, input_dim)
        out, (h, c) = self.lstm(x)
        return (h, c)

class CommentDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, num_layers=1):
        super(CommentDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens, hidden, cell):
        # input_tokens: (B, seq_len)
        embedded = self.embedding(input_tokens)
        out, (h, c) = self.lstm(embedded, (hidden, cell))
        logits = self.output_fc(out)  # (B, seq_len, vocab_size)
        return logits, (h, c)

class MultiModalLSTM(nn.Module):
    def __init__(self, vocab_size, video_feature_dim, audio_feature_dim, hidden_dim=512):
        super(MultiModalLSTM, self).__init__()
        self.video_encoder = VideoEncoder(video_feature_dim, hidden_dim)
        self.audio_encoder = AudioEncoder(audio_feature_dim, hidden_dim)
        self.decoder       = CommentDecoder(vocab_size, embed_dim=300, hidden_dim=hidden_dim)

        self.fuse_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fuse_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, video_feats, audio_feats, text_inputs):
        (vh, vc) = self.video_encoder(video_feats)
        (ah, ac) = self.audio_encoder(audio_feats)

        fused_h = torch.cat([vh[-1], ah[-1]], dim=-1)  # (B, hidden_dim*2)
        fused_c = torch.cat([vc[-1], ac[-1]], dim=-1)

        fused_h = self.fuse_h(fused_h).unsqueeze(0)  # (1, B, hidden_dim)
        fused_c = self.fuse_c(fused_c).unsqueeze(0)

        logits, _ = self.decoder(text_inputs, fused_h, fused_c)
        return logits

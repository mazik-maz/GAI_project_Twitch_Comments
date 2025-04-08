# model.py
import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        """
        d_model: embedding dimension
        nhead: number of attention heads
        num_layers: how many TransformerEncoder layers
        dim_feedforward: feedforward network size
        dropout: dropout rate
        """
        super().__init__()
        # We define a single shared encoder (nn.TransformerEncoder) with 'num_layers' layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # to support (B, T, D) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vid_embeds, src_key_padding_mask=None):
        """
        vid_embeds: shape (B, T, d_model)
        src_key_padding_mask: optional (B, T) boolean mask for padding
        Returns: (B, T, d_model)
        """
        # Pass through the Transformer encoder
        out = self.transformer_encoder(vid_embeds, src_key_padding_mask=src_key_padding_mask)
        return out

class AudioEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, aud_embeds, src_key_padding_mask=None):
        """
        aud_embeds: (B, A_time, d_model)
        """
        out = self.transformer_encoder(aud_embeds, src_key_padding_mask=src_key_padding_mask)
        return out

class CommentDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # For tokens => embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        # We also might want positional embeddings for the decoder
        # For simplicity, let's do a small learnable positional embedding:
        self.pos_embedding = nn.Embedding(1000, d_model)  # up to 1000 tokens long, for example

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, vocab_size)

    def forward(self,
                tgt_tokens,
                memory,  # shape (B, S, d_model) from encoders
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        tgt_tokens: (B, tgt_len)
        memory: (B, src_len, d_model) from the combined/fused video+audio
        """
        B, tgt_len = tgt_tokens.shape
        # 1) embed
        tok_emb = self.embedding(tgt_tokens)  # (B, tgt_len, d_model)
        # add positional embeddings
        positions = torch.arange(tgt_len, device=tok_emb.device).unsqueeze(0).expand(B, tgt_len)
        pos_emb = self.pos_embedding(positions)  # (B, tgt_len, d_model)

        # sum them
        dec_input = tok_emb + pos_emb

        # 2) pass through transformer decoder
        # PyTorch's nn.TransformerDecoder expects shape (B, tgt_len, d_model) if batch_first=True
        # We'll provide the masks if needed
        out = self.transformer_decoder(
            dec_input,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (B, tgt_len, d_model)

        # 3) project to vocab
        logits = self.output_fc(out)  # (B, tgt_len, vocab_size)
        return logits

class MultiModal(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        video_feature_dim=2048,
        audio_feature_dim=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        # 1) Project raw frame features -> d_model
        self.video_proj = nn.Linear(video_feature_dim, d_model)
        # 2) Project raw audio features -> d_model
        self.audio_proj = nn.Linear(audio_feature_dim, d_model)

        # 3) Video transformer encoder
        self.video_encoder = VideoEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # 4) Audio transformer encoder
        self.audio_encoder = AudioEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 5) Final fuse: We can do simple concatenation or a cross-attention approach.
        # For simplicity, we'll just concat the final states along time dimension
        # or we'll do something simpler: let's just produce an average pooling of each
        # then combine them. (Better solutions exist, but let's keep it straightforward.)
        self.video_pool = nn.AdaptiveAvgPool1d(1)  # we can average pool over T
        self.audio_pool = nn.AdaptiveAvgPool1d(1)

        # 6) Text transformer decoder
        self.text_decoder = CommentDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(
        self,
        video_feats,        # (B, T, video_feature_dim)
        audio_feats,        # (B, A_time, audio_feature_dim)
        text_tokens         # (B, seq_len)
    ):
        """
        We'll do a simple approach: pass video_feats -> video_encoder, audio_feats -> audio_encoder,
        do a global average pool on each -> shape (B, 1, d_model) for both video & audio
        then cat them => shape (B, 2, d_model) as memory for the text decoder
        (In a more advanced approach, we might want to keep all time steps for cross-attention.)
        """
        # 1) Project to d_model
        vid_emb = self.video_proj(video_feats)     # (B, T, d_model)
        aud_emb = self.audio_proj(audio_feats)     # (B, A_time, d_model)

        # 2) Encode
        vid_enc = self.video_encoder(vid_emb)      # (B, T, d_model)
        aud_enc = self.audio_encoder(aud_emb)      # (B, A_time, d_model)

        # 3) Simple average pooling for demonstration
        # shape => (B, d_model)
        vid_pooled = vid_enc.mean(dim=1, keepdim=True)  # or adaptive pool
        aud_pooled = aud_enc.mean(dim=1, keepdim=True)

        # 4) Concatenate => (B, 2, d_model)
        memory = torch.cat([vid_pooled, aud_pooled], dim=1)

        # 5) Decode text
        # We can supply no mask for now, or you can create a standard subsequent mask for target tokens
        logits = self.text_decoder(
            tgt_tokens=text_tokens,
            memory=memory
        )  # => (B, seq_len, vocab_size)

        return logits

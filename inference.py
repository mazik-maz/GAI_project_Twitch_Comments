# inference.py
import torch
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular mask of shape (sz, sz),
    with True in the upper triangle (where we want to block attention).
    If using batch_first, adapt accordingly.
    """
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0,1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask

def generate_comment(
    model, 
    video_tensor,       # shape (T, video_dim) or (video_dim,)
    audio_tensor,       # shape (A_time, audio_dim) or (audio_dim,)
    start_token_idx, 
    end_token_idx, 
    max_len=20,
    device='cuda',
    temperature=1.0,
    top_k=5
):
    """
    We'll assume 'model' is the MultiModal (Transformer) that expects
    video_feats => (B, T, video_dim)
    audio_feats => (B, A_time, audio_dim)
    Then does an internal pooling or uses the entire sequence.

    We'll do token-by-token decoding with a causal mask.
    """
    model.eval()
    with torch.no_grad():
        # 1) Add batch dimension for video
        if video_tensor.dim() == 1:
            # e.g. (video_dim,) => (1, video_dim)
            video_tensor = video_tensor.unsqueeze(0)
            # => shape (1, video_dim)
            # want => (1, T=1, video_dim)
            video_tensor = video_tensor.unsqueeze(0)
        elif video_tensor.dim() == 2:
            # e.g. (T, video_dim) => (B=1, T, dim)
            video_tensor = video_tensor.unsqueeze(0)
        else:
            # already 3D? (B, T, dim)? Then we skip
            pass

        # 2) Add batch dimension for audio
        if audio_tensor.dim() == 1:
            # (audio_dim,) => (1, audio_dim) => (1,1,audio_dim)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            # (A_time, audio_dim) => (1,A_time,audio_dim)
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            # already 3D
            pass

        # Move to device
        video_tensor = video_tensor.to(device)
        audio_tensor = audio_tensor.to(device)

        # 3) Forward pass in the model
        # But the model forward(...) also wants 'text_tokens'.
        # We'll do partial: we only want the encoders output
        # then store it as memory for decoding step by step.
        # So we can replicate the same logic as 'forward' but skip text.

        # Specifically, we can replicate the code from model's forward but
        # pass a dummy text or create a function 'encode()' in the model for clarity.
        # For simplicity, let's do inline:

        B, T, video_dim = video_tensor.shape
        _, A_time, audio_dim = audio_tensor.shape

        #  a) project => (B,T,d_model) or (B,A_time,d_model)
        vid_emb = model.video_proj(video_tensor)   # => (1, T, d_model)
        aud_emb = model.audio_proj(audio_tensor)   # => (1, A_time, d_model)

        #  b) encode
        vid_enc = model.video_encoder(vid_emb)     # => (1, T, d_model)
        aud_enc = model.audio_encoder(aud_emb)     # => (1, A_time, d_model)

        #  c) average pool or whichever approach your model uses
        vid_pooled = vid_enc.mean(dim=1, keepdim=True)  # shape (1,1,d_model)
        aud_pooled = aud_enc.mean(dim=1, keepdim=True)  # shape (1,1,d_model)

        memory = torch.cat([vid_pooled, aud_pooled], dim=1)  # (1,2,d_model)

        # Now 'memory' is shape (B=1, S=2, d_model), a valid 3D tensor.

        # 4) Start decoding tokens one by one
        generated_tokens = []
        current_token = torch.tensor([[start_token_idx]], device=device)  # (1,1)

        hidden_seq = current_token  # This will hold all tokens so far

        for step in range(max_len):
            # build the subsequent mask for current sequence length
            tgt_len = hidden_seq.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(device)

            # pass through the model's text_decoder
            # shape => (1, tgt_len, vocab_size)
            logits = model.text_decoder(
                tgt_tokens=hidden_seq,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # we want the last time step => shape (vocab_size,)
            next_logits = logits[0, -1, :]

            # apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # top-k sampling
            if top_k is not None and top_k > 0:
                top_vals, top_inds = torch.topk(next_logits, k=top_k)
                probs = F.softmax(top_vals, dim=-1)
                chosen_idx_in_topk = torch.multinomial(probs, 1)
                next_token = top_inds[chosen_idx_in_topk]
            else:
                # or full sampling
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            token_id = next_token.item()
            generated_tokens.append(token_id)

            if token_id == end_token_idx:
                break

            # append the new token
            next_token = next_token.unsqueeze(0)  # => shape(1,1)
            hidden_seq = torch.cat([hidden_seq, next_token], dim=1)

    return generated_tokens

import torch
import torch.nn.functional as F

def generate_comment(
    model,
    video_tensor,
    audio_tensor,
    start_token_idx,
    end_token_idx,
    max_len=20,
    device='cuda',
    temperature=1.0,
    top_k=5
):
    """
    We'll do naive flatten for video, transpose for audio, then decode.
    Decoding uses top-k sampling with an optional temperature.
    """
    model.eval()
    with torch.no_grad():
        # 1) Preprocess input
        video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, 3, 224, 224)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)  # (1, n_mels, time)
        
        B, T, C, H, W = video_tensor.shape
        video_flat = video_tensor.reshape(B, T, -1)  # (1, T, 3*224*224)
        audio_t = audio_tensor.transpose(1, 2)       # (1, time, n_mels)

        # 2) Encode
        (vh, vc) = model.video_encoder(video_flat)
        (ah, ac) = model.audio_encoder(audio_t)

        fused_h = torch.cat([vh[-1], ah[-1]], dim=-1)  # shape (1, hidden_dim*2)
        fused_c = torch.cat([vc[-1], ac[-1]], dim=-1)

        fused_h = model.fuse_h(fused_h).unsqueeze(0)   # (1, 1, hidden_dim)
        fused_c = model.fuse_c(fused_c).unsqueeze(0)

        # 3) Decoding loop
        generated_tokens = []
        current_token = torch.tensor([[start_token_idx]], device=device)  # shape (1,1)

        hidden = fused_h
        cell   = fused_c
        for _ in range(max_len):
            embed = model.decoder.embedding(current_token)  # (1,1,embed_dim)
            out, (hidden, cell) = model.decoder.lstm(embed, (hidden, cell))
            logits = model.decoder.output_fc(out)  # (1,1,vocab_size)
            logits = logits[0, -1, :]              # shape (vocab_size,)

            # 4) Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # 5) Restrict to top_k
            if top_k is not None and top_k > 0:
                top_values, top_indices = torch.topk(logits, k=top_k)
                # Convert top_values -> probabilities
                probs = F.softmax(top_values, dim=-1)
                # Sample 1 index from top_k
                next_token_id = torch.multinomial(probs, 1)
                # Map back to original token ID
                next_token = top_indices[next_token_id]
            else:
                # fallback: pure sampling from entire vocab
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            token_id = next_token.item()
            generated_tokens.append(token_id)

            if token_id == end_token_idx:
                break

            # Prepare next iteration
            current_token = next_token.unsqueeze(0)  # shape (1,1)

    return generated_tokens

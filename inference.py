# inference.py
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
    Decoding using top-k sampling, with temperature. 
    video_tensor: (T, 3, 224, 224)
    audio_tensor: (n_mels, time)
    """
    model.eval()
    with torch.no_grad():
        video_tensor = video_tensor.unsqueeze(0).to(device)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)

        B, T, C, H, W = video_tensor.shape
        video_flat = video_tensor.reshape(B, T, -1)
        audio_t    = audio_tensor.transpose(1, 2)

        # Encode
        (vh, vc) = model.video_encoder(video_flat)
        (ah, ac) = model.audio_encoder(audio_t)

        fused_h = torch.cat([vh[-1], ah[-1]], dim=-1)
        fused_c = torch.cat([vc[-1], ac[-1]], dim=-1)
        fused_h = model.fuse_h(fused_h).unsqueeze(0)
        fused_c = model.fuse_c(fused_c).unsqueeze(0)

        generated_tokens = []
        current_token = torch.tensor([[start_token_idx]], device=device)  # shape (1,1)

        hidden = fused_h
        cell   = fused_c

        for _ in range(max_len):
            embed = model.decoder.embedding(current_token)
            out, (hidden, cell) = model.decoder.lstm(embed, (hidden, cell))
            logits = model.decoder.output_fc(out)  # (1,1,vocab_size)
            logits = logits[0, -1, :]  # shape (vocab_size,)

            # temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k sampling
            if top_k is not None and top_k > 0:
                top_vals, top_inds = torch.topk(logits, k=top_k)
                probs = F.softmax(top_vals, dim=-1)
                next_id_in_topk = torch.multinomial(probs, 1)
                next_token = top_inds[next_id_in_topk]
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            token_id = next_token.item()
            generated_tokens.append(token_id)

            if token_id == end_token_idx:
                break

            current_token = next_token.unsqueeze(0)  # (1,1)

    return generated_tokens

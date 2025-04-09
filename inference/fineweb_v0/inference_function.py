import torch
import torch.nn.functional as F
from torch.amp import autocast
device = "cuda"


def inference(
        model,
        tokenizer,
        input_text: str,
        sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        max_gen_length: int = 1050,
        max_context_length: int = 1024,
):
    model.eval()
    generated = tokenizer.encode(input_text)

    print(input_text, end="", flush=True)

    with autocast("cuda"), torch.no_grad():
        for _ in range(max_gen_length):
            context = generated[-max_context_length:]
            input_ids = torch.Tensor([context]).long().to(device)

            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            if not sample:
                probability_dist = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(probability_dist).item()
            else:
                for token_id in set(generated):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

                next_token_logits -= temperature

                if top_k > 0:
                    vals, _ = torch.topk(next_token_logits, top_k)
                    threshold = vals[:, -1].unsqueeze(1)
                    next_token_logits = torch.where(
                        next_token_logits < threshold,
                        torch.full_like(next_token_logits, -float("Inf")),
                        next_token_logits
                    )

                if top_p < 1:
                    sorted_logits, sorted_index = torch.sort(next_token_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    to_remove = remove_mask.scatter(1, sorted_index, remove_mask)
                    next_token_logits = next_token_logits.masked_fill(to_remove, -float("Inf"))

                probability_dist = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probability_dist, num_samples=1).item()

            token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(token_str, end="", flush=True)

            generated.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)
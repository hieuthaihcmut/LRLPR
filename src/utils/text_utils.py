import torch

chars = ["<blank>"] + [str(i) for i in range(10)] + [chr(ord("A")+i) for i in range(26)]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

def ctc_decode(logits):
    idx = torch.softmax(logits, dim=-1).argmax(dim=-1)
    res = []
    for b in range(idx.size(0)):
        out, prev = [], None
        for t in range(idx.size(1)):
            c = int(idx[b, t])
            if c != 0 and c != prev: 
                out.append(itos[c])
            prev = c
        res.append("".join(out))
    return res
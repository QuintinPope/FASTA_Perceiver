import torch
import random



def ASCII_tokenize(strings, mask_prob = 0, mask_index = 1):
    if type(strings) == str:
        strings = [strings]
    charvalues = [[(mask_index if random.random() < mask_prob else ord(c)) for c in s if ord(c) < 256] for s in strings]
    max_len = max([len(chars) for chars in charvalues])
    for chars in charvalues:
        while len(chars) < max_len:
            chars.append(0)
    return torch.tensor(charvalues)

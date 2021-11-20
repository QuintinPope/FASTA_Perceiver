import torch
import random



def ASCII_tokenize(strings, mask_prob = 0, mask_index = 1, nucleotides = True, nucleotide_vocab = None, amino_acid_vocab = None):
    if type(strings) == str:
        strings = [strings]
    char_ids = []
    for s in strings:
        char_ids.append([])
        looking_at_text = False
        for c in s:
            if not looking_at_text:
                if nucleotides:
                    char_id = nucleotide_vocab.lookup(c)
                else:
                    char_id = amino_acid_vocab.lookup(c)
                if c == '|':
                    looking_at_text = True
                    break
            else:
                char_id = ord(c) if ord(c) < 256 else 2
            if random.random() < mask_prob:
                char_id = mask_index
            char_ids[-1].append(char_id)
                
    #charvalues = [[(mask_index if random.random() < mask_prob else ord(c)) for c in s if ord(c) < 256] for s in strings]
    max_len = max([len(chars) for chars in char_ids])
    for chars in charvalues:
        while len(chars) < max_len:
            chars.append(0)
    return torch.tensor(charvalues)

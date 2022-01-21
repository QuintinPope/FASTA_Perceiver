import torch
#import torch.nn as nn
import random



def ASCII_tokenize(
          strings, 
          mask_prob = 0, 
          mask_index = 1, 
          unk_index = 2, 
          cls_index = 3, 
          nucleotides = True, 
          nucleotide_vocab = None, 
          amino_acid_vocab = None, 
          split_token_types = False, 
          left_pad = 1,
          max_length = 2048
    ):
    if type(strings) == str:
        strings = [strings]
    char_ids = []
    for s in strings:
        char_ids.append([cls_index for _ in range(left_pad)])
        looking_at_text = False
        for c in s:
            c = str.upper(c)
            if not looking_at_text:
                if c == '|':
                    looking_at_text = True
                    break
                if nucleotides:
                    char_id = nucleotide_vocab[c]
                else:
                    char_id = amino_acid_vocab[c]
            else:
                char_id = ord(c) if ord(c) < 256 else 2
            if random.random() < mask_prob:
                char_id = mask_index
            char_ids[-1].append(char_id)
                
    #charvalues = [[(mask_index if random.random() < mask_prob else ord(c)) for c in s if ord(c) < 256] for s in strings]
    max_len = max([len(chars) for chars in char_ids])
    for chars in char_ids:
        while len(chars) < max_len:
            chars.append(0)
    return torch.tensor(char_ids)



def ASCII_tokenize_split(
          strings, 
          mask_prob = 0, 
          mask_index = 1, 
          unk_index = 2, 
          cls_index = 3, 
          nucleotides = True, 
          nucleotide_vocab = None, 
          amino_acid_vocab = None, 
          split_token_types = False,
          device = 'cuda:0',
          max_length = 2048
    ):

    if type(strings) == str:
        strings = [strings]
    nuc_ids = []
    str_ids = []
    for s in strings:
        nuc_ids.append([cls_index])
        str_ids.append([])
        looking_at_text = False
        for c in s:
            if not looking_at_text:
                c = str.upper(c)
                if c == '|':
                    looking_at_text = True
                else:
                    try:
                        if nucleotides:
                            char_id = nucleotide_vocab[c]
                        else:
                            char_id = amino_acid_vocab[c]
                    except:
                        char_id = unk_index
            if looking_at_text:
                char_id = ord(c) if ord(c) < 256 else 2
            # Only mask unk_token and non-special tokens:
            if random.random() < mask_prob and c != '|' and (char_id > 10 or char_id == 2):
                char_id = mask_index
            if looking_at_text and split_token_types:
                str_ids[-1].append(char_id)
            else:
                nuc_ids[-1].append(char_id)

    max_str_len = max([len(chars) for chars in str_ids])
    max_nuc_len = max([len(chars) for chars in nuc_ids])
    for chars in nuc_ids:
        while len(chars) < max_nuc_len:
            chars.append(0)
    for chars in str_ids:
        while len(chars) < max_str_len:
            chars.append(0)

    if split_token_types:
        return torch.tensor(nuc_ids)[:, :max_length].to(device), torch.tensor(str_ids)[:, :max_length].to(device)
    return torch.tensor(nuc_ids)[:, :max_length].to(device)


def compute_accuracy(logits, seq, labels):
    preds = torch.argmax(logits, dim = 2)
    correct_preds = torch.sum(preds == labels)
    total_preds = torch.sum(seq == 1)
    accuracy = correct_preds / total_preds
    return accuracy, preds, correct_preds, total_preds


def compute_task_accuracy(logits, seq, labels):
    preds = torch.argmax(logits, dim = 2)
    #print(preds)
    #print(labels)
    #print(preds[:,0])
    correct_preds_sol = torch.sum(preds[:,0] == labels[:,0])
    total_preds_sol = torch.sum(seq[:,0] == 1)
    accuracy_sol = correct_preds_sol / total_preds_sol

    correct_preds_loc = torch.sum(preds[:,1] == labels[:,1])
    total_preds_loc = torch.sum(seq[:,1] == 1)
    #print(total_preds_loc)
    accuracy_loc = correct_preds_loc / total_preds_loc

    return preds, accuracy_sol, correct_preds_sol, total_preds_sol, accuracy_loc, correct_preds_loc, total_preds_loc




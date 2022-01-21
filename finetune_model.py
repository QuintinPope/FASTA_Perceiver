import torch
import perceiver_pytorch

import argparse
from ranger21 import Ranger21
from helpers import *
from prepare_data import *
import torch_optimizer
import fixed_emb_perceiver
#import two_channel_perceiver
import transformers

from tqdm import tqdm 

nucleotide_vocab = {
    'A': 257,
    'C': 258,
    'T': 259,
    'G': 260,
    'R': 261,
    'S': 262,
    'U': 263,
    'N': 264
    }

amino_acid_vocab = {
    'A': 271,
    'C': 285,
    'D': 275,
    'E': 274,
    'F': 272,
    'G': 286,
    'H': 288,
    'I': 287,
    'K': 278,
    'L': 277,
    'M': 270,
    'N': 282,
    'P': 281,
    'Q': 284,
    'R': 280,
    'S': 273,
    'T': 289,
    'V': 276,
    'W': 283,
    'X': 290,
    'Y': 279,
    'Z': 291,
    '*': 292,
    '-': 293,
    'U': 294,
    'B': 295
    }

loc_counts = {'Cell.membrane': 599, 'Cytoplasm': 1676, 'Endoplasmic.reticulum': 527, 'Golgi.apparatus': 234, 'Lysosome/Vacuole': 166, 'Mitochondrion': 943, 'Nucleus': 2922, 'Peroxisome': 73, 'Plastid': 486, 'Extracellular': 838}
membrane_counts = {'M': 1963, 'S': 2869, 'U': 3632}

loc_to_char_vocab = {
    'Cell.membrane': 0,
    'Cytoplasm': 1,
    'Endoplasmic.reticulum': 2,
    'Golgi.apparatus': 3,
    'Lysosome/Vacuole': 4,
    'Mitochondrion': 5,
    'Nucleus': 6,
    'Peroxisome': 7,
    'Plastid': 8,
    'Extracellular': 9
    }


parser = argparse.ArgumentParser(description='Trains a perceiver on FASTA data.')
parser.add_argument("--data", default='none', help="Path to the file containing downstream finetuning data.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for finetuning")
parser.add_argument("--epochs", type=int, default=1, help="Number of epoch to run finetuning")
parser.add_argument("--model_path", default='../models/', help="Location from which to load pretrained models")
parser.add_argument("--output_path", default='../models/', help="Location to save finetuned models")
parser.add_argument("--device", default="cuda", help="Hardware device used to finetune model. Either \"cuda\" or \"cpu\".")
parser.add_argument("--opt", default="Adam", help="Optimizer to use for model finetuning")
parser.add_argument("--lr", type=float, default=0.001, help="Finetuning learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizers that support it.")
parser.add_argument("--warm_up", type=bool, default=True, help="Wheather to use a learning rate warm up.")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for all latents (not yet implemented).")
parser.add_argument("--multi_gpu", type=bool, default=False, help="Wheather to use all available gpus for finetuning.")
parser.add_argument("--task", default="sol", help="finetuning task selector.")
parser.add_argument("--max_len", type=int, default=2048, help="Maximum input length.")
parser.add_argument("--latent_dim", type=int, default=512, help="Size of hidden representations.")
parser.add_argument("--num_latents", type=int, default=256, help="Number of latent vectors used in perceiver attention.")
parser.add_argument("--pos_emb", default="abs", help="Type of positional embedding to use")
parser.add_argument("--left_pad", type=int, default=1, help="Number of left padding tokens to prepend to all inputs; used for downstream predictions.")
parser.add_argument("--mask_prob", type=float, default=0.0, help="Tokenizer mask probability. Set to > 0 to pretrain on masked language modeling..")


vocab_size = 300
args = parser.parse_args()


if args.model_path == 'none':
    model = fixed_emb_perceiver.FixedPosPerceiverLM(
        num_tokens = vocab_size,        # number of tokens
        dim = 256,                      # dimension of sequence to be encoded
        depth = 6,                      # depth of net
        max_seq_len = args.max_len,     # maximum sequence length
        num_latents = args.num_latents, # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = args.latent_dim,   # latent dimension
        cross_heads = 1,                # number of heads for cross attention. paper said 1
        latent_heads = 8,               # number of heads for latent self attention, 8
        cross_dim_head = 64,            # number of dimensions per cross attention head
        latent_dim_head = 64,           # number of dimensions per latent self attention head
        weight_tie_layers = False,      # whether to weight tie layers (optional, as indicated in the diagram)
        pos_emb = args.pos_emb, 
        dropout = args.dropout
    ).to(args.device)
elif args.model_path == 'bert':
    config = transformers.BertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_len,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None
    )
    model = transformers.BertForMaskedLM(config).to(args.device)
else:
    model = torch.load(args.model_path)

print(model)
#model = torch.nn.DataParallel(model)

training_texts, testing_texts, validation_texts = load_downstream(args.data, task=args.task, test_size=0.12, val_size=0.0)

if args.task == 'sol':
    training_locs = training_texts[' membrane']
    testing_locs = testing_texts[' membrane']
    training_texts = training_texts[(training_locs != "U").values]
    testing_texts = testing_texts[(testing_locs != "U").values]

    #logits_to_classes = torch.nn.Linear(vocab_size, 2)
loss_function = torch.nn.CrossEntropyLoss()

if str.lower(args.opt) == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'lamb':
    opt = torch_optimizer.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'adafactor':
    opt = torch_optimizer.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'ranger21':
    opt = Ranger21(model.parameters(), lr=args.lr, num_batches_per_epoch=int(len(training_texts) / args.batch_size), num_epochs=args.epochs, weight_decay=args.weight_decay, use_warmup=args.warm_up)
elif str.lower(args.opt) == 'adahessian':
    opt = torch_optimizer.Adahessian(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

half_max_len = int((args.max_len - args.left_pad) / 2)

for epoch in range(args.epochs):
    total_n_correct_sol = 0
    total_n_preds_sol = 0
    total_n_correct_loc = 0
    total_n_preds_loc = 0
    total_loss = 0
    print("Training epoch " + str(epoch) + ":")
    #continue
    for i in range(0, len(training_texts), args.batch_size):
        if args.task in ['sol', 'loc+sol']:
            mem = training_texts[' membrane'][i : i + args.batch_size]
            batch = training_texts['input'][i : i + args.batch_size]
            batch = [s.replace(" ", "")[:half_max_len] + s.replace(" ", "")[-half_max_len:] for s in batch.values]
            mem = [amino_acid_vocab[c] for c in mem.values]
        if args.task in ['loc+sol']:
            locs = training_texts[' loc'][i : i + args.batch_size].values
            loc_chars = [loc_to_char_vocab[l] for l in locs]
            #for l in locs:
            #    loc_chars.append(loc_to_char_vocab[l])
                

        #print(mem, batch)
        seq = ASCII_tokenize(batch, mask_prob = args.mask_prob, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, left_pad = args.left_pad).to(args.device)
        #labels = torch.nn.functional.one_hot(\
        #    ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab), \
        #    num_classes=512).float().to(args.device)
        labels = ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, left_pad = args.left_pad).to(args.device)
        for j in range(len(batch)):
            if args.task in ['sol', 'loc+sol']:
                if mem[j] != amino_acid_vocab['U']:
                    labels[j][0] = mem[j]
                    seq[j][0] = 1
            if args.task in ['loc+sol']:
                labels[j][1] = loc_chars[j]
                seq[j][1] = 1
        labels[seq != 1] = -100

        if args.model_path == 'bert':
            logits = model(seq).logits
        else:
            logits = model(seq)
        #probs = torch.nn.functional.softmax(logits, dim=2)
        loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
        #accuracy, preds, n_correct, n_preds = compute_accuracy(logits, seq, labels)
        preds, accuracy_sol, correct_preds_sol, total_preds_sol, accuracy_loc, correct_preds_loc, total_preds_loc = compute_task_accuracy(logits, seq, labels)
        if i % 120 == -1:
            print("labels:", labels)
            print("seq:", seq)
            print("preds:", preds)
            print("loc_chars:", loc_chars)
            print("mem:", mem)
            print("accuracy_sol:", accuracy_sol)
            print("accuracy_loc:", accuracy_loc)
            print("accuracy_sol, correct_preds_sol, total_preds_sol, accuracy_loc, correct_preds_loc, total_preds_loc:", accuracy_sol, correct_preds_sol, total_preds_sol, accuracy_loc, correct_preds_loc, total_preds_loc)
        total_n_correct_sol += correct_preds_sol
        total_n_preds_sol += total_preds_sol

        total_n_correct_loc += correct_preds_loc
        total_n_preds_loc += total_preds_loc
        total_loss += loss.item()
        #print(loss.item(), accuracy.item())

        opt.zero_grad()
        if str.lower(args.opt) == 'adahessian':
            loss.backward(create_graph = True)
        else:
            loss.backward()
        opt.step()
    print(total_loss * args.batch_size / len(training_texts), (total_n_correct_loc / total_n_preds_loc).item(), (total_n_correct_sol / total_n_preds_sol).item())
    total_n_correct_sol = 0
    total_n_preds_sol = 0
    total_n_correct_loc = 0
    total_n_preds_loc = 0
    total_loss = 0
    print("Testing epoch " + str(epoch) + ":")
    with torch.no_grad():
        for i in range(0, len(testing_texts), args.batch_size):
            if args.task in ['sol', 'loc+sol']:
                mem = testing_texts[' membrane'][i : i + args.batch_size]
                batch = testing_texts['input'][i : i + args.batch_size]
                batch = [s.replace(" ", "")[:half_max_len] + s.replace(" ", "")[-half_max_len:] for s in batch.values]
                #batch = [s.replace(" ", "")[:args.max_len] for s in batch.values]
                mem = [amino_acid_vocab[c] for c in mem.values]
            if args.task in ['loc+sol']:
                locs = testing_texts[' loc'][i : i + args.batch_size].values
                loc_chars = [loc_to_char_vocab[l] for l in locs]

            seq = ASCII_tokenize(batch, mask_prob = args.mask_prob, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, left_pad = args.left_pad).to(args.device)
            #labels = torch.nn.functional.one_hot(\
            #    ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab), \
            #    num_classes=512).float().to(args.device)
            labels = ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, left_pad = args.left_pad).to(args.device)

            for j in range(len(batch)):
                if args.task in ['sol', 'loc+sol']:
                    if mem[j] != amino_acid_vocab['U']:
                        labels[j][0] = mem[j]
                        seq[j][0] = 1
                if args.task in ['loc+sol']:
                    labels[j][1] = loc_chars[j]
                    seq[j][1] = 1
            labels[seq != 1] = -100

            if args.model_path == 'bert':
                logits = model(seq).logits
            else:
                logits = model(seq)
            #probs = torch.nn.functional.softmax(logits, dim=2)
            loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
            #accuracy, preds, n_correct, n_preds = compute_accuracy(logits, seq, labels)
            preds, accuracy_sol, correct_preds_sol, total_preds_sol, accuracy_loc, correct_preds_loc, total_preds_loc = compute_task_accuracy(logits, seq, labels)
            total_n_correct_sol += correct_preds_sol
            total_n_preds_sol += total_preds_sol

            total_n_correct_loc += correct_preds_loc
            total_n_preds_loc += total_preds_loc
            total_loss += loss.item()
            #print(loss.item(), accuracy.item())
    print(total_loss * args.batch_size / len(testing_texts), (total_n_correct_loc / total_n_preds_loc).item(), (total_n_correct_sol / total_n_preds_sol).item())

    if args.output_path != 'none':
        torch.save(model, args.output_path + "perceiver_model_epoch_" + str(epoch) + ".pth")


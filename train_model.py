import torch
import perceiver_pytorch
import argparse
from ranger21 import Ranger21
from helpers import *
from prepare_data import *
import torch_optimizer
import two_channel_perceiver


nucleotide_vocab = {
    'A': 257,
    'C': 258,
    'T': 259,
    'G': 260,
    'R': 261,
    'S': 262,
    'U': 263
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
    '-': 293
    }


#path_to_FASTA_file = '/Users/quintinpope/Documents/hmmer_tutorial/uniprot_sprot.fasta'
#path_to_hmmer_text_file = '/Users/quintinpope/Documents/hmmer_tutorial/hmmscan_Pfam-A_uniprot_sprot.txt'


parser = argparse.ArgumentParser(description='Trains a perceiver on FASTA data.')
parser.add_argument("--fasta_file", default='none', help="Path to the FASTA file containing training data.")
parser.add_argument("--hmmer_output", default='none', help="Text output from HMMER3 while annotating protein domains in the FASTA data file.")
parser.add_argument("--SRA_dir", default='none', help="Path to a directory containing a FASTA file generated from SRA runs and a metadata file.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for pretraining")
parser.add_argument("--latent_dim", type=int, default=512, help="Size of hidden representations.")
parser.add_argument("--num_latents", type=int, default=256, help="Number of latent vectors used in perceiver attention.")
parser.add_argument("--epochs", type=int, default=1, help="Number of epoch to run pretraining")
parser.add_argument("--output_path", default='../models/', help="Location to save pretrained models")
parser.add_argument("--device", default="cuda", help="Hardware device used to pretrain model. Either \"cuda\" or \"cpu\".")
parser.add_argument("--opt", default="Adam", help="Optimizer to use for model training")
parser.add_argument("--lr", type=float, default=0.001, help="Pretraining learning rate.")
parser.add_argument("--seperate_channels", type=bool, default=False, help="Should we use different perceivers for the text and nucleotide data?")
parser.add_argument("--multi_gpu", type=bool, default=False, help="Wheather to use all available gpus for pretraining.")




args = parser.parse_args()

if args.SRA_dir == 'none':
    training_texts = load_data_files(args.fasta_file, args.hmmer_output)
else:
    training_texts = load_SRA_files(args.SRA_dir)
#print("Number of pretraining texts = " + str(len(training_texts)))

vocab_size = 300
if args.seperate_channels:
    model = two_channel_perceiver.PerceiverLMTwoChannel(
        num_tokens = vocab_size,        # number of tokens
        dim = 32,                       # dimension of sequence to be encoded
        depth = 6,                      # depth of net
        max_seq_len = 2048,             # maximum sequence length
        num_latents = args.num_latents, # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = args.latent_dim,   # latent dimension
        cross_heads = 1,                # number of heads for cross attention. paper said 1
        latent_heads = 8,               # number of heads for latent self attention, 8
        cross_dim_head = 64,            # number of dimensions per cross attention head
        latent_dim_head = 64,           # number of dimensions per latent self attention head
        weight_tie_layers = False       # whether to weight tie layers (optional, as indicated in the diagram)
    ).to(args.device)
else:
    model = perceiver_pytorch.PerceiverLM(
        num_tokens = vocab_size,        # number of tokens
        dim = 32,                       # dimension of sequence to be encoded
        depth = 6,                      # depth of net
        max_seq_len = 2048,             # maximum sequence length
        num_latents = args.num_latents, # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = args.latent_dim,   # latent dimension
        cross_heads = 1,                # number of heads for cross attention. paper said 1
        latent_heads = 8,               # number of heads for latent self attention, 8
        cross_dim_head = 64,            # number of dimensions per cross attention head
        latent_dim_head = 64,           # number of dimensions per latent self attention head
        weight_tie_layers = False       # whether to weight tie layers (optional, as indicated in the diagram)
    ).to(args.device)

if args.multi_gpu:
    model = torch.nn.DataParallel(model)

loss_function = torch.nn.CrossEntropyLoss()
if str.lower(args.opt) == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
elif str.lower(args.opt) == 'lamb':
    opt = torch_optimizer.Lamb(model.parameters(), lr=args.lr)
elif str.lower(args.opt) == 'adafactor':
    opt = torch_optimizer.Adafactor(model.parameters(), lr=args.lr)
elif str.lower(args.opt) == 'ranger21':
    opt = Ranger21(model.parameters(), lr=args.lr, num_batches_per_epoch=int(len(training_texts) / args.batch_size), num_epochs=args.epochs)

#print(len(training_texts), "training texts.")
for epoch in range(args.epochs):
    #print(epoch)
    #continue
    for i in range(0, len(training_texts), args.batch_size):
        batch = training_texts[i : i + args.batch_size]
        seq = ASCII_tokenize(batch, mask_prob = 0.15, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab).to(args.device)
        #labels = torch.nn.functional.one_hot(\
        #    ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab), \
        #    num_classes=512).float().to(args.device)
        labels = ASCII_tokenize(batch, mask_prob = 0, nucleotides = False, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab).to(args.device)
        labels[seq != 1] = -100
        logits = model(seq)
        loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
        print(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()


    torch.save(model, args.output_path + "perceiver_model_epoch_" + str(epoch) + ".pth")

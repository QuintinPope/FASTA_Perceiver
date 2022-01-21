import torch
import perceiver_pytorch
import argparse
from ranger21 import Ranger21
from helpers import *
from prepare_data import *
import torch_optimizer
import two_channel_perceiver
import fixed_emb_perceiver
from madgrad import MADGRAD


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
parser.add_argument("--save_every", type=int, default=5, help="Save model every n epochs")
parser.add_argument("--device", default="cuda", help="Hardware device used to pretrain model. Either \"cuda\" or \"cpu\".")
parser.add_argument("--opt", default="Adam", help="Optimizer to use for model training")
parser.add_argument("--lr", type=float, default=0.001, help="Pretraining learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizers that support it.")
parser.add_argument("--model_type", type=str, default="perceiver", help="Should we use different perceivers for the text and nucleotide data?")
parser.add_argument("--multi_gpu", type=bool, default=False, help="Wheather to use all available gpus for pretraining")
parser.add_argument("--seperate_channels", type=bool, default=False, help="Wheather to use a two-channel perceiver model")
parser.add_argument("--pos_emb", default="abs", help="Type of positional embedding to use")




args = parser.parse_args()

if args.SRA_dir == 'none':
    use_nuc_vocab = False
    training_texts = load_AA_files(args.fasta_file, args.hmmer_output)
else:
    use_nuc_vocab = True
    training_texts = load_SRA_files(args.SRA_dir)
#print("Number of pretraining texts = " + str(len(training_texts)))

vocab_size = 300
if args.model_type == 'perceiver' and args.seperate_channels:
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
        weight_tie_layers = False,      # whether to weight tie layers (optional, as indicated in the diagram
        pos_emb = args.pos_emb
    ).to(args.device)
elif args.model_type == 'perceiver':
    model = fixed_emb_perceiver.FixedPosPerceiverLM(
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
        weight_tie_layers = False,      # whether to weight tie layers (optional, as indicated in the diagram)
        pos_emb = args.pos_emb
    ).to(args.device)
elif args.model_type == 'bert':
    print("Pretraining BERT is not yet implemented") 
else:
    print("Error:", args.model_type, "not a known model type.")

if args.multi_gpu:
    model = torch.nn.DataParallel(model)

print(model)

loss_function = torch.nn.CrossEntropyLoss()
if str.lower(args.opt) == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'lamb':
    opt = torch_optimizer.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'adafactor':
    opt = torch_optimizer.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'ranger21':
    opt = Ranger21(model.parameters(), lr=args.lr, num_batches_per_epoch=int(len(training_texts) / args.batch_size), num_epochs=args.epochs, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'madgrad':
    opt = MADGRAD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif str.lower(args.opt) == 'adahessian':
    opt = torch_optimizer.Adahessian(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
 

print(len(training_texts), "training texts.")
#print(args.seperate_channels)
for epoch in range(args.epochs):
    print("Epoch ", epoch)
    for i in range(0, len(training_texts), args.batch_size):
        log_line = ''
        batch = training_texts[i : i + args.batch_size]
        #print(batch)
        if args.seperate_channels:
            nuc_seq, str_seq = ASCII_tokenize_split(batch, mask_prob = 0.15, nucleotides = use_nuc_vocab, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, split_token_types = True, device = args.device)
            nuc_labels, str_labels = ASCII_tokenize_split(batch, mask_prob = 0, nucleotides = use_nuc_vocab, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab, split_token_types = True, device = args.device)

            #print(batch[0])
            #print(nuc_labels.size(), str_labels.size(), nuc_seq.device)
            nuc_labels[nuc_seq != 1] = -100
            str_labels[str_seq != 1] = -100
            labels = torch.cat((nuc_labels, str_labels), dim=1)

            logits = model(nuc_seq, str_seq)
            
            nuc_end = len(nuc_labels[0])
            nuc_logits = logits[:, :nuc_end, :]
            str_logits = logits[:, nuc_end:, :]

            nuc_loss = loss_function(nuc_logits.view(-1, vocab_size), nuc_labels.view(-1))
            nuc_preds = torch.argmax(nuc_logits, dim = 2)
            nuc_correct_preds = torch.sum(nuc_preds == nuc_labels)
            nuc_total_preds = torch.sum(nuc_seq == 1)
            nuc_accuracy = nuc_correct_preds / nuc_total_preds

            str_loss = loss_function(str_logits.view(-1, vocab_size), str_labels.view(-1))
            str_preds = torch.argmax(str_logits, dim = 2)
            str_correct_preds = torch.sum(str_preds == str_labels)
            str_total_preds = torch.sum(str_seq == 1)
            str_accuracy = str_correct_preds / str_total_preds

            log_line = log_line + ", " + str(nuc_loss) + ", " + str(nuc_accuracy) + ", " + str(str_loss) + ", " + str(str_accuracy)
            #print(labels, logits)
            #print(labels.size())
        else:
            seq = ASCII_tokenize_split(batch, mask_prob = 0.15, nucleotides = use_nuc_vocab, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab).to(args.device)
            labels = ASCII_tokenize_split(batch, mask_prob = 0, nucleotides = use_nuc_vocab, nucleotide_vocab = nucleotide_vocab, amino_acid_vocab = amino_acid_vocab).to(args.device)
            #print(labels, seq, batch)
            #seq = torch.tensor([[k % 5 for k in range(100)]]).cuda()
            #labels = torch.tensor([[k % 5 for k in range(100)]]).cuda()
            labels[seq != 1] = -100
            #print(labels, seq, batch)
            logits = model(seq)
            #print(logits)
            #print(labels.view(-1))
        #print(logits.size())
        probs = torch.nn.functional.softmax(logits, dim=2)
        #print(torch.sum(probs, dim=2), torch.sum(probs, dim=2).size())
        loss = loss_function(logits.view(-1, vocab_size), labels.view(-1))
        preds = torch.argmax(logits, dim = 2)
        correct_preds = torch.sum(preds == labels)
        total_preds = torch.sum(seq == 1)
        accuracy = correct_preds / total_preds
        log_line = str(loss.item()) + ", " + str(accuracy.item()) + log_line
        #if i % 20 == 0:
        #    print(preds)
        print(log_line)

        opt.zero_grad()
        if str.lower(args.opt) == 'adahessian':
            loss.backward(create_graph = True)
        else:
            loss.backward()
        opt.step()


    if args.output_path != 'none' and epoch % args.save_every == 0 and epoch != 0:
        torch.save(model, args.output_path + "perceiver_model_epoch_" + str(epoch) + ".pth")

import torch
import perceiver_pytorch
import argparse
from helpers import *
from prepare_data import *


#path_to_FASTA_file = '/Users/quintinpope/Documents/hmmer_tutorial/uniprot_sprot.fasta'
#path_to_hmmer_text_file = '/Users/quintinpope/Documents/hmmer_tutorial/hmmscan_Pfam-A_uniprot_sprot.txt'


parser = argparse.ArgumentParser(description='Trains a perceiver on FASTA data.')


parser.add_argument("--fasta_file", help="Path to the FASTA file containing training data.")
parser.add_argument("--hmmer_output", help="Text output from HMMER3 while annotating protein domains in the FASTA data file.")


args = parser.parse_args()


training_texts = load_data_files(args.fasta_file, args.hmmer_output)


model = perceiver_pytorch.PerceiverLM(
    num_tokens = 256,          # number of tokens
    dim = 32,                    # dimension of sequence to be encoded
    depth = 6,                   # depth of net
    max_seq_len = 2048,          # maximum sequence length
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)
loss_function = torch.nn.BCEWithLogitsLoss()

opt = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(20):
    batch = training_texts[:16]
    seq = ASCII_tokenize(batch, mask_prob = 0.15)
    labels = torch.nn.functional.one_hot(ASCII_tokenize(batch, mask_prob = 0), num_classes=256).float()
    logits = model(seq) # (1, 512, 20000)
    loss = loss_function(logits, labels)
    print(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()


torch.save(model, "perceiver_model.pth")

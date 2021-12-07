import pandas as pd
import numpy as np
from Bio import SeqIO, SearchIO
from collections import defaultdict
import torch
import os

path_to_FASTA_file = '/content/drive/MyDrive/Microbiome Deep Learning/FASTA_sample_data/small_uniprot_sprot.fasta.txt'
path_to_hmmer_text_file = '/content/drive/MyDrive/Microbiome Deep Learning/FASTA_sample_data/small_seq_set_Pfams.txt'

def load_SRA_files(path_to_dir):
    training_texts = []
    files = os.listdir(path_to_dir)
    metadata_dict = {}
    metadata_file = open(path_to_dir + "SraRunTable.txt", 'r')
    metadata_lines = metadata_file.read().split("\nSRR")
    for line in metadata_lines:
        line.replace("\n", "")
        try:
            exp_id = int(line[:8])
        except:
            #print(line[:8])
            continue
        metadata_text = line[8:]
        metadata_dict[exp_id] = metadata_text[1:]
    metadata_file.close()
    for file_name in files:
        #print(file_name)
        f = open(path_to_dir + file_name, 'r')
        if '.fasta' in file_name:
            
            exp_id = -1
            for line in f.read().split('\n'):
                if len(line) < 1:
                    continue
                if line[0] == '>':
                    for end in range(3, 12, -1):
                        try:
                            exp_id = int(line[4:end])
                            break
                        except:
                            pass
                else:
                    if exp_id in metadata_dict:
                        metadata = metadata_dict[exp_id]
                    else:
                        metadata = ''
                    training_texts.append(line + "|" + metadata)
        f.close()
    rng = np.random.default_rng()
    training_texts = rng.permutation(training_texts).tolist()
    return training_texts

def load_AA_files(path_to_FASTA_file, path_to_hmmer_text_file):
    reduced_properties = ["bias",
                          "bitscore",
                          "description",
                          "domain_exp_num",
                          "domain_obs_num",
                          "evalue",
                          "id",
                          "query_description",
                          "query_id"]

    with open(path_to_FASTA_file) as fasta_file:
        sequence_dict = defaultdict(list)
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            sequence_dict['query_id'].append(seq_record.id)
            sequence_dict['seqs'].append(str(seq_record.seq))
            sequence_dict['desc'].append(seq_record.description)

    sequence_df = pd.DataFrame.from_dict(sequence_dict)

    hits_dict = defaultdict(list)
    with open(path_to_hmmer_text_file) as handle:
        for queryresult in SearchIO.parse(handle, 'hmmer3-text'):
            for hit in queryresult.hits:
                for attrib in reduced_properties:
                    hits_dict[attrib].append(getattr(hit, attrib))

    hits_df = pd.DataFrame.from_dict(hits_dict)
    joined_df = hits_df.set_index('query_id').join(other=sequence_df.set_index('query_id'), on='query_id' )

    training_texts = []

    for i in range(len(sequence_df)):
        seq = sequence_df.iloc[i]['seqs']
        desc = sequence_df.iloc[i]['desc']
        query_id = sequence_df.iloc[i]['query_id']
        seq_hits = hits_df[hits_df['query_id'] == query_id]
        hits_str = "||".join([str(hit[1]) + '|' + str(hit[5]) + '|' + hit[2] + '|' + hit[6] for hit in seq_hits.values])
        training_texts.append(seq + "|" + desc + "||" + hits_str)
    return training_texts

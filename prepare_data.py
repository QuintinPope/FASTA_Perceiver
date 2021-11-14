import pandas as pd
import numpy as np
from Bio import SeqIO, SearchIO
from collections import defaultdict
import torch


path_to_FASTA_file = '/content/drive/MyDrive/Microbiome Deep Learning/FASTA_sample_data/small_uniprot_sprot.fasta.txt'
path_to_hmmer_text_file = '/content/drive/MyDrive/Microbiome Deep Learning/FASTA_sample_data/small_seq_set_Pfams.txt'


def load_data_files(path_to_FASTA_file, path_to_hmmer_text_file):
    full_properties = ["bias",
                       "bitscore",
                       "description",
                       "dbxrefs",
                       "domain_exp_num",
                       "domain_obs_num",
                       "evalue",
                       "fragments",
                       "hsps",
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

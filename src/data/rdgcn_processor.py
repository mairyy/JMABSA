"""
Prepare vocabulary, initial word vectors and generate syn dep.
"""
import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter
import os
import json
import networkx as nx
from pprint import pprint

MAX_TREE_DIS = 10

def syn_dep_adj_generation(head, dep, vocab_dep):
    syn_dep_edge = []
    for node_s_id, (node_e_id, d) in enumerate(zip(head, dep)):
        if node_e_id == 0:
            continue
        syn_dep_edge.append((node_s_id, node_e_id-1, vocab_dep.stoi.get(d)))
    return syn_dep_edge

def syn_dis_adj_generation(head):
    syn_nx = syn_network_generation(head)
    syn_dis_edge = []
    for node_s_id in syn_nx.nodes:
        for node_e_id in syn_nx.nodes:
            try:
                tree_distance = nx.dijkstra_path_length(syn_nx, source=node_s_id, target=node_e_id)
                tree_distance = tree_distance if tree_distance <= MAX_TREE_DIS else MAX_TREE_DIS
            except:
                tree_distance = MAX_TREE_DIS
            syn_dis_edge.append((node_s_id, node_e_id, tree_distance))
    return syn_dis_edge

def syn_network_generation(head):
    syn_nx = nx.Graph()
    syn_nx.add_nodes_from(range(len(head)))
    syn_nx.add_edges_from([(node_1, node_2 - 1) for node_1, node_2 in enumerate(head) if node_2 != 0])
    return syn_nx

def syn_adj_generation(file_prefix, vocab_dep):
    for file_type in ['train', 'test', 'dev']:
        if os.path.exists(file_prefix + '/' + file_type + '_preprocessed.json'):
            continue
        else:
            with open(file_prefix + '/' + file_type + '.json', 'r') as f:
                file = json.load(f)
                for data in file:
                    head = data['head']
                    dep = data['deprel']
                    data['syn_dep_adj'] = syn_dep_adj_generation(head, dep, vocab_dep)
                    data['syn_dis_adj'] = syn_dis_adj_generation(head)
                file_ = open(file_prefix + '/' + file_type + '_preprocessed.json', 'w')
                file_.write(json.dumps(file))
                file_.close()
                print(file_prefix + '/' + file_type + '_preprocessed.json has been saved...')
    
class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='./src/data/twitter2017', help='TACRED directory.')
    parser.add_argument('--vocab_dir', default='./src/data/twitter2017', help='Output vocab directory.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    test_file = args.data_dir + '/test.json'
    dev_file = args.data_dir + '/dev.json'

    # output files
    # token
    #vocab_tok_file = args.vocab_dir + '/vocab_tok.vocab'
    # position
    #vocab_post_file = args.vocab_dir + '/vocab_post.vocab'
    # pos_tag
    #vocab_pos_file = args.vocab_dir + '/vocab_pos.vocab'
    # dep_rel
    vocab_dep_file = args.vocab_dir + '/vocab_dep.vocab'
    # polarity
    #vocab_pol_file = args.vocab_dir + '/vocab_pol.vocab'

    # load files
    print("loading files...")
    train_dep = load_tokens(train_file)
    test_dep = load_tokens(test_file)
    dev_dep = load_tokens(dev_file)

    # counters
    #token_counter = Counter(train_tokens+test_tokens)
    #pos_counter = Counter(train_pos+test_pos)
    dep_counter = Counter(train_dep+test_dep+dev_dep)
    #max_len = max(train_max_len, test_max_len)
    #post_counter = Counter(list(range(-max_len, max_len)))
    #pol_counter = Counter(['positive', 'negative', 'neutral'])

    # build vocab
    print("building vocab...")
    #token_vocab = VocabHelp(token_counter, specials=['<pad>', '<unk>'])
    #pos_vocab = VocabHelp(pos_counter, specials=['<pad>', '<unk>'])
    dep_vocab = VocabHelp(dep_counter, specials=['<pad>', '<unk>'])
    #post_vocab = VocabHelp(post_counter, specials=['<pad>', '<unk>'])
    #pol_vocab = VocabHelp(pol_counter, specials=[])
    #print("token_vocab: {}, pos_vocab: {}, dep_vocab: {}, post_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(pos_vocab), len(dep_vocab), len(post_vocab), len(pol_vocab)))
    print("dep_vocab: {}".format(len(dep_vocab)))

    print("dumping to files...")
    #token_vocab.save_vocab(vocab_tok_file)
    #pos_vocab.save_vocab(vocab_pos_file)
    dep_vocab.save_vocab(vocab_dep_file)
    #post_vocab.save_vocab(vocab_post_file)
    #pol_vocab.save_vocab(vocab_pol_file)

    print("generate syntatic adj matrix...")
    print(dep_vocab.itos, dep_vocab.stoi)
    #twitter2015
    #['<pad>', '<unk>', 'ROOT', 'punct', 'compound', 'prep', 'pobj', 'nsubj', 'det', 'dobj', 'amod', 'dep', 'advmod', 'aux', 'nmod', \
    # 'nummod', 'appos', 'npadvmod', 'conj', 'poss', 'cc', 'ccomp', 'advcl', 'attr', 'xcomp', 'acl', 'acomp', 'relcl', 'prt', 'mark', \
    # 'pcomp', 'auxpass', 'nsubjpass', 'intj', 'case', 'neg', 'quantmod', 'agent', 'dative', 'oprd', 'expl', 'parataxis', 'csubj', \
    # 'predet', 'preconj', 'meta'] {'<pad>': 0, '<unk>': 1, 'ROOT': 2, 'punct': 3, 'compound': 4, 'prep': 5, 'pobj': 6, 'nsubj': 7, \
    # 'det': 8, 'dobj': 9, 'amod': 10, 'dep': 11, 'advmod': 12, 'aux': 13, 'nmod': 14, 'nummod': 15, 'appos': 16, 'npadvmod': 17, \
    # 'conj': 18, 'poss': 19, 'cc': 20, 'ccomp': 21, 'advcl': 22, 'attr': 23, 'xcomp': 24, 'acl': 25, 'acomp': 26, 'relcl': 27, 'prt': 28, \
    # 'mark': 29, 'pcomp': 30, 'auxpass': 31, 'nsubjpass': 32, 'intj': 33, 'case': 34, 'neg': 35, 'quantmod': 36, 'agent': 37, 'dative': 38, \
    # 'oprd': 39, 'expl': 40, 'parataxis': 41, 'csubj': 42, 'predet': 43, 'preconj': 44, 'meta': 45
    #twitter2017
    #['<pad>', '<unk>', 'compound', 'punct', 'ROOT', 'prep', 'pobj', 'nsubj', 'det', 'dobj', 'amod', 'advmod', 'aux', 'nummod', 'nmod', \
    # 'dep', 'poss', 'appos', 'conj', 'cc', 'npadvmod', 'ccomp', 'attr', 'case', 'advcl', 'xcomp', 'acomp', 'mark', 'prt', 'relcl', 'acl', \
    # 'pcomp', 'auxpass', 'neg', 'nsubjpass', 'quantmod', 'intj', 'dative', 'agent', 'oprd', 'predet', 'expl', 'meta', 'csubj', 'parataxis',
    #  'preconj'] {'<pad>': 0, '<unk>': 1, 'compound': 2, 'punct': 3, 'ROOT': 4, 'prep': 5, 'pobj': 6, 'nsubj': 7, 'det': 8, 'dobj': 9, 
    # 'amod': 10, 'advmod': 11, 'aux': 12, 'nummod': 13, 'nmod': 14, 'dep': 15, 'poss': 16, 'appos': 17, 'conj': 18, 'cc': 19, 
    # 'npadvmod': 20, 'ccomp': 21, 'attr': 22, 'case': 23, 'advcl': 24, 'xcomp': 25, 'acomp': 26, 'mark': 27, 'prt': 28, 'relcl': 29, 
    # 'acl': 30, 'pcomp': 31, 'auxpass': 32, 'neg': 33, 'nsubjpass': 34, 'quantmod': 35, 'intj': 36, 'dative': 37, 'agent': 38, 
    # 'oprd': 39, 'predet': 40, 'expl': 41, 'meta': 42, 'csubj': 43, 'parataxis': 44, 'preconj': 45}
    syn_adj_generation(args.data_dir, dep_vocab)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            #tokens.extend(d['token'])
            #pos.extend(d['pos'])
            dep.extend(d['deprel'])
    #print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return dep

if __name__ == '__main__':
    main()

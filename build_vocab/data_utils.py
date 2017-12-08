#-*- coding: utf-8 -*-
# Copyright 2016 AC Technologies LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for tokenizing, creation vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


def create_vocabulary(data):
  """Create vocabulary from input data.
  Input data is assumed to contain one word per line.

  Args:
    data: word list that will be used to create vocabulary.

  Rerurn:
    vocab: vocabulary dictionary. In this dictionary keys are symbols
           and values are their indexes.
  """
  vocab = {}
  for line in data:
    for item in line:
      if item in vocab:
        vocab[item] += 1
      else:
        vocab[item] = 1
  vocab_list = _START_VOCAB + sorted(vocab)
  vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
  return vocab

def create_word(data):
  """Create word from input data.
  Input data is assumed to contain one word per line.

  Args:
    data: word list that will be used to create vocabulary.

  Rerurn:
    vocab: vocabulary dictionary. In this dictionary keys are symbols
           and values are their indexes.
  """
  vocab = {}
  for line in data:

    if ''.join(line) in vocab:
      vocab[''.join(line)] += 1
    else:
      vocab[''.join(line)] = 1
  vocab_list = _START_VOCAB + sorted(vocab)
  vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
  return vocab


def save_vocabulary(vocab, vocabulary_path):
  """Save vocabulary file in vocabulary_path.
  We write vocabulary to vocabulary_path in a one-token-per-line format,
  so that later token in the first line gets id=0, second line gets id=1,
  and so on.

  Args:
    vocab: vocabulary dictionary.
    vocabulary_path: path where the vocabulary will be created.

  """
  print("Creating vocabulary %s" % (vocabulary_path))
  with codecs.open(vocabulary_path, "w", "utf-8") as vocab_file:
    for symbol in sorted(vocab, key=vocab.get):
      vocab_file.write(symbol + '\n')

def save_paired(vocab,vocab_paired, vocabulary_path):
  """Save vocabulary paired file in vocabulary_path.
  We write vocabulary to vocabulary_path in a one-token-per-line format,
  so that later token in the first line gets id=0, second line gets id=1,
  and so on.

  Args:
    vocab: vocabulary dictionary.
    vocab_paired: vocabulary dictionary paired.
    vocabulary_path: path where the vocabulary will be created.

  """
  print("Creating vocabulary %s" % (vocabulary_path))
  with codecs.open(vocabulary_path, "w", "utf-8") as vocab_file:
    for vocab1,vocab2 in zip(vocab,vocab_paired):
      vocab_file.write(''.join(vocab1)+' '+''.join(vocab2) + '\n')


def split_to_grapheme_phoneme(inp_dictionary):
  """Split input dictionary into two separate lists with graphemes and phonemes.

  Args:
    inp_dictionary: input dictionary.
  """
  graphemes, phonemes = [], []
  for line in inp_dictionary:
    split_line = line.strip().split()
    if len(split_line) > 1:
      graphemes.append(list(split_line[0]))
      phonemes.append(split_line[1:])
  return graphemes, phonemes


def collect_pronunciations(dic_lines):
  '''Create dictionary mapping word to its different pronounciations.
  '''
  dic = {}
  for line in dic_lines:
    lst = line.strip().split()
    if len(lst) > 1:
      if lst[0] not in dic:
        dic[lst[0]] = [" ".join(lst[1:])]
      else:
        if not " ".join(lst[1:]) in dic[lst[0]]:
          dic[lst[0]].append(" ".join(lst[1:]))
    elif len(lst) == 1:
      print("WARNING: No phonemes for word '%s' line ignored" % (lst[0]))
  return dic


def split_dictionary_from_data(train_file, valid_file=None, test_file=None):
  """Split source dictionary to train, validation and test sets.
  """
  with codecs.open(train_file+"_text.txt", "r", "utf-8") as f:
    texts = f.readlines()
  with codecs.open(train_file+"_trans.txt", "r", "utf-8") as f:
    trans = f.readlines()
  source_dic = []
  for text,tran in zip(texts,trans):
    for text_word, tran_word in zip(text.split(),tran.split()):
      source_dic.append(text_word+" "+" ".join(tran_word))

  train_dic, valid_dic, test_dic = [], [], []
  if valid_file:
    with codecs.open(valid_file + "_text.txt", "r", "utf-8") as f:
      texts = f.readlines()
    with codecs.open(valid_file + "_trans.txt", "r", "utf-8") as f:
      trans = f.readlines()
      valid_dic = []
    for text, tran in zip(texts, trans):
      for text_word, tran_word in zip(text.split(), tran.split()):
        valid_dic.append(text_word + " " + " ".join(tran_word))
  if test_file:
    with codecs.open(test_file + "_text.txt", "r", "utf-8") as f:
      texts = f.readlines()
    with codecs.open(test_file + "_trans.txt", "r", "utf-8") as f:
      trans = f.readlines()
      test_dic = []
    for text, tran in zip(texts, trans):
      for text_word, tran_word in zip(text.split(), tran.split()):
        test_dic.append(text_word + " " + " ".join(tran_word))


  dic = collect_pronunciations(source_dic)

  # Split dictionary to train, validation and test (if not assigned).
  for i, word in enumerate(dic):
    for pronunciations in dic[word]:
      if i % 20 == 0 and not valid_file:
        valid_dic.append(word + ' ' + pronunciations)
      elif (i % 20 == 1 or i % 20 == 2) and not test_file:
        test_dic.append(word + ' ' + pronunciations)
      else:
        train_dic.append(word + ' ' + pronunciations)
  return train_dic, valid_dic, test_dic

def prepare_g2p_from_naive_data(model_dir, train_path, valid_path, test_path):
  """Create vocabularies into model_dir, create ids data lists.

  Args:
    model_dir: directory in which the data sets will be stored;
    train_path: path to training dictionary;
    valid_path: path to validation dictionary;
    test_path: path to test dictionary.

  """
  # Create train, validation and test sets.
  train_dic, valid_dic, test_dic = split_dictionary_from_data(train_path, valid_path,
                                                    test_path)
  # Split dictionaries into two separate lists with graphemes and phonemes.
  train_gr, train_ph = split_to_grapheme_phoneme(train_dic)
  valid_gr, valid_ph = split_to_grapheme_phoneme(valid_dic)
  test_gr, test_ph = split_to_grapheme_phoneme(test_dic)

  # Load/Create vocabularies.
  if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

  save_paired(train_gr, train_ph, os.path.join(model_dir, "train_vocab.paired"))
  save_paired(train_ph, train_gr, os.path.join(model_dir, "train_vocab.paired_inverse"))

  save_paired(valid_gr, valid_ph, os.path.join(model_dir, "valid_vocab.paired"))
  save_paired(valid_ph, valid_gr, os.path.join(model_dir, "valid_vocab.paired_inverse"))

  save_paired(test_gr, test_ph, os.path.join(model_dir, "test_vocab.paired"))
  save_paired(test_ph, test_gr, os.path.join(model_dir, "test_vocab.paired_inverse"))


  if 1==0:
    ph_vocab = create_vocabulary(train_ph)
    gr_vocab = create_vocabulary(train_gr)
    save_vocabulary(ph_vocab, os.path.join(model_dir, "vocab.phoneme"))
    save_vocabulary(gr_vocab, os.path.join(model_dir, "vocab.grapheme"))

  if 1==0:
    word_text = create_word(train_gr)
    word_tras = create_word(train_ph)
    save_vocabulary(word_text, os.path.join(model_dir, "vocab.word_text"))
    save_vocabulary(word_tras, os.path.join(model_dir, "vocab.word_tran"))

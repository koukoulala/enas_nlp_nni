import os
import io
import sys
import csv
import numpy as np
import math
import tensorflow as tf
import gensim.models.keyedvectors as word2vec
from pathlib import Path
from ptb import *

global only_sentence, use_embed_layer, phrase_min_length, slot_num
only_sentence = False
phrase_min_length=0
embed_dim = 300
slot_num = 1

def set_random_seed(seed):
  print("-" * 80)
  print("set random seed for data reading: {}".format(seed))
  np.random.seed(seed)

def sst_load_trees(filename):
  trees = read_trees(filename)
  return trees

def yelp_load_phrases(filename, train_ratio=1.0,
                      valid_ratio=0.0, valid_phrases=[]):
  phrases = []
  csv.field_size_limit(1000000000)
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for slots in csv_reader:
      if len(slots) < 2:
        continue
      global slot_num
      slot_num = len(slots) - 1
      sentence = ""
      for i in range(1, len(slots)):
        sentence += slots[i] + " <sep> "
      sentence = sentence.replace("\\n", " <end> ")
      sentence = sentence.strip()
      split_label = int(slots[0])
      pair = (sentence, split_label)
      if train_ratio == 1.0:
        phrases.append(pair)
      else:
        rand_portion = np.random.random()
        if rand_portion < train_ratio:
          phrases.append(pair)
        elif rand_portion < train_ratio + valid_ratio:
          valid_phrases.append(pair)
  np.random.shuffle(phrases)
  return phrases

def sst_get_id_input(content, word_id_dict, max_input_length):
  words = content.split(' ')
  sentence = []
  field = []
  index, f_index = 0, 1
  for i in range(0, max(max_input_length, len(words))):
    if len(words) <= index:
      id = word_id_dict["<pad>"]
    else:
      word = words[index].strip()
      if word in word_id_dict:
        id = word_id_dict[word]
      else:
        id = word_id_dict["<unknown>"]
    if index < max_input_length:
      sentence.append(id)
      if len(words) <= index:
        field.append(0)
      else:
        field.append(f_index)
    index += 1
    if "<sep>" in word_id_dict and id == word_id_dict["<sep>"]:
      f_index += 1
  return sentence, field

def sst_get_phrases(trees, sample_ratio=1.0,
                    is_binary=False, only_sentence=False):
  all_phrases = []
  for tree in trees:
    if only_sentence == True:
      sentence = get_sentence_by_tree(tree)
      label = int(tree.label)
      pair = (sentence, label)
      all_phrases.append(pair)
    else:
      phrases = get_phrases_by_tree(tree)
      sentence = get_sentence_by_tree(tree)
      pair = (sentence, int(tree.label))
      all_phrases.append(pair)
      all_phrases.extend(phrases)
  np.random.shuffle(all_phrases)
  result_phrases = []
  for pair in all_phrases:
    if is_binary:
      phrase = pair[0]
      label = pair[1]
      if label <= 1:
        pair = (phrase, 0)
      elif label >= 3:
        pair = (phrase, 1)
      else:
        continue
    if sample_ratio == 1.0:
      result_phrases.append(pair)
    else:
      rand_portion = np.random.random()
      if rand_portion < sample_ratio:
        result_phrases.append(pair)
  return result_phrases

def get_phrases_by_tree(tree):
  phrases = []
  if tree == None:
    return phrases
  if tree.is_leaf():
    pair = (tree.word, int(tree.label))
    phrases.append(pair)
    return phrases
  left_child_phrases = get_phrases_by_tree(tree.subtrees[0])
  right_child_phrases = get_phrases_by_tree(tree.subtrees[1])
  phrases.extend(left_child_phrases)
  phrases.extend(right_child_phrases)
  sentence = get_sentence_by_tree(tree)
  pair = (sentence, int(tree.label))
  phrases.append(pair)
  return phrases

def get_sentence_by_tree(tree):
  sentence = ""
  if tree == None:
    return sentence
  if tree.is_leaf():
    return tree.word
  left_sentence = get_sentence_by_tree(tree.subtrees[0])
  right_sentence = get_sentence_by_tree(tree.subtrees[1])
  sentence = left_sentence + " " + right_sentence
  return sentence.strip()

def get_word_id_dict(word_num_dict, word_id_dict, min_count):
  for word in word_num_dict:
    count = word_num_dict[word]
    if count >= min_count:
      index = len(word_id_dict)
      if word not in word_id_dict:
        word_id_dict[word] = index
  return word_id_dict

def load_word_num_dict(phrases, word_num_dict):
  for (sentence, label) in phrases:
    words = sentence.split(' ')
    for cur_word in words:
      word = cur_word.strip()
      if word not in word_num_dict:
        word_num_dict[word] = 1
      else:
        count = word_num_dict[word]
        word_num_dict[word] += 1
  return word_num_dict

def init_trainable_embedding(embedding, word_id_dict,
                word_embed_model, unknown_word_embed):
  embed_dim = unknown_word_embed.shape[0]
  embedding[0] = np.zeros(embed_dim)
  embedding[1] = unknown_word_embed
  for word in word_id_dict:
    id = word_id_dict[word]
    if id == 0 or id == 1:
      continue
    if word in word_embed_model:
      embedding[id] = word_embed_model[word]
    else:
      embedding[id] = np.random.rand(embed_dim) / 2.0 - 0.25
  print("init embedding: {0}".format(embedding))
  return embedding

def sst_get_trainable_data(phrases, word_id_dict,
                           split_label, max_input_length, is_binary):
  images, bow_images, labels, datasets = [], [], [], []

  for (phrase, label) in phrases:
    if len(phrase.split(' ')) < phrase_min_length:
      continue
    phrase_input, field_input = sst_get_id_input(phrase,
                                                 word_id_dict,
                                                 max_input_length)
    images.append(phrase_input)
    bow_images.append(field_input)
    labels.append(int(label))
    datasets.append("sst")
  labels = np.array(labels, dtype=np.int32)
  datasets = np.array(datasets, dtype=np.str)
  if split_label == 1:
    split_label_str = "train"
  elif split_label == 2:
    split_label_str = "test"
  else:
    split_label_str = "valid"
  images = np.reshape(images, [-1, max_input_length])  #(N, len)
  images = images.astype(np.int32)
  bow_images = np.reshape(bow_images, [-1, max_input_length])
  bow_images = bow_images.astype(np.int32)
  print(split_label_str, images.shape, labels.shape, datasets.shape)
  print(images)
  return images, bow_images, labels, datasets

def yelp_get_trainable_data(phrases, word_id_dict, dataset_name,
                 word_embed_model, split_label, max_input_length):
  images, bow_images, labels, datasets = [], [], [], []
  bow_max_input_length = 10
  for (phrase, label) in phrases:
    phrase_input, field_input = sst_get_id_input(phrase,
                                                 word_id_dict,
                                                 max_input_length)
    bow_phrase_input, bow_field_input = sst_get_id_input(phrase,
                                                         word_id_dict,
                                                         bow_max_input_length)
    images.append(phrase_input)
    bow_images.append(field_input)
    labels.append(int(label) - 1)
    datasets.append(dataset_name)
  labels = np.array(labels, dtype=np.int32)
  datasets = np.array(datasets, dtype=np.str)
  if split_label == 1:
    split_label_str = "train"
  elif split_label == 2:
    split_label_str = "test"
  else:
    split_label_str = "valid"
  images = np.reshape(images, [-1, max_input_length])  #(N, len)
  images = images.astype(np.int32)
  bow_images = np.reshape(bow_images, [-1, max_input_length])
  bow_images = bow_images.astype(np.int32)
  print(split_label_str, images.shape, labels.shape, datasets.shape)
  print(images)
  return images, bow_images, labels, datasets

def load_glove_model(filename):
  embedding_dict = {}
  with open(filename) as f:
    for line in f:
      vocab_word, vec = line.strip().split(' ', 1)
      embed_vector = list(map(float, vec.split()))
      embedding_dict[vocab_word] = embed_vector
  return embedding_dict

def load_embedding(embedding_model):
  word_embed_model = {}

  if embedding_model == "word2vec" or embedding_model == "all":
    embedding_data_path = '/data/yujwang/word2vec-GoogleNews-vectors'
    embedding_data_file = os.path.join(embedding_data_path,'GoogleNews-vectors-negative300.bin')
    word_embed_model["word2vec"] = word2vec.KeyedVectors.load_word2vec_format(embedding_data_file, binary=True)

  if embedding_model == "glove" or embedding_model == "all":
    embedding_data_path = '/data/yujwang'
    embedding_data_file = os.path.join(embedding_data_path, 'glove.840B.300d.txt')
    word_embed_model["glove"] = load_glove_model(embedding_data_file)

  unknown_word_embed = np.random.rand(embed_dim)    #create random vector in [0, 1)
  unknown_word_embed = (unknown_word_embed - 0.5) / 2.0
  return word_embed_model, unknown_word_embed

def read_data_sst(word_id_dict, word_num_dict, data_path, max_input_length,
                  embedding_model, min_count, train_ratio, valid_ratio,
                  is_binary=False, is_valid=False, cache={}):
  """Reads SST format data. Always returns NHWC format

  Returns:
    sentences: np tensor of size [N, H, W, C=1]
    labels: np tensor of size [N]
  """

  images, labels, datasets = {}, {}, {}

  if len(cache) == 0:
    print("-" * 80)
    print("Reading SST data")

    train_file_name = os.path.join(data_path, 'train.txt')
    valid_file_name = os.path.join(data_path, 'dev.txt')
    test_file_name = os.path.join(data_path, 'test.txt')

    train_trees = sst_load_trees(train_file_name)
    train_phrases = sst_get_phrases(train_trees,
                                    train_ratio,
                                    is_binary,
                                    only_sentence)
    print("finish load train_phrases")
    valid_trees = sst_load_trees(valid_file_name)
    valid_phrases = sst_get_phrases(valid_trees,
                                    valid_ratio,
                                    is_binary,
                                    only_sentence or is_valid)
    if is_valid == False:
      train_phrases = train_phrases + valid_phrases
      valid_phrases = None
    test_trees = sst_load_trees(test_file_name)
    test_phrases = sst_get_phrases(test_trees,
                                   1.0,
                                   is_binary,
                                   only_sentence=True)
    print("finish load test_phrases")

    cache["train"] = train_phrases
    cache["valid"] = valid_phrases
    cache["test"] = test_phrases
  else:
    train_phrases = cache["train"]
    valid_phrases = cache["valid"]
    test_phrases = cache["test"]

  #get word_id_dict
  word_id_dict["<pad>"] = 0
  word_id_dict["<unknown>"] = 1
  load_word_num_dict(train_phrases, word_num_dict)
  print("finish load train words: {0}".format(len(word_num_dict)))
  if valid_phrases != None:
    load_word_num_dict(valid_phrases, word_num_dict)
  load_word_num_dict(test_phrases, word_num_dict)
  print("finish load test words: {0}".format(len(word_num_dict)))
  word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
  print("after trim words: {0}".format(len(word_id_dict)))

  if embedding_model != "none":
    word_embed_model, unknown_word_embed = load_embedding(embedding_model)
    embedding = {}
    for model_name in word_embed_model:
      embedding[model_name] = np.random.random(
              [len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
      embedding[model_name] = init_trainable_embedding(embedding[model_name],
              word_id_dict, word_embed_model[model_name],
              unknown_word_embed)
    embedding["none"] = np.random.random(
            [len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding["none"][0] = np.zeros([embed_dim])
    embedding["field"] = np.random.random(
            [slot_num + 1, embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding["field"][0] = np.zeros([embed_dim])

  else:
    embedding = {}
    embedding["none"] = np.random.random(
      [len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding["none"][0] = np.zeros([embed_dim])
    embedding["field"] = np.random.random(
      [slot_num + 1, embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding["field"][0] = np.zeros([embed_dim])

  print("finish initialize word embedding")

  images["train"], images["train_bow_ids"], labels["train"], datasets["train"] \
          = sst_get_trainable_data(train_phrases, word_id_dict,
                   1, max_input_length, is_binary)
  images["test"], images["test_bow_ids"], labels["test"], datasets["test"] \
          = sst_get_trainable_data(test_phrases, word_id_dict,
                  2, max_input_length, is_binary)
  if is_valid == True:
    images["valid"], images["valid_bow_ids"], labels["valid"], datasets["valid"]\
            = sst_get_trainable_data(valid_phrases, word_id_dict,
                     3, max_input_length, is_binary)
  else:
    images["valid"], images["valid_bow_ids"], labels["valid"], datasets["valid"]\
            = None, None, None, None

  return images, labels, datasets, embedding

def read_data_yelp(word_id_dict, word_num_dict, data_path, max_input_length,
                   embedding_model, min_count, train_ratio, valid_ratio,
                   is_valid=False, cache={}):
  """Reads yelp format data. Always returns NHWC format

  Returns:
    sentences: np tensor of size [N, H, W, C=1]
    labels: np tensor of size [N]
  """
  images, labels, datasets = {}, {}, {}
  dataset_name = data_path.split('/')[1]
  print("dataset_name: {0}".format(dataset_name))

  if len(cache) == 0:
    print("-" * 80)
    print("Reading data")

    train_file_name = os.path.join(data_path, 'train.csv')
    test_file_name = os.path.join(data_path, 'test.csv')
    if is_valid == False:
      train_phrases = yelp_load_phrases(train_file_name, train_ratio)
      valid_phrases = None
    else:
      valid_phrases = []
      train_phrases = yelp_load_phrases(train_file_name,
                                        train_ratio,
                                        valid_ratio,
                                        valid_phrases)
    test_phrases = yelp_load_phrases(test_file_name)
    cache["train"] = train_phrases
    cache["valid"] = valid_phrases
    cache["test"] = test_phrases
    print("finish load data")
  else:
    train_phrases = cache["train"]
    valid_phrases = cache["valid"]
    test_phrases = cache["test"]

  #get word_id_dict
  word_id_dict["<pad>"] = 0
  word_id_dict["<unknown>"] = 1

  load_word_num_dict(train_phrases, word_num_dict)
  print("finish load train words: {0}".format(len(word_num_dict)))
  if valid_phrases != None:
    load_word_num_dict(valid_phrases, word_num_dict)
  load_word_num_dict(test_phrases, word_num_dict)
  print("finish load test words: {0}".format(len(word_num_dict)))
  word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
  print("after trim words: {0}".format(len(word_id_dict)))

  word_embed_model, unknown_word_embed = load_embedding(embedding_model)
  embedding = {}
  for model_name in word_embed_model:
    embedding[model_name] = np.random.random(
            [len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding[model_name] = init_trainable_embedding(embedding[model_name],
            word_id_dict, word_embed_model[model_name], unknown_word_embed)
  embedding["none"] = np.random.random(
          [len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
  embedding["none"][0] = np.zeros([embed_dim])
  embedding["field"] = np.random.random(
          [slot_num + 1, embed_dim]).astype(np.float32) / 2.0 - 0.25
  embedding["field"][0] = np.zeros([embed_dim])

  print("slot num: {0}".format(slot_num))
  print("finish initialize word embedding")

  images["train"], images["train_bow_ids"], labels["train"], datasets["train"] \
          = yelp_get_trainable_data(train_phrases, word_id_dict, dataset_name,
                                    word_embed_model, 1, max_input_length)
  images["test"], images["test_bow_ids"], labels["test"], datasets["test"] \
          = yelp_get_trainable_data(test_phrases, word_id_dict, dataset_name,
                                    word_embed_model, 2, max_input_length)
  if is_valid == True:
    images["valid"], images["valid_bow_ids"], labels["valid"], datasets["valid"]\
            = yelp_get_trainable_data(valid_phrases, word_id_dict, dataset_name,
                                      word_embed_model, 3, max_input_length)
  else:
    images["valid"], images["valid_bow_ids"], labels["valid"], datasets["valid"]\
            = None, None, None, None

  return images, labels, datasets, embedding

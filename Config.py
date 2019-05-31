from dataloader import ChineseDatasetReader
from models import LstmTagger, MultiHeadSelection, ChineseSentenceTaggerPredictor
from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np

import json

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models import Model, crf_tagger
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor


class Config(object):
    def __init__(self):
        relation_vocab_path = './raw_data/chinese/relation_vocab.json'

        self.hidden_dim = 300
        self.relation_vocab = json.load(open(relation_vocab_path, 'r'))
        self.relation_num = len(self.relation_vocab)

        self.binary_threshold = 0.6

if __name__ == "__main__":
    reader = ChineseDatasetReader()
    # train_dataset = reader.read('raw_data/chinese/train_data.json')
    validation_dataset = reader.read('raw_data/chinese/dev_data.json')
    test_dataset = reader.read('tests/fixtures/chinese_test_data.json')

    # vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    # vocab = Vocabulary.from_instances(validation_dataset)
    vocab = Vocabulary.from_instances(test_dataset)

    config = Config()

    EMBEDDING_DIM = 200
    HIDDEN_DIM = 300
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    # model = LstmTagger(word_embeddings, lstm, vocab)
    model = MultiHeadSelection(config, word_embeddings, lstm, vocab)
    # model = crf_tagger.CrfTagger(vocab=vocab, encoder=lstm, text_field_embedder=word_embeddings)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=2,
                              sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=test_dataset,
                      validation_dataset=test_dataset,
                      patience=5,
                      num_epochs=10,
                      cuda_device=cuda_device)

    trainer.train()

    # evaluation
    sentence = "《网游之最强时代》是创世中文网连载的小说，作者是江山"
    predictor = ChineseSentenceTaggerPredictor(model, dataset_reader=reader)
    tags = predictor.predict("《网游之最强时代》是创世中文网连载的小说，作者是江山")['span_tags']
    print('*' * 30)
    # print(tags)
    print(''.join(tags))
    print(''.join([
        'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'B', 'I', 'I', 'I',
        'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I'
    ]))

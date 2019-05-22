from dataloader import ChineseDatasetReader
from models import LstmTagger
from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

if __name__ == "__main__":
    reader = ChineseDatasetReader()
    # train_dataset = reader.read('raw_data/chinese/train_data.json')
    # validation_dataset = reader.read('raw_data/chinese/dev_data.json')
    test_dataset = reader.read('tests/fixtures/chinese_test_data.json')

    # vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    vocab = Vocabulary.from_instances(test_dataset, min_count={'text':0, 'bio':0})
    print(vocab.get_vocab_size('text'))
    print(vocab.get_vocab_size('bio'))
    namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int))

    for data in test_dataset:
        data.count_vocab_items(namespace_token_counts)
    print(namespace_token_counts)
    print(data.fields['text'])

    exit()

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('text'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)
    # if torch.cuda.is_available():
    if 1 == 2:
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=20,
                              sorting_keys=[("text", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=test_dataset,
                      validation_dataset=test_dataset,
                      patience=10,
                      num_epochs=1000,
                      cuda_device=cuda_device)

    print(vocab.get_vocab_size('text'))
    print(vocab.get_vocab_size('bio'))
    print(vocab.get_token_to_index_vocabulary('text'))

    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("《网游之最强时代》是创世中文网连载的小说，作者是江山")['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print(''.join(
        [model.vocab.get_token_from_index(i, 'bio') for i in tag_ids]))
    print(''.join([
        'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'B', 'I', 'I', 'I',
        'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I'
    ]))

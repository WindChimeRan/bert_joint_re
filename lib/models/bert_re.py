from typing import Iterator, List, Dict, Any, Set, Optional
from overrides import overrides

from lib.metrics import F1Selection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, crf_tagger
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

# from lib.models.gpu_mem_track import MemTracker
# import inspect

# TODO: rewrite crf? for computing efficiency.
class MultiHeadSelection(Model):
    def __init__(self,
                 config,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 tagger: Model = crf_tagger.CrfTagger) -> None:
        super().__init__(vocab)
        # TODO
        self.config = config
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.tagger = tagger(vocab=vocab,
                             encoder=self.encoder,
                             text_field_embedder=self.word_embeddings)
        self.relation_emb = Embedding(num_embeddings=config.relation_num,
                                      embedding_dim=50)

        self.selection_u = nn.Linear(config.hidden_dim, 50)
        self.selection_v = nn.Linear(config.hidden_dim, 50)
        self.selection_uv = nn.Linear(100, 50)

        self.selection_loss = nn.BCEWithLogitsLoss()

        self.accuracy = F1Selection()

    def inference(self, tokens, span_dict, selection_logits, output):
        span_dict = self.tagger.decode(span_dict)
        output['span_tags'] = span_dict['tags']

        selection_tags = torch.sigmoid(
            selection_logits) > self.config.binary_threshold
        output['selection_triplets'] = self.selection_decode(tokens, span_dict['tags'], selection_tags)

        return output

    @overrides
    def forward(
            self,  # type: ignore
            tokens: Dict[str, torch.LongTensor],
            tags: torch.LongTensor = None,
            selection: torch.FloatTensor = None,
            spo_list: Optional[List[Dict[str, str]]] = None,
            # pylint: disable=unused-argument
            **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(self.word_embeddings(tokens), mask)

        output = {}

        if tags is not None:
            span_dict = self.tagger(tokens, tags)
            span_loss = span_dict['loss']
        else:
            span_dict = self.tagger(tokens)
            span_loss = 0

        # forward multi head selection
        u = torch.tanh(self.selection_u(encoded_text)).unsqueeze(1)
        v = torch.tanh(self.selection_v(encoded_text)).unsqueeze(2)
        u = u + torch.zeros_like(v)
        v = v + torch.zeros_like(u)
        uv = torch.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # if inference
        output = self.inference(tokens, span_dict, selection_logits, output)
        self.accuracy(output['selection_triplets'], spo_list)

        selection_dict = {}
        if selection is not None:
            selection_loss = self.selection_loss(selection_logits, selection)
            selection_dict['loss'] = selection_loss
            output['loss'] = span_loss + selection_loss

        return output

    def selection_decode(self, tokens, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        # selection_tags[0, 0, 1, 1] = 1
        # temp

        text = [[
            self.vocab.get_token_from_index(token,
                                            namespace='tokens')
            for token in instance_token
        ] for instance_token in tokens['tokens'].tolist()]

        def find_entity(pos, text, sequence_tags):
            entity = []

            if len(sequence_tags) < len(text):
                return 'NA'

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, o, p, s = idx[i].tolist()
            object = find_entity(o, text[b], sequence_tags[b])
            subject = find_entity(s, text[b], sequence_tags[b])
            predicate = self.config.relation_vocab_from_idx[p]
            if object != 'NA' and subject != 'NA':
                triplet = {'object': object, 'predicate': predicate, 'subject': subject}
                result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.accuracy.get_metric(reset)


class LstmTagger(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: Dict[str, torch.Tensor],
                bio: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(text)
        embeddings = self.word_embeddings(text)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if bio is not None:
            self.accuracy(tag_logits, bio, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, bio, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Predictor.register('chinese-tagger')
class ChineseSentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """

    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = dataset_reader._tokenizer

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]

        return self._dataset_reader.text_to_instance(sentence)
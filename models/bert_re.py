from typing import Iterator, List, Dict, Any
import torch
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

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


# TODO: rewrite crf? for computing efficiency.
class MultiHeadSelection(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 tagger: Model = crf_tagger.CrfTagger) -> None:
        super().__init__(vocab)
        # TODO
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.tagger = tagger(vocab=vocab,
                             encoder=self.encoder,
                             text_field_embedder=self.word_embeddings)
        # self.relation_emb = torch.nn.Embedding(vocab.get_vocab_size('relations'), 200)
        self.relation_emb = Embedding(
            num_embeddings=vocab.get_vocab_size('relations'),
            embedding_dim=200)

    @overrides
    def forward(
            self,  # type: ignore
            tokens: Dict[str, torch.LongTensor],
            tags: torch.LongTensor = None,
            selection: torch.LongTensor = None,
            metadata: List[Dict[str, Any]] = None,
            # pylint: disable=unused-argument
            **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(self.word_embeddings(tokens), mask)

        output = {}

        if tags is not None:
            span_dict = self.tagger(tokens, tags)
            span_loss = span_dict['loss']
            output['loss'] = span_loss
        else:
            span_dict = self.tagger(tokens)
        
        span_dict = self.tagger.decode(span_dict)
        output['span_tags'] = span_dict['tags'] 
        
        span_tags = span_dict['tags']
        selection_dict = {}
        if selection is not None:
            selection_loss = 0
            selection_dict['loss'] = selection_loss
            if 'loss' in output:
                output['loss'] += selection_loss
            else:
                output['loss'] = selection_loss
        
        return output
        
        



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
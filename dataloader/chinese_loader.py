from typing import Dict, List, Tuple
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, ArrayField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, CharacterTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# two seperate data reader for self-att network in ner setting


@DatasetReader.register("chinese")
class ChineseDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sentence from baidupedia, in which triplets can be extracted.
    eg:
    {"postag": [{"word": "《", "pos": "w"}, {"word": "我们的少年时代", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "程砚秋", "pos": "nr"}, {"word": "在", "pos": "p"}, {"word": "该剧", "pos": "r"}, {"word": "中", "pos": "f"}, {"word": "饰演", "pos": "v"}, {"word": "富家千金", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "陶西", "pos": "v"}, {"word": "前女友", "pos": "n"}],
    "text": "《我们的少年时代》程砚秋在该剧中饰演富家千金，陶西前女友",
    "spo_list": [{"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "程砚秋", "subject": "我们的少年时代"}]}

    Expected format for each input line: {"tokens": "text", "text": "text", "spo_list": "text"}
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``
    where the ``label`` is derived from the venue of the paper.
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s",
                        file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                instance = json.loads(line)
                text = instance['text']
                spo_list = instance['spo_list']
                if not self.check_valid(text, spo_list):
                    continue
                yield self.text_to_instance(text, spo_list)

    @overrides
    def text_to_instance(self,
                         text: str,
                         spo_list: List[Dict[str, str]] = None) -> Instance:
        # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'tokens': text_field}
        # TODO allennlp.data.fields.array_field.ArrayField for multi-head-selection

        if spo_list is not None:
            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio: List[str] = self.spo_to_bio(text, entities)

            # fields['spo_list'] = ListField(spo_list)
            # fields['entities'] = ListField(entities)
            # fields['relations'] = ListField(relations)

            fields['tags'] = SequenceLabelField(labels=bio,
                                               sequence_field=text_field)
            # selection = self.spo_to_selection(text, spo_list)
            # fields['selection'] = None
        return Instance(fields)

    def check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        for t in spo_list:
            if t['object'] not in text or t['subject'] not in text:
                return False
        return True

    def spo_to_selection(self, text: str, spo_list: List[Dict[str, str]]):
        # TODO
        return None

    def spo_to_entities(self, text: str,
                        spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t['object'] for t in spo_list) | set(t['subject']
                                                            for t in spo_list)
        return list(entities)

    def spo_to_relations(self, text: str,
                         spo_list: List[Dict[str, str]]) -> List[str]:
        return [t['predicate'] for t in spo_list]

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        bio = ['O'] * len(text)
        for e in entities:
            begin = text.find(e)
            end = begin + len(e) - 1

            assert end <= len(text)

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio

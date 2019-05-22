# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from dataloader import ChineseDatasetReader

import os


class TestChineseDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = ChineseDatasetReader()
        print(os.getcwd())
        instances = ensure_list(
            reader.read('tests/fixtures/chinese_test_data.json'))

        instance1 = {
            "postag": [{
                "word": "《",
                "pos": "w"
            }, {
                "word": "网游之最强时代",
                "pos": "nw"
            }, {
                "word": "》",
                "pos": "w"
            }, {
                "word": "是",
                "pos": "v"
            }, {
                "word": "创世中文网",
                "pos": "nz"
            }, {
                "word": "连载",
                "pos": "v"
            }, {
                "word": "的",
                "pos": "u"
            }, {
                "word": "小说",
                "pos": "n"
            }, {
                "word": "，",
                "pos": "w"
            }, {
                "word": "作者",
                "pos": "n"
            }, {
                "word": "是",
                "pos": "v"
            }, {
                "word": "江山",
                "pos": "n"
            }],
            "text":
            "《网游之最强时代》是创世中文网连载的小说，作者是江山",
            "spo_list": [{
                "predicate": "连载网站",
                "object_type": "网站",
                "subject_type": "网络小说",
                "object": "创世中文网",
                "subject": "网游之最强时代"
            }, {
                "predicate": "作者",
                "object_type": "人物",
                "subject_type": "图书作品",
                "object": "江山",
                "subject": "网游之最强时代"
            }]
        }

        fields = instances[0].fields
        assert list(
            instance1["text"]) == [t.text for t in fields["text"].tokens]

        # assert instance1["spo_list"] == fields["spo_list"].field_list
        # assert set(['江山', '网游之最强时代',
        #             '创世中文网']) == set(fields["entities"].field_list)

        # assert sorted(['连载网站', '作者']) == sorted(fields["relations"])
        assert fields["bio"].labels == [
            'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'B', 'I', 'I',
            'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I'
        ]

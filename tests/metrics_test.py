# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from lib.dataloader import ChineseDatasetReader
from lib.metrics import F1Selection
import os


class TestChineseDatasetMetrics(AllenNlpTestCase):
    def setUp(self):
        super(TestChineseDatasetMetrics, self).setUp()
        self.reader = ChineseDatasetReader()
        self.instances = ensure_list(
            self.reader.read('tests/fixtures/chinese_test_data.json'))
        self.instance1 = {
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
        print('setup!')
    def test_f1(self):
        pass
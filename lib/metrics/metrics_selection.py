from typing import List, Optional, Union, Dict, overload

import torch
from overrides import overrides

from allennlp.training.metrics.metric import Metric

@Metric.register("f1_selection")
class F1Selection(Metric):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    @overrides
    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    @overrides
    def get_metric(self,
                   reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {
                "precision": p,
                "recall": r,
                "fscore": f1
        }
        # print(', '.join(["%s: %.4f" % (name, value)
        #               for name, value in
        #               result.items() if not name.startswith("_")]) + " ||")
        return result

    @overrides
    def __call__(self,
                 predictions: List[List[Dict[str, str]]],
                 gold_labels: List[List[Dict[str, str]]],
                 mask: Optional[torch.Tensor] = None):
        

        
        for g, p in zip(gold_labels, predictions):
            g_set = set('_'.join((gg['object'], gg['predicate'], gg['subject'])) for gg in g)
            p_set = set('_'.join((pp['object'], pp['predicate'], pp['subject'])) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)
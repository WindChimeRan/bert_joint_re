import json
import os

schema_path = './raw_data/chinese/all_50_schemas'
relation_vocab_path = './raw_data/chinese/relation_vocab.json'

if __name__ == "__main__":
    if not os.path.exists(schema_path):
        raise FileNotFoundError('file not found')
    relation_vocab = {}
    i = 0
    for line in open(schema_path, 'r'):
        relation = json.loads(line)['predicate']
        if relation not in relation_vocab:
            relation_vocab[relation] = i
            i += 1
    relation_vocab['N'] = i
    json.dump(relation_vocab, open(relation_vocab_path, 'w'), ensure_ascii=False)
import json
import os

schema_path = './raw_data/chinese/all_50_schemas'
relation_vocab_path = './raw_data/chinese/relation_vocab.json'

if __name__ == "__main__":
    if not os.path.exists(schema_path):
        raise FileNotFoundError('file not found')
    relation_vocab = {}
    for i, line in enumerate(open(schema_path, 'r')):
        relation_vocab[json.loads(line)['predicate']] = i
    json.dump(relation_vocab, open(relation_vocab_path, 'w'), ensure_ascii=False)
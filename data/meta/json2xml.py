import re
import json
from efficiency.log import fwrite
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Dataset converter')
parser.add_argument('-dataset', default='webnlg_sent', help='which dataset to preprocess')
args = parser.parse_args()

folder = 'data/{}/'.format(args.dataset)

file_head = '<?xml version="1.0" encoding="UTF-8" ?><doc>'
file_tail = '</doc>'

doc_tail = '</abstract></text>'
doc_head_templ = '<text id="{}"> <title> A Title </title> <abstract>'


def data2xml(data, doc_prefix='A01'):
    content = file_head
    content += '\n\n'
    relations = []

    for ix, item in enumerate(data):
        doc_id = '{}-{:04d}'.format(doc_prefix, ix)
        doc_head = doc_head_templ.format(doc_id)

        content += doc_head
        content += '\n\n'

        ner2ent = item['ner2ent']
        target = item['target']
        for tag in {'<i>', '</i>'}:
            target = target.replace(tag, '')

        words = target.split()
        ners = {ner: words.index(ner) for ner in item['ner2ent'] if ner in words}
        ners = sorted(ners.items(), key=lambda i: i[-1])

        ner2text = {}
        ent2entid = {}
        for ent_i, (ner, _) in enumerate(ners):
            ent_i += 1
            ent = ner2ent[ner]
            tag_id = '{}.{}'.format(doc_id, ent_i)
            ner2text[ner] = '<entity id="{}">{}</entity>'.format(tag_id, ent)
            ent2entid[ent] = (ent_i, tag_id)

        doc = [ner2text.get(w, w) for w in words]
        doc = ' '.join(doc)
        doc = doc.replace('& ','&amp; ')

        content += doc

        content += '\n\n'
        content += doc_tail
        content += '\n\n'

        triples = item['triples']
        for head, rel, tail in triples:
            if (head not in ent2entid) or (tail not in ent2entid):
                continue
            head_ent_i, head_tag_i = ent2entid[head]
            tail_ent_i, tail_tag_i = ent2entid[tail]
            rel = rel.replace('(', '-')
            rel = rel.replace(')', '-')

            if head_ent_i < tail_ent_i:
                relation = '{}({},{})'.format(rel, head_tag_i, tail_tag_i)
            else:
                relation = '{}({},{},REVERSE)'.format(rel, head_tag_i,
                                                      tail_tag_i)
            relations.append(relation)

    content += file_tail
    return content, relations

class NLP:
    def __init__(self):
        from transformers import BertTokenizer
        bert_type = "bert-base-uncased"
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_type)

    def word_tokenize(self, text):
        toks = self.bert_tokenizer.tokenize(text)
        tok_ls = []
        for tok in toks:
            if tok.startswith('##'):
                tok_ls[-1] += tok[2:]
            else:
                tok_ls.append(tok)
        new_text = ' '.join(tok_ls)
        return new_text


def data_tokenize(data):

    nlp = NLP()
    for ix, line in tqdm(enumerate(data)):
        target = line['target']
        ner2ent = line['ner2ent']
        triples = line['triples']

        ner_converter = {ner: ner.replace('_', '').lower() for ner in ner2ent}
        ner2ent = {ner_converter[ner]: nlp.word_tokenize(ent)
                   for ner, ent in ner2ent.items()}
        triples = [(nlp.word_tokenize(h), r, nlp.word_tokenize(t)) for h,r,t in triples]
        target = ' '.join([ner_converter.get(i, i) for i in target.split()] )
        target = nlp.word_tokenize(target)

        line['target'] = target
        line['ner2ent'] = ner2ent
        line['triples'] = triples
    return data

for split_ix, split in enumerate(['train', 'valid', 'test']):
    with open('{}{}.json'.format(folder, split)) as f:
        data = json.load(f)

    data = data_tokenize(data)
    content, relations = data2xml(data, doc_prefix='B{:02d}'.format(split_ix))
    fwrite(content, folder + split + '.xml')
    fwrite('\n'.join(relations), folder + split + '.key.txt')


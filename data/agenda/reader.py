import json
from collections import Counter
from efficiency.function import shell, flatten_list
from efficiency.log import fwrite, show_var
from efficiency.nlp import NLP


class Cleaner:
    @property
    def filter_dic(self):
        return {
            ('unprocessed.train.json', 134248,
             '"types": "<task> <method> <otherscientificterm> <task> <method> <method> <method> <method>",'
             ):
                '"types": "<task> <method> <otherscientificterm> <task> <method> <method> <method> <method> <method>",'
            ,
        }

    def __init__(self):
        self.fname_ends = [k[0] for k in self.filter_dic]

    def clean(self, filename):
        fname_end = filename.rsplit('/', 1)[-1]

        if fname_end not in self.fname_ends: return

        with open(filename, encoding="utf-8", errors='ignore') as f:
            lines = []
            content = f.readlines()
            for line_ix, line in enumerate(content):
                line = self.filter_line(fname_end, line_ix, line)
                if line: lines.append(line)
        if lines != content:
            # import pdb;
            # pdb.set_trace()
            fwrite(''.join(lines), filename)

    def filter_line(self, fname_end, line_ix, line):
        text = line.strip()
        key = (fname_end, line_ix, text)
        if key in self.filter_dic:
            # fwrite(json.dumps(key) + '\n', 'temp.txt', mode='a')
            new_text = self.filter_dic[key]
            if not new_text: return False
            line = line.replace(text, new_text)

        return line


class DataConverter:
    REL_SEP = ' -- '

    def __init__(self):
        self.bad_alignments = 0
        self.nlp = NLP()

    def get_triples(self, relations):
        triples = []
        for rel in relations:
            toks = rel.split(self.REL_SEP)
            tok_ix, predicate = [(ix, tok) for ix, tok in enumerate(toks)
                                 if tok.isupper()][0]
            subj = self.REL_SEP.join(toks[:tok_ix])
            obj = self.REL_SEP.join(toks[tok_ix + 1:])
            if len(predicate.split()) > 1:
                import pdb;pdb.set_trace()
            predicate = predicate.replace(' ', '-')
            triples.append((subj, predicate, obj))
        return triples

    def get_ner2ent(self, entities, types):
        from collections import defaultdict

        ners = defaultdict(list)
        ner2ent = {}
        types = types.split()

        if len(entities) != len(types):
            entities = [ent for ent in entities if ent]
            types = [typ for typ in types if typ]
            if len(entities) != len(types):
                self.bad_alignments += 1  # TODO: here are the bad alignments
        # manual fix of ` {'OTHERSCIENTIFICTERM_0': ''} `
        ent2typ = {k: v for k, v in zip(entities, types) if v and k}

        for ent, typ in ent2typ.items():
            typ = typ[1:-1]
            ner2ent['<{typ}_{ix}>'.format(typ=typ, ix=len(ners[typ]))] = ent
            ners[typ] += [ent]
        return ner2ent

    def get_target(self, target_txt, ner2ent):
        target = target_txt
        for ner, ent in ner2ent.items():
            target = target.replace(' {} '.format(ent), ' {} '.format(ner))

        target = ' '.join([word[1:-1].upper() if word in ner2ent else word
                           for word in target.split()])
        return target

    def get_target_txt(self, target_txt):
        return self.nlp.word_tokenize(target_txt)

    @staticmethod
    def check_data(triples, target, target_txt, ner2ent, line):
        if len(set(ner2ent.keys())) != len(set(ner2ent.values())): import \
            pdb;pdb.set_trace()


        ents = flatten_list([(subj, obj) for subj, predi, obj in triples])
        if set(ents) - set(ner2ent.values()):
            import pdb;
            pdb.set_trace()
        return

        ent_not_mentioned = [ent_ix for ent_ix, ent in enumerate(ents) if
                             ent not in target_txt]
        if ent_not_mentioned:
            import pdb;
            pdb.set_trace()

        target_keywords = [word for word in target if
                           word.isupper() and '_' in word]
        if set(target_keywords) != set(ner2ent.keys()): import \
            pdb;pdb.set_trace()

        rels = [predi for subj, predi, obj in triples]
        if_rels_are_all_one_word = all(len(rel) == 1 for rel in rels)
        if not if_rels_are_all_one_word: import pdb;pdb.set_trace()




def get_box(data, save_file):
    formatted = []
    converter = DataConverter()
    for line in data:
        relations = line['relations']
        entities = line['entities']
        types = line['types']
        target_txt = line['abstract_og']

        triples = converter.get_triples(relations)
        ner2ent = converter.get_ner2ent(entities, types)

        target = converter.get_target(target_txt, ner2ent)
        target_txt = converter.get_target_txt(target_txt)

        ner2ent = {k[1:-1].upper(): v for k, v in ner2ent.items()}

        converter.check_data(triples, target, target_txt, ner2ent, line)

        new_line = {
            'triples': triples,
            'target': target,
            # 'target_txt': target_txt,
            'ner2ent': ner2ent,
        }
        formatted += [new_line]
    print('[Info] Saving {} data to {}'.format(len(formatted), save_file))
    fwrite(json.dumps(formatted, indent=4), save_file)

    show_var(['converter.bad_alignments'])
    return


def analyze(data):
    show_var(["len(data)"])
    diff = [len(rel.split(' -- ')) != 3 for line in data for rel in
            line['relations']]
    rels = [rel.split(' -- ')[1] for line in data for rel in
            line['relations']]
    n_rels = [len(line['relations']) for line in data]
    n_words = [len(line['abstract_og'].split()) for line in data]
    cnt_rels = Counter(rels)
    cnt_n_rels = Counter(n_rels)
    min(list(Counter(n_words).keys()))
    import pdb;
    pdb.set_trace()
    sum(diff)


def tokenize(data, field):
    import spacy
    spacy_en = spacy.load('en')

    def spacy_tok(text):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split()).lower()
        toks = [tok.text for tok in spacy_en.tokenizer(text)]
        return ' '.join(toks)

    tok_data = [
        spacy_tok(line) if field in ['Summary'] else
        ';;\t'.join(['\t'.join([spacy_tok(item) for item in triple.split('\t')])
                     for triple in line.split(';;\t')]) if field in ['Triples']
        else line

        for line in data
    ]
    return tok_data


def download():
    print('[Info] Downloading AGENDA data...')
    cmd = \
        'curl -O https://raw.githubusercontent.com/rikdz/GraphWriter/master/data/unprocessed.tar.gz; ' \
        'mkdir -p data/agenda/raw; ' \
        'rm data/agenda/raw/* 2>/dev/null; ' \
        'tar -C data/agenda/raw -xvf unprocessed.tar.gz; ' \
        'mv data/agenda/raw/unprocessed.val.json data/agenda/raw/unprocessed.valid.json; ' \
        'rm unprocessed.tar.gz'
    shell(cmd)


def main():
    download()

    file_tmpl = 'data/agenda/raw/unprocessed.{}.json'
    save_dir = 'data/agenda/'
    types = 'train valid test'.split()
    cleaner = Cleaner()
    for typ in types:
        file = file_tmpl.format(typ)
        cleaner.clean(file)
        with open(file) as f: data = json.load(f)
        get_box(data, save_file=save_dir + typ + '.json')
        shell('rm ' + file)


if __name__ == "__main__":
    main()

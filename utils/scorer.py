import numpy as np

__all__ = ['get_f1']

def get_f1(gold_file, pred_file, is_file_content=False, precision=4 , no_rel_name = None , logger = None):
    from sklearn.metrics import f1_score , precision_score , recall_score

    rel_str2id = _get_rels([gold_file, pred_file], is_file_content , no_rel_name = no_rel_name)
    classes = rel_str2id.classes_

    gold_labels = _get_data(gold_file, classes, is_file_content=is_file_content)
    pred_labels = _get_data(pred_file, classes, is_file_content=is_file_content)
    keys = sorted(set(gold_labels.keys()) | set(pred_labels.keys()))

    gold = [gold_labels.get(k, no_rel_name) for k in keys]
    pred = [pred_labels.get(k, no_rel_name) for k in keys]
    gold = rel_str2id.transform(gold)
    pred = rel_str2id.transform(pred)

    #import pdb
    #pdb.set_trace()

    pos_classes = set(classes) - {no_rel_name}
    pos_labels = rel_str2id.transform(list(pos_classes))

    f1_micro = f1_score(gold, pred, average='micro', labels=pos_labels, zero_division=0)

    preci = precision_score(gold, pred, average='macro', labels=pos_labels, zero_division=0)
    recal =    recall_score(gold, pred, average='macro', labels=pos_labels, zero_division=0)

    if logger is not None:
        logger.log ("precision / recall = %.4f , %.4f" % (float(preci) , float(recal)))
    if (preci + recal) == 0:
        f1_macro = 0
    else:
        f1_macro = (2 * preci * recal) / (preci + recal)

    return round(f1_micro, precision), round(f1_macro, precision)

    # gold_matrix = _file2matrix(gold_file)

def _get_data(file, classes, is_file_content=False):
    if not is_file_content:
        with open(file) as f:
            file_content = f.read()
    else:
        file_content = file
    data = [line.strip() for line in file_content.split('\n') if line.strip()]

    rel_labels = {}
    for line in data:
        rel_str = line.rsplit('(', 1)[0]

        entity_str = line.rsplit('(', 1)[-1][:-1]
        ent0_str, ent1_str, *reverse = entity_str.split(',')
        doc_id = ent0_str.split('.', 1)[0]
        ent0 = int(ent0_str.split('.', 1)[-1]) - 1
        ent1 = int(ent1_str.split('.', 1)[-1]) - 1
        if reverse:
            key = (doc_id, (ent1, ent0))
        else:
            key = (doc_id, (ent0, ent1))
        if rel_str not in classes:
            from efficiency.log import show_var
            show_var(['file', 'data'])
            print('[Warn] rel_str({}) is not among pre-defined classes({})'.format(line, classes))
        else:
            rel_labels[key] = rel_str
    # assert len(rel_labels) == len(data), "# lines does not match with # data"

    return rel_labels


def _get_rels(file_list, is_file_content=False , no_rel_name = "NONE"):
    from sklearn.preprocessing import LabelEncoder

    data = []
    for file in file_list:
        if is_file_content:
            data += [line.strip() for line in file.split('\n') if line.strip()]
        else:
            with open(file) as f:
                data += [line.strip() for line in f if line.strip()]

    relations = [no_rel_name]
    for line in data:
        rel = line.rsplit('(', 1)[0]

        relations.append(rel)
    rel_str2id = LabelEncoder()
    rel_str2id.fit(relations)
    # rel_str2id.classes_
    return rel_str2id

def _file2matrix(rel_file, data_file):
    import xmltodict

    with open(rel_file) as f:
        data = [line.strip() for line in f if line.strip()]

    matrices = {}

    with open(data_file) as f:
        xml_data = f.read()
        docs = xmltodict.parse(xml_data)
        docs = docs['doc']['text']
        for doc in docs:
            doc_id = doc['@id']
            entities = doc['abstract']['entity']
            num_ents = \
            sorted(int(entity['@id'].rsplit('.')[-1]) for entity in entities)[
                -1]
            if num_ents < len(entities):
                import pdb;
                pdb.set_trace()
            matrices[doc_id] = np.zeros((num_ents, num_ents), dtype=object)

    for line in data:
        rel_str = line.rsplit('(', 1)[0]

        entity_str = line.rsplit('(', 1)[-1][:-1]
        ent0_str, ent1_str, *reverse = entity_str.split(',')
        doc_id = ent0_str.split('.', 1)[0]
        ent0 = int(ent0_str.split('.', 1)[-1]) - 1
        ent1 = int(ent1_str.split('.', 1)[-1]) - 1
        if reverse:
            matrices[doc_id][ent1, ent0] = rel_str
        matrices[doc_id][ent0, ent1] = rel_str

    import pdb;
    pdb.set_trace()

    'RESULT(E03-2013.7,E03-2013.11,REVERSE)' \
    'RESULT(E03-2013.7,E03-2013.11)'


def test():
    # data_file = 'data/webnlg_sent/valid.xml'
    # data_file = 'data/semeval_2018_task7/1.1.text.xml'
    # rel_file = 'data/webnlg_sent/valid.key.txt'
    rel_file = 'data/semeval_2018_task7/1.1.relations.txt'

    pred_file = 'tmp_12692505.txt'
    pred_file = 'tmp_18033851.txt'
    score = get_f1(rel_file, pred_file)
    print(score)
    # _file2matrix(rel_file, data_file)


if __name__ == '__main__':
    test()

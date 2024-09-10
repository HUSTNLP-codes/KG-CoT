import torch
import os
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from utils.misc import invert_dict

def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, entity_range, origin_question = batch
    topic_entity = torch.stack(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = torch.stack(answer)
    entity_range = torch.stack(entity_range)
    return topic_entity, question, answer, entity_range, origin_question


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, entity_range, origin_question = self.questions[index]
        topic_entity = self.toOneHot(topic_entity)
        answer = self.toOneHot(answer)
        entity_range = self.toOneHot(entity_range)
        return topic_entity, question, answer, entity_range, origin_question

    def __len__(self):
        return len(self.questions)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.ent2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, input_dir, fn, bert_name, ent2id, rel2id, id2name, batch_size, training=False):
        print('Reading questions from {}'.format(fn))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2name = id2name
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)



        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        for line in open(os.path.join(input_dir, 'KG/triplets1.txt')):
            l = line.strip().split('\t')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            sub_map[s].append((p, o))
            so_map[(s, o)].append(p)


        data = []
        for line in open(fn):
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 3:
                continue
            question = line[0].strip()
            head = line[1].replace('[', '').replace(']', '').split('|')
            ans = line[2].strip().split('|')

            entity_range = set()
            for h in head:  # 2-hop
                for p1, o1 in sub_map[h]:
                    entity_range.add(o1)
                    for p2, o2 in sub_map[o1]:
                        entity_range.add(o2)
                        # for p3, o3 in sub_map[o2]:
                        #     entity_range.add(o3)

            entity_range = [ent2id[o] for o in entity_range]

            head = [ent2id[_] for _ in head]
            origin_question = question.strip()
            question = self.tokenizer(question.strip(), max_length=256, padding='max_length', return_tensors="pt")
            # print(ans)
            ans = [ent2id[a] for a in ans]
            data.append([head, question, ans, entity_range, origin_question])

        print('data number: {}'.format(len(data)))
        
        dataset = Dataset(data, ent2id)


        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )


def load_data(input_dir, bert_name, batch_size):
    cache_fn = os.path.join(input_dir, 'processed_data/processed.pt')
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples, train_data, test_data = pickle.load(fp)
        print('Train number: {}, test number: {}'.format(len(train_data.dataset), len(test_data.dataset)))
    else:
        print('Read data...')

        # 图谱数据
        ent2id, id2name, count = {}, {}, 0
        for line in open(os.path.join(input_dir, 'Dic/entity.txt')):
            # split = line.strip().split('\t')
            # if split[0].strip() not in ent2id:
            #     ent2id[split[0].strip()] = count
            #     id2name[count] = split[1].strip()
            #     count = count + 1
            if line.strip() not in ent2id:
                ent2id[line.strip()] = count
                count = count + 1
        count = 0
        rel2id = {}
        for line in open(os.path.join(input_dir, 'Dic/relation.txt')):
            if line.strip() not in rel2id:
                rel2id[line.strip()] = count
                count = count + 1

        triples = []
        for line in open(os.path.join(input_dir, 'KG/triplets.txt')):
            l = line.strip().split('\t')
            s = ent2id[l[0].strip()]
            p = rel2id[l[1].strip()]
            o = ent2id[l[2].strip()]
            triples.append((s, p, o))
            p_rev = rel2id[l[1].strip()+'_reverse']
            triples.append((o, p_rev, s))
        triples = torch.LongTensor(triples)


        train_data = DataLoader(input_dir, os.path.join(input_dir, 'qa_train_webqsp.txt'), bert_name, ent2id, rel2id, id2name, batch_size, training=True)
        test_data = DataLoader(input_dir, os.path.join(input_dir, 'qa_test_webqsp.txt'), bert_name, ent2id, rel2id, id2name, batch_size)
    
        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples, train_data, test_data), fp)

    return ent2id, rel2id, triples, train_data, test_data

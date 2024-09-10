import os
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device



def find_path(args, model, data, device, triples):
    model.eval()
    count = 0
    correct = 0
    correct5 = 0
    correct10 = 0
    hop_count = defaultdict(list)
    file_path = os.path.join(args.input_dir, 'qid2name.txt')
    with open(file_path, 'r', encoding='UTF-8') as readname:
        ent2id, id2name, count = {}, {}, 0
        for line in readname:
            split = line.strip().split('\t')
            if split[0].strip() not in ent2id:
                ent2id[split[0].strip()] = count
                id2name[count] = split[1].strip()
                count = count + 1
    count = 0
    data.id2name = id2name
    with torch.no_grad():
        save_path = os.path.join(args.save_path, 'output_path.json')
        with open(save_path, 'w', encoding='UTF-8') as f:
            for batch in tqdm(data, total=len(data)):
                outputs = model(*batch_device(batch, device)) # [bsz, Esize]
                topic_entities, questions, answers, entity_range, origin_question = batch
                answerslist = []
                for i in range(answers.shape[0]):
                    answerslist.append(answers[i].nonzero().squeeze(-1).tolist())
                # e_score = outputs['e_score'].cpu()
                indeices = outputs['hop_attn'].max(1).indices
                e_score = torch.stack([outputs['ent_probs'][indeices[i]][i] for i in range(answers.shape[0])], dim=0).cpu()
                topk_indices = e_score.topk(30, dim=1).indices.tolist()
                scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
                match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
                count += len(match_score)
                correct += sum(match_score)
                for i in range(len(match_score)):
                    h = outputs['hop_attn'][i].argmax().item()
                    hop_count[h].append(match_score[i])

                for i in range(len(match_score)):
                    # if match_score[i] == 0:
                    if True:
                        # print('================================================================')
                        # question_ids = batch[1]['input_ids'][i].tolist()
                        # question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        # print(' '.join(question_tokens))
                        # topic_id = topic_entities[i].argmax(0).item()
                        # print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        current_entities = topic_entities[i].nonzero().squeeze(-1).tolist()
                        current_path = [data.id2name[_] if data.id2name[_] != '-' else data.id2ent[_] for _ in current_entities]
                        rel_score = []
                        for t in range(indeices[i] + 1):
                            rel_idx = outputs['rel_probs'][t][i].gt(0.5).nonzero().squeeze(1).tolist()
                            tmp_entities, tmp_path, tmp_rel_score = [], [], []
                            for a in range(len(current_entities)):
                                one_hop_graph = triples[triples[:,0] == current_entities[a]]
                                one_hop_rel = one_hop_graph[:,1].tolist()
                                mask = []
                                for j in range(len(one_hop_rel)):
                                    mask.append(one_hop_rel[j] in rel_idx)
                                if len(mask) == 0:
                                    continue
                                one_hop_graph = one_hop_graph[mask]
                                tmp_entities += one_hop_graph[:,2].tolist()
                                for j in range(one_hop_graph.shape[0]):
                                    tmp_rel_score.append((0 if t == 0 else rel_score[a]) + outputs['rel_probs'][t][i][one_hop_graph[j][1].item()].item())
                                    tmp_path.append(current_path[a] + ' --> ' + data.id2rel[one_hop_graph[j][1].item()] + ' --> ' +
                                                    (data.id2name[one_hop_graph[j][2].item()] if data.id2name[one_hop_graph[j][2].item()] != '-' else data.id2ent[one_hop_graph[j][2].item()]))
                            current_entities = tmp_entities
                            current_path = tmp_path
                            rel_score = tmp_rel_score
                        rel_score = [tmp_rel_score[_] / (indeices[i] + 1).item() for _ in range(len(tmp_rel_score))]
                        pred_path = defaultdict(list)
                        for j in range(len(current_entities)):
                            if (current_entities[j] in topk_indices[i]):
                                pred_path[data.id2name[current_entities[j]] if data.id2name[current_entities[j]] != '-' else data.id2ent[current_entities[j]]].append(
                                    {'path':current_path[j], 'average_rel_score':rel_score[j], 'entity_score':e_score[i][current_entities[j]].item(), 'is_golden_answer':current_entities[j] in answerslist[i], 'priority':topk_indices[i].index(current_entities[j])})

                        piece = {'question': origin_question[i],
                                'topic_entity': [data.id2name[_] if data.id2name[_] != '-' else data.id2ent[_] for _ in topic_entities[i].nonzero().squeeze(-1).tolist()],
                                'answer': [data.id2name[_] if data.id2name[_] != '-' else data.id2ent[_] for _ in answers[i].nonzero().squeeze(-1).tolist()],
                                'is_solved': match_score[i] == 1,
                                'top10_path': pred_path}
                        piece_str = json.dumps(piece)
                        f.write(piece_str + '\n')
                        # print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        # print('----')
                        # print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        # print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        # print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        # print(' '.join(question_tokens))
                        # print(outputs['hop_attn'][i].tolist())
                        ans = set(answers[i].gt(0.9).nonzero().squeeze(1).tolist())
                        pred5 = set(e_score[i].topk(5).indices.tolist())
                        pred10 = set(e_score[i].topk(10).indices.tolist())
                        if len(pred5.intersection(ans)) != 0:
                            correct5 += 1
                        if len(pred10.intersection(ans)) != 0:
                            correct10 += 1

    acc = correct / count
    acc5 = correct5 / count
    acc10 = correct10 / count
    print(acc, acc5, acc10)

    return acc




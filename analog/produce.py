import os
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device



def find_path(args, model, data, device):
    model.eval()
    count = 0
    correct = 0
    correct5 = 0
    correct10 = 0
    correct20 = 0
    correct30 = 0
    correct40 = 0
    hop_count = defaultdict(list)
    with torch.no_grad():
        save_path = os.path.join(args.save_path, 'output_path.json')
        with open(save_path, 'w', encoding='UTF-8') as f:
            for batch in tqdm(data, total=len(data)):
                outputs = model(*batch_device(batch, device)) # [bsz, Esize]
                topic_entities, questions, answers, triples, entity_range, origin_question = batch
                answerslist = []
                for i in range(answers.shape[0]):
                    answerslist.append(answers[i].nonzero().squeeze(-1).tolist())
                # e_score = outputs['e_score'].cpu()
                indeices = outputs['hop_attn'].max(1).indices
                e_score = torch.stack([outputs['ent_probs'][indeices[i]][i] for i in range(answers.shape[0])], dim=0).cpu()
                top10_indices = e_score.topk(30, dim=1).indices.tolist()
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
                        target_path, target_entities, target_rel_score = [], [], []
                        rel_score = []
                        for t in range(indeices[i] + 1):
                            rel_idx = outputs['rel_probs'][t][i].gt(0.7).nonzero().squeeze(1).tolist()
                            rel_idx_sup = outputs['rel_probs'][t][i].gt(0.01).nonzero().squeeze(1).tolist()
                            tmp_entities, tmp_path, tmp_rel_score = [], [], []
                            for a in range(len(current_entities)):
                                one_hop_graph = triples[i][triples[i][:,0] == current_entities[a]]
                                one_hop_rel = one_hop_graph[:,1].tolist()
                                mask = []
                                for j in range(len(one_hop_rel)):
                                    mask.append(one_hop_rel[j] in rel_idx)
                                if True not in mask:
                                    mask = []
                                    for j in range(len(one_hop_rel)):
                                        mask.append(one_hop_rel[j] in rel_idx_sup)
                                one_hop_graph = one_hop_graph[mask]
                                tmp_entities += one_hop_graph[:,2].tolist()
                                for j in range(one_hop_graph.shape[0]):
                                    tmp_rel_score.append((0 if t == 0 else rel_score[a]) + outputs['rel_probs'][t][i][one_hop_graph[j][1].item()].item())
                                    tmp_path.append(current_path[a] + ' --> ' + data.id2rel[one_hop_graph[j][1].item()] + ' --> ' +
                                                    (data.id2name[one_hop_graph[j][2].item()] if data.id2name[one_hop_graph[j][2].item()] != '-' else data.id2ent[one_hop_graph[j][2].item()]))

                            current_entities = tmp_entities
                            current_path = tmp_path
                            rel_score = tmp_rel_score
                            for a in range(len(tmp_entities)):
                                if tmp_entities[a] in top10_indices[i]:
                                    target_path.append(tmp_path[a])
                                    target_entities.append(tmp_entities[a])
                                    target_rel_score.append(tmp_rel_score[a] / (t + 1))

                        pred_path = defaultdict(list)
                        for j in range(len(target_path)):
                            pred_path[data.id2name[target_entities[j]] if data.id2name[target_entities[j]] != '-' else data.id2ent[target_entities[j]]].append(
                                {
                                    'path':target_path[j], 'average_rel_score':target_rel_score[j], 'entity_score':e_score[i][target_entities[j]].item(), 'is_golden_answer':target_entities[j] in answerslist[i], 'priority':top10_indices[i].index(target_entities[j])
                                }, )

                        piece = {'question': origin_question[i],
                                'topic_entity': [data.id2name[_] if data.id2name[_] != '-' else data.id2ent[_] for _ in topic_entities[i].nonzero().squeeze(-1).tolist()],
                                'answer': [data.id2name[_] if data.id2name[_] != '-' else data.id2ent[_] for _ in answers[i].nonzero().squeeze(-1).tolist()],
                                'is_solved': match_score[i] == 1,
                                'top10_path': pred_path}

                        print(piece)
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
                        pred20 = set(e_score[i].topk(20).indices.tolist())
                        pred30 = set(e_score[i].topk(30).indices.tolist())
                        pred40 = set(e_score[i].topk(40).indices.tolist())
                        if len(pred5.intersection(ans)) != 0:
                            correct5 += 1
                        if len(pred10.intersection(ans)) != 0:
                            correct10 += 1
                        if len(pred20.intersection(ans)) != 0:
                            correct20 += 1
                        if len(pred30.intersection(ans)) != 0:
                            correct30 += 1
                        if len(pred40.intersection(ans)) != 0:
                            correct40 += 1


    acc = correct / count
    acc5 = correct5 / count
    acc10 = correct10 / count
    acc20 = correct20 / count
    acc30 = correct30 / count
    acc40 = correct40 / count
    print(acc, acc5, acc10, acc20, acc30, acc40)

    return acc




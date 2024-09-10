import os
import torch
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device



def validate(args, model, data, device, verbose=False, thresholds=0.9, sim=0, rerank_flag=False):
    model.eval()
    count = 0
    correct = 0
    correct5 = 0
    correct10 = 0
    hop_count = defaultdict(list)
    hop_att_count = defaultdict(int)
    hop_pred_count = defaultdict(int)
    num_answers_total = 0  # TP + FN
    num_answers_pred_total = 0  # TP + FP
    TP_total = 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device)) # [bsz, Esize]
            topic_entities, questions, answers, triples, entity_range, origin_question = batch
            labels = torch.nonzero(answers)
            answer_list = [[] for _ in range(topic_entities.shape[0])]
            for x in labels:
                answer_list[x[0].item()].append(x[1].item())
            num_answers = sum(len(x) for x in answer_list)
            num_answers_total += num_answers
            # e_score = outputs['e_score'].cpu()
            indeices = outputs['hop_attn'].max(1).indices
            e_score = torch.stack([outputs['ent_probs'][indeices[i]][i] for i in range(answers.shape[0])], dim=0).cpu()
            e_score_answers = torch.where(e_score >= thresholds)
            num_pred = e_score_answers[0].shape[0]
            num_answers_pred_total += num_pred

            TP = 0
            for i in range(e_score_answers[0].shape[0]):
                if e_score_answers[1][i].item() in answer_list[e_score_answers[0][i].item()]:
                    TP += 1
            TP_total += TP

            # ###########################################################################
            # for i in range(topic_entities.shape[0]):
            #     indices = torch.topk(e_score[i], len(answer_list[i])).indices
            #     for j in indices:
            #         if j.item() in answer_list[i]:
            #             TP += 1
            # TP_total += TP
            # ###########################################################################



            topic_entities_idx = torch.nonzero(topic_entities)
            for item in topic_entities_idx:
                e_score[item[0], item[1]] = 0

            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze(-1).tolist()
            count += len(match_score)
            correct += sum(match_score)
            for i in range(len(match_score)):
                ans = set(answers[i].gt(0.9).nonzero().squeeze(1).tolist())
                pred5 = set(e_score[i].topk(3).indices.tolist())
                pred10 = set(e_score[i].topk(20).indices.tolist())
                if len(pred5.intersection(ans)) != 0:
                    correct5 += 1
                if len(pred10.intersection(ans)) != 0:
                    correct10 += 1
            # for i in range(len(match_score)):
            #     h_pred = outputs['hop_attn'][i].argmax().item()
            #     h = hops[i] - 1
            #     hop_count[h].append(match_score[i])
            #     hop_att_count[h] += (h == h_pred)
            #     hop_pred_count[h_pred] += 1


            if verbose:
                answers = batch[2]
                for i in range(len(match_score)):
                    # if match_score[i] == 0:
                    if True:
                        print('================================================================')
                        question_ids = batch[1]['input_ids'][i].tolist()
                        question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        print(' '.join(question_tokens))
                        topic_id = batch[0][i].argmax(0).item()
                        print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        for t in range(outputs['hop_attn'].shape[1]):
                            print('>>>>>>> step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x, y in
                                zip(question_tokens, outputs['word_attns'][t][i].squeeze().tolist())])
                            print('> Attention: ' + tmp)
                            # print('> Relation:')
                            # rel_idx = outputs['rel_probs'][t][i].gt(0.9).nonzero().squeeze(1).tolist()
                            # for x in rel_idx:
                            #     print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()))

                            print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        # print('----')
                        # print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        # print('> maxes: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].topk(5).indices.tolist()])))
                        # print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        # print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print(' '.join(question_tokens))

                        ans = set(answers[i].gt(0.9).nonzero().squeeze(1).tolist())
                        pred5 = set(e_score[i].topk(5).indices.tolist())
                        pred10 = set(e_score[i].topk(10).indices.tolist())
                        if len(pred5.intersection(ans)) != 0:
                            correct5 += 1
                        if len(pred10.intersection(ans)) != 0:
                            correct10 += 1
                        print(outputs['hop_attn'][i].tolist())
                        # embed()
    acc = correct / count
    acc5 = correct5 /count
    acc10 = correct10 / count
    print(acc, acc5, acc10)
    # acc_hop = ('real hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
    #     sum(hop_count[0]) / (len(hop_count[0]) + 0.1),
    #     len(hop_count[0]),
    #     sum(hop_count[1]) / (len(hop_count[1]) + 0.1),
    #     len(hop_count[1])
    # ))
    # acc_hop_att = ('real hop-att accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
    #     hop_att_count[0] / (len(hop_count[0]) + 0.1),
    #     hop_pred_count[0],
    #     hop_att_count[1] / (len(hop_count[1]) + 0.1),
    #     hop_pred_count[1]
    # ))

    precision = TP_total / (num_answers_pred_total + 0.1)
    recall = TP_total / (num_answers_total + 0.1)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    f1_info = ("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))

    return acc,  f1_info


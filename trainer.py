import os.path
import time
import torch
from copy import deepcopy
import torch.nn as nn

def SeqTrain(epochs, model, train_loader, val_loader, writer, args):
    if args.is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(args.device)
    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model,  find_unused_parameters=True,device_ids=[args.local_rank], output_device=args.local_rank)
    best_metric = 0
    all_time = 0
    val_all_time = 0
    for epoch in range(epochs):
        since = time.time()
        optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)
        tmp = time.time() - since
        print('one epoch train:', tmp)
        all_time += tmp
        val_since = time.time()
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)
        val_tmp = time.time() - val_since
        print('one epoch val:', val_tmp)
        val_all_time += val_tmp
        if args.is_pretrain == 0 and 'acc' in args.task_name:
            if metrics['NDCG@20'] >= 0.0193:
                break
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.save_path, f'{args.model_name}_seed{args.seed}_is_pretrain_{args.is_pretrain}_best_model_lr{args.lr}_wd{args.weight_decay}_block{args.block_num}_hd{args.hidden_size}_emb{args.embedding_size}.pth'))
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model



def SequenceTrainer(epoch, model, dataloader, optimizer, writer, args): #schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        logits = model(seqs) # B x T x V
        logits = logits.view(-1, logits.size(-1)) # (B*T) x V
        labels = labels.view(-1)  # B*T

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer


def Sequence_full_Validate(epoch, model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)
            scores = scores.mean(1)
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics


def recalls_and_ndcgs_for_ks(scores, labels, ks, args):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float()).to(args.device)
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(args.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg

    return metrics
import torch
from torch import nn, optim
from .data_loader import HFMRELoader
from .utils import AverageMeter
from tqdm import tqdm
import pdb

class HFMRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 batch_size=16,
                 num_nodes = 6,
                 max_epoch=5,
                 lr=2e-5,
                 weight_decay=1e-5,
                 opt='adamw',
                 warmup_step=0,
                 devices=[0]):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data path, rel2id, tokenizer, batch_size, shuffle, num_workers=1
        if train_path != None:
            self.train_loader = HFMRELoader(
                num_nodes = num_nodes,
                path = train_path,
                rel2id = model.rel2id,
                tokenizer = model.sentences_encoder.tokenize,
                batch_size = batch_size,
                shuffle = True)

        if val_path != None:
            self.val_loader = HFMRELoader(
                num_nodes=num_nodes,
                path = val_path,
                rel2id = model.rel2id,
                tokenizer = model.sentences_encoder.tokenize,
                batch_size = batch_size,
                shuffle = False)

        if test_path != None:
            self.test_loader = HFMRELoader(
                num_nodes=num_nodes,
                path = test_path,
                rel2id = model.rel2id,
                tokenizer = model.sentences_encoder.tokenize,
                batch_size = batch_size,
                shuffle = False)
        # Model
        self.device=torch.device('cuda:{}'.format(devices[0]))
        self.model = nn.DataParallel(model, device_ids=devices)
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss(reduction='sum')
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")

        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        self.ckpt = ckpt

    def train_model(self, metric='auc'):
        best_metric = 0
        bag_logits_dict = {}
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                self.optimizer.zero_grad()
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass
                label = data[0]
                bag_name = data[1]
                nums=data[2]
                token, mask = data[3].squeeze(1), data[4].squeeze(1)
                rel_scores = self.model(token, mask,nums)
                label = torch.stack(label).to(self.device)
                loss = self.criterion(rel_scores, label)
                pred = (rel_scores >= 0.5)*torch.tensor([[1]*rel_scores.shape[1]]*rel_scores.shape[0]).to(self.device)
                acc = float((pred.view(-1) == label.view(-1)).long().sum().item())/label.view(-1).shape[0]
                pos_total = (label.view(-1) != 0).long().sum().item()
                pos_correct = ((pred.view(-1) == label.view(-1)).long()*(label.view(-1) != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step() 

            # Val
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("AUC: %.4f" % result['auc'])
            print("Previous best auc on val set: %f" % (best_metric))
            if result['auc'] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.model.eval()
        bag_logits_dict = {}
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass
                label = data[0]
                bag_name = data[1]
                nums = data[2]
                token, mask = data[3].squeeze(1), data[4].squeeze(1)
                logits = self.model(token, mask, nums,  False)


                
                for i in range(logits.shape[0]):
                    for relid in range(self.model.module.num_class):
                    	if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({'entpair': bag_name[i][:2], 'relation': self.model.module.id2rel[relid],'score': logits[i][relid].item()})
                    	#pred_result.append({'entpair': bag_name[i][:2], 'relation': self.model.module.id2rel[relid],
                       #                       'score': logits[i][relid].item()})
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)

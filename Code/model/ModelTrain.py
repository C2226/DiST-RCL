import os
import time
import copy
import numpy as np
import torch
from torch import nn
import logging
from model import MainModel

class ModelTrain(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, lr=1e-3, epoches=50, patience=5, result_dir='./',
                 hash_id=None, **kwargs):
        super(BaseModel, self).__init__()

        self.epoches = epoches
        self.lr = lr
        self.patience = patience
        self.device = device

        self.model_save_dir = os.path.join(result_dir, hash_id)
        self.model = MainModel(event_num, metric_num, node_num, device, **kwargs)
        self.model.to(device)


    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()

        # HR@k
        hrs = {1: 0, 3: 0, 5: 0}
        mrr_sum = 0.0

        TP, FP, FN, TN = 0, 0, 0, 0
        valid_cnt = 0
        batch_cnt, epoch_loss = 0, 0.0

        with torch.no_grad():
            for graph, ground_truths in test_loader:
                res = self.model.forward(graph.to(self.device), ground_truths)

                pred_probs = res["pred_prob"]  # [B, N]
                y_preds = res["y_pred"]

                for idx, ranked_nodes in enumerate(y_preds):
                    gt = ground_truths[idx].item()

                    if gt == -1:
                        if ranked_nodes[0] == -1:
                            TN += 1
                        else:
                            FP += 1
                        continue
                    else:
                        valid_cnt += 1
                        if ranked_nodes[0] == -1:
                            FN += 1
                            continue
                        else:
                            TP += 1

                    rank_list = list(ranked_nodes)
                    if gt in rank_list:
                        rank = rank_list.index(gt) + 1

                        for k in hrs:
                            if rank <= k:
                                hrs[k] += 1

                        mrr_sum += 1.0 / rank

                epoch_loss += res["loss"].item()
                batch_cnt += 1

        pos = TP + FN
        eval_results = {
            "F1": TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0,
            "Rec": TP * 1.0 / pos if pos > 0 else 0,
            "Pre": TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0,
        }

        for k in hrs:
            eval_results[f"HR@{k}"] = hrs[k] / pos if pos > 0 else 0

        eval_results["MRR"] = mrr_sum / valid_cnt if valid_cnt > 0 else 0

        logging.info(
            "{} -- {}".format(
                datatype,
                ", ".join([k + ": " + f"{v:.4f}" for k, v in eval_results.items()])
            )
        )

        return eval_results


    def fit(self, train_loader, test_loader=None, evaluation_epoch=10):
        best_hr1, coverage, best_state, eval_res = -1, None, None, None
        pre_loss, worse_count = float("inf"), 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epoches + 1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for graph, label in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(graph.to(self.device), label)['loss']
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start

            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss,
                                                                               epoch_time_elapsed))

            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else:
                worse_count = 0
            pre_loss = epoch_loss

            if (epoch + 1) % evaluation_epoch == 0:
                test_results = self.evaluate(test_loader, datatype="Test")
                if test_results["HR@1"] > best_hr1:
                    best_hr1, eval_res, coverage = test_results["HR@1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict())

                self.save_model(best_state)

        if coverage > 5:
            logging.info("* Best result got at epoch {} with HR@1: {:.4f}".format(coverage, best_hr1))
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage

    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)

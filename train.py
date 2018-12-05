import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PPIDataset
from models import StackedAutoencoder, Classifier
from tqdm import tqdm

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
# Train params
parser.add_argument("--data-type", type=str, default="yeast", help="human/yeast")
parser.add_argument("--data-dir", type=str, default="data/yeast_ac")
parser.add_argument("--feature-type", type=str, default="AC")
parser.add_argument("--model", type=str, default="SAE", help="SAE/Classifier")
parser.add_argument("--repetition-number", type=int, default=0, help="0-9")
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--debug", type=str_to_bool, default=True)
parser.add_argument("--model-save-dir", type=str, default="model_checkpoints")
parser.add_argument("--load-SAE", type=str, default="", help="path to saved SAE model")
parser.add_argument("--load-classifier", type=str, default="", help="path to saved classifier model")
parser.add_argument("--num-folds-to-use", type=int, default=1, help="num folds to train with (max 10)")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--log-frequency", type=int, default=25)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--prediction-threshold", type=float, default=-1)
# Model params
parser.add_argument("--input-size", type=int, default=420)
parser.add_argument("--hidden-layer-size", type=int, default=400)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--momentum", type=float, default=0.5)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not args.debug:
    # set up model checkpoint dir
    suffix = "_" + time.strftime("%y%m%d_%H%M%S")
    if not os.path.isdir(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    checkpoint_dir = os.path.join(args.model_save_dir, args.model + suffix)
    os.makedirs(checkpoint_dir)
    print("creating checkpoint dir {}".format(checkpoint_dir))
    with open(os.path.join(checkpoint_dir, "params.txt"), "w") as f:
        f.write(str(args))

# TP/FP/TN/FN
def compute_metrics(predictions, labels):
    # compute TP/FP/TN/FN rates
    n = labels.shape[0]
    num_pos = (labels == 1).float().sum().item()
    num_neg = n - num_pos
    tp = ((labels == 1) & ((labels - predictions) == 0)).float().sum().item()
    fp = ((predictions - labels) == 1).float().sum().item()
    tn = ((labels == 0) & ((labels - predictions) == 0)).float().sum().item()
    fn = ((predictions - labels) == -1).float().sum().item()
    if tp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    return precision, tp/(tp+fn), fp/(fp+tn), 1 - fp/(fp+tn), 1 - tp/(tp+fn), (tp + tn)/(num_pos + num_neg)

if args.prediction_threshold != -1:
    pred_thresholds = [args.prediction_threshold]
else:
    pred_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
reps = 1

for pt in pred_thresholds:
    for rep in range(reps):
        # metrics averaged over k-folds
        cv_ppv = []
        cv_tpr = []
        cv_fpr = []
        cv_tnr = []
        cv_fnr = []
        cv_acc = []
        for nf in range(args.num_folds_to_use):
            # Create datasets
            train_dataset = PPIDataset(True, args.data_type, rep, 
                    nf, args.data_dir, args.feature_type)
            test_dataset = PPIDataset(False, args.data_type, rep,
                    nf, args.data_dir, args.feature_type)

            # Create dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                    shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=False)
            test_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                    shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)

            # models
            if args.model.lower() == "sae":
                model = StackedAutoencoder(args.input_size, args.hidden_layer_size)
                if args.load_SAE != "":
                    print("loading SAE from {}...".format(args.load_SAE))
                    model.load_state_dict(torch.load(args.load_SAE))
                loss_fn = nn.MSELoss()
            elif args.model.lower() == "classifier":
                sae = StackedAutoencoder(args.input_size, args.hidden_layer_size)
                if args.load_SAE != "":
                    print("loading SAE from {}...".format(args.load_SAE))
                    sae.load_state_dict(torch.load(args.load_SAE))
                model = Classifier(sae, args.hidden_layer_size)
                if args.load_classifier != "":
                    print("loading classifier from {}...".format(args.load_classifier))
                    model.load_state_dict(torch.load(args.load_classifier))
                loss_fn = nn.BCEWithLogitsLoss()

            model = model.to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            #optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

            best_test_loss = np.inf
            best_ppv = 0
            best_tpr = 0
            best_fpr = 1
            best_tnr = 0
            best_fnr = 1
            for e in tqdm(range(args.num_epochs)):
                
                #print("<============ model train =============>")
                
                for batch_idx, batch in tqdm(enumerate(train_dataloader)):
                    label = batch["label"].to(args.device)
                    input = batch["x"].to(args.device)
                    pred = model(input)
                    
                    if args.model.lower() == "sae": 
                        loss = loss_fn(pred, input)
                    elif args.model.lower() == "classifier":
                        loss = loss_fn(pred, label.float())
                    #if batch_idx % args.log_frequency == 0:
                    #    print("TRAIN: epoch {}/{}, batch: {}/{}, loss: {}".format(
                    #        e, args.num_epochs, batch_idx,
                    #        int(np.floor(len(train_dataset)/args.batch_size)),
                    #        loss.item()))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                #print("<============ model test =============>")

                with torch.no_grad():
                    losses = []
                    epoch_ppv = []
                    epoch_tpr = []
                    epoch_fpr = []
                    epoch_tnr = []
                    epoch_fnr = []
                    for batch_idx, batch in tqdm(enumerate(test_dataloader)):
                        label = batch["label"].to(args.device)
                        input = batch["x"].to(args.device)
                        pred = model(input)
                        if args.model.lower() == "sae": 
                            loss = loss_fn(pred, input)
                            losses.append(loss.item())
                        elif args.model.lower() == "classifier":
                            pred = (pred.sigmoid() > pt).long()
                            ppv, tpr, fpr, tnr, fnr, acc = compute_metrics(pred, label)
                            epoch_ppv.append(ppv)
                            epoch_tpr.append(tpr)
                            epoch_fpr.append(fpr)
                            epoch_tnr.append(tnr)
                            epoch_fnr.append(fnr)
                            losses.append(1 - acc) # error rate
                    if args.model.lower() == "sae":
                        print("TEST: fold: {}/{}, epoch {}/{}, MSE loss: {}".format(
                            nf, args.num_folds_to_use, e, args.num_epochs, np.mean(losses)))
                    elif args.model.lower() == "classifier":
                        print("TEST: fold: {}/{}, epoch {}/{}, Accuracy: {}".format(
                            nf, args.num_folds_to_use, e, args.num_epochs, 1 - np.mean(losses)))
                    
                    scheduler.step()

                    if not args.debug and  np.mean(losses) < best_test_loss:
                        best_test_loss = np.mean(losses)
                        #model_file_name = "model-{}-best.pth".format(args.model)
                        #torch.save(model.state_dict(), os.path.join(checkpoint_dir, model_file_name))

                    if np.mean(epoch_ppv) > best_ppv:
                        best_ppv = np.mean(epoch_ppv)
                    if np.mean(epoch_tpr) > best_tpr:
                        best_tpr = np.mean(epoch_tpr)
                    if np.mean(epoch_fpr) < best_fpr:
                        best_fpr = np.mean(epoch_fpr)
                    if np.mean(epoch_tnr) > best_tnr:
                        best_tnr = np.mean(epoch_tnr)
                    if np.mean(epoch_fnr) < best_fnr:
                        best_fnr = np.mean(epoch_fnr)
            cv_ppv.append(best_ppv)
            cv_tpr.append(best_tpr)
            cv_fpr.append(best_fpr)
            cv_tnr.append(best_tnr)
            cv_fnr.append(best_fnr)
            cv_acc.append(1 - best_test_loss)
        if not args.debug:
            # write results to CSV for processing
            for name, arr in zip(["ppv", "acc", "tpr", "fpr", "tnr", "fnr"], [cv_ppv, cv_acc, cv_tpr, cv_fpr, cv_tnr, cv_fnr]):
                with open(os.path.join(checkpoint_dir, "{}-results-threshold-{}-repetition-{}.csv").format(
                    name, pt, rep), "w") as f:
                    
                    # header
                    header = ""
                    for i in range(1,args.num_folds_to_use+1):
                        header += "fold{},".format(i)
                    f.write(header[:-1] + "\n")

                    result = ""
                    for i in range(args.num_folds_to_use):
                        if i < args.num_folds_to_use-1:
                            result += "{},".format(arr[i])
                        else:
                            result += "{}\n".format(arr[i])
                    f.write(result)

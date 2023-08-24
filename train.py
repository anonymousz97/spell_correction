import pandas as pd
from vocab import Vocab
import utils
from sc_model import SC
import torch.nn as nn
import torch
from utils import pad_sents, check_and_reduce_text, norm_text
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from create_data import create
import logging


def length(data):
    return torch.LongTensor([len(seq) for seq in data]).to(device)

#def compare(check, correct):
    #x = check.strip().lower().split()
    #y = correct.strip().lower().split()
    #label = []
    #for i in range(len(x)):
        #if x[i] == y[i]:
            #label.append("0")
        #else:
            #label.append("1")
    #return  " ".join(label)

#def new_check(x):
    #a = x["Check"].split()
    #b = x["Correct"].split()
    #for i in range(len(a)):
        #if a[i] != b[i]:
            #if a[i] not in model.vocab:
                #a[i] = reduce_wrong_word(a[i])
    #return " ".join(a)

def parse_args():
    par = argparse.ArgumentParser()
    par.add_argument("--first_train", type=int, default=1, help="Create new model to train or load it from a path")
    par.add_argument("--cuda", type = int, default = 0)
    par.add_argument("--epochs", type=int, default=500, help="The number of epochs")
    par.add_argument("--path_model", type=str, default="model.bin", help="Path to save or load model")
    par.add_argument("--batch_size", type=int, default=64, help="Batch size")
    par.add_argument("--base_path", type = int, default = 1)
    par.add_argument("--lr", type=float, default=0.00015, help="Learning rate")
    par.add_argument("--vocab_path", type=str, default="vocab.txt", help="Path to file txt including vocab")
    par.add_argument("--first_data_set", type = int, default = 1)
    par.add_argument("--acc_base", type = float, default = 0)
    par.add_argument("--log_number", type = str, default = "1")
    return par.parse_args()



def collate_fn(batch):
    x1,x2, y1,y2 = zip(*batch)
    y1 = pad_sents(y1, "0")
    y2 = pad_sents(y2, "0")
    for i in range(len(y1)):
        y1[i] = list(map(int, y1[i]))
        y2[i] = list(map(int, y2[i]))
    return x1, x2, torch.Tensor(y1).to(device), torch.Tensor(y2).to(device)

def f1_score(correct, predict, total):
    recall = correct/total
    precision = correct/predict
    return 2*recall*precision/(recall + precision)


# def read_data(filename):
#     data = pd.read_csv(filename)
#     return list(data.iloc[:, 0]), list(data.iloc[:, 1]), list(data.iloc[:, 2])


def loss_function(model, x1, x2, y1, y2):
    check, check_upper, lengths= model(x1)

    loss = (loss_check(check, y1) + loss_check(check_upper, y2)) / (2*sum(lengths))

    #label = y.float().reshape(-1).squeeze()

    #source_padded = model.vocab.to_input_tensor(x2, device=model.device).permute(1, 0).reshape(-1).squeeze()

    #correct = correct.reshape(source_padded.shape[0], -1)

    #l_correct = loss_correct(correct, source_padded) @ label / sum(label)

    #         loss = loss +  l_correct
    #loss = loss + l_correct
    return loss

if __name__ == '__main__':
    args = parse_args()
    log_name = "checkpointv" + args.log_number +".log"
    logging.basicConfig(filename=log_name,level=logging.DEBUG, filemode="a")
    cuda = "cuda:"+str(args.cuda) 
    device = cuda if torch.cuda.is_available() else "cpu"
    first_data_set = args.first_data_set
    base_path = "checkpointv" + str(args.base_path) + "/" 
    model_save_path = base_path + args.path_model
    num_layers,d_model,nhead,dropout ,sub_num_layers ,sub_d_model ,sub_nhead ,sub_dropout = 6, 768, 8, 0.1, 3, 256, 8, 0.1
    if args.first_train == 1:
        v = Vocab.from_corpus(args.vocab_path, 1000000, 1)
        logging.info("Create new model ! Num_layers = {}, d_model = {}, nhead = {}, dropout = {:.2f}, sub_num_layers = {}, sub_d_model = {}, sub_nhead = {}, sub_dropout = {:.2f}".format(num_layers, d_model, nhead, float(dropout), sub_num_layers, sub_d_model, sub_nhead, float(sub_dropout)))
        model = SC(v,
                   num_layers = int(num_layers),
                   d_model = int(d_model),
                   nhead = int(nhead),
                   dropout = float(dropout),
                   sub_num_layers = int(sub_num_layers),
                   sub_d_model=int(sub_d_model),
                   sub_nhead= int(sub_nhead),
                   sub_dropout= float(sub_dropout),
                   max_length = 512).to(device)
    else:
        model = SC.load(model_save_path).to(device)
    data_path = "data_eval.csv"
    logging.info("Load data eval " +  data_path)
    data = pd.read_csv(data_path)
    logging.info("Eval data length: " +  str(len(data)))
    eval_check = list(map(lambda x: x.strip().split(), list(data.iloc[:, 0])))
    eval_correct = list(map(lambda x: x.strip().split(), list(data.iloc[:, 1])))
    eval_label = list(map(lambda x: x.strip().split(), list(data.iloc[:, 2])))
    eval_label_cap = list(map(lambda x: x.strip().split(), list(data.iloc[:, 3])))

    #eval_label = pad_sents(eval_label, "0")
    #for i in range(len(eval_label)):
        #eval_label[i] = list(map(int, eval_label[i]))
    #eval_label = torch.Tensor(eval_label).cuda()
    # param
    #print("Acc_u: ", acc_u)
    logging.info("Batch size " +  str(args.batch_size))
    batch_size = args.batch_size
    logging.info("Lr:" +  str(args.lr))
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_check = nn.BCELoss(reduction="sum")
    loss_correct = nn.CrossEntropyLoss(reduction="none")
    lr_decay = 0.4
    logging.info("Epochs:" + str(args.epochs))
    epochs = args.epochs
    patience = 0
    k = 0

    # train
    data_eval = list(zip(eval_check, eval_correct, eval_label, eval_label_cap))
    eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    del(data_eval)
    del(eval_check)
    del(eval_correct)
    del(eval_label)
    del(eval_label_cap)
    min_loss = 0
    pa = 0
    if args.acc_base ==0 and args.first_train != 1:
        model.eval()
        with torch.no_grad():
            correct = 0
            error = 0
            predict_labels = 0
            correct_u = 0
            error_u = 0
            predict_labels_u = 0

            for eval_check, eval_correct, eval_label, eval_label_upper in eval_loader:
                c, e, p, c_u, e_u, p_u = model.evaluate_accuracy(eval_check, eval_correct, eval_label, eval_label_upper)
                correct += c 
                error += e 
                predict_labels +=p
                correct_u += c_u 
                error_u += e_u
                predict_labels_u +=p_u

            score = f1_score(correct,predict_labels, error)
            score_u = f1_score(correct_u,predict_labels_u, error_u)
        best_score = (score + score_u)/2
    else:
        best_score = args.acc_base
    logging.info("Base F1_score: {:.4f} %".format(best_score))
    for epoch in range(epochs):
        if epoch < first_data_set - 1:
            continue
        # print("Epoch: ", (epoch+1))
        data_path = "data_train.txt"
        logging.info("Load data train" + data_path)
        data = create("data_train.txt")
        # read_data
        data_check = list(map(lambda x: x.strip().split(), list(data.iloc[:, 0])))
        # for line in list(data.iloc[:, 0]):
        #     data_check.append(line.strip().split()):
        data_correct = list(map(lambda x: x.strip().split(), list(data.iloc[:, 1])))
        # for line in list(data.iloc[:, 1]):
        #     data_correct.append(line.strip().split())
        data_label = list(map(lambda x: x.strip().split(), list(data.iloc[:, 2])))
        data_label_upper = list(map(lambda x: x.strip().split(), list(data.iloc[:, 3])))
        # for line in list(data.iloc[:, 2]):
        #     data_label.append(line.strip().split())
        del(data)
        data = list(zip(data_check, data_correct, data_label, data_label_upper))
        del(data_label_upper)
        del(data_check)
        del(data_correct)
        del(data_label)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        del(data)
        model.train()
        total_loss = 0
        for x1, x2, y1,y2 in tqdm(loader):
            optimizer.zero_grad()

            loss = loss_function(model, x1, x2, y1,y2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        logging.info("Training Loss = {:.4f} with lr = {}".format(total_loss, optimizer.param_groups[0]['lr']))
        # Early stop and decay lr
        if epoch % 1 == 0:
            # print("Loss: {:.2f}, epoch: {}".format(total_loss, epoch))
            model.eval()
            logging.info("Eval...")
            with torch.no_grad():
                correct = 0
                error = 0
                predict_labels = 0
                correct_u = 0
                error_u = 0
                predict_labels_u = 0

                for eval_check, eval_correct, eval_label, eval_label_cap in eval_loader:
                    c, e, p, c_u, e_u, p_u = model.evaluate_accuracy(eval_check, eval_correct, eval_label, eval_label_cap)
                    correct +=c 
                    error +=e 
                    predict_labels +=p
                    correct_u +=c_u
                    error_u +=e_u
                    predict_labels_u +=p_u
                score =  (f1_score(correct,predict_labels, error) + f1_score(correct_u,predict_labels_u, error_u))/2
            if score > best_score:
                patience = 0
                best_score = score
                logging.info("Save model F1_score = {:.4f} %".format(best_score))
                model_save_path = base_path + "model" + "{:.4f}".format(best_score) + ".bin"
                model.save(model_save_path)

            else:
                patience += 1
            if patience == 3:
                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * lr_decay
                logging.info('Load previously best model and decay learning rate to {}'.format(lr))
                if lr <= 0.000001:
                    break
                model = SC.load(model_save_path).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # reset patience
                patience = 0


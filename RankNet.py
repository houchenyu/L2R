import torch
from torch.nn import functional as F
import argparse
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
    return v


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores)/idcg(scores)


def delta_ndcg(scores, p, q):
    """
    swap the i-th and j-th doucment, compute the absolute value of NDCG delta
    :param scores: a score list of documents
    :param p, q: the swap positions of documents
    :return: the absolute value of NDCG delta
    """
    s2 = scores.copy()  # new score list
    s2[p], s2[q] = s2[q], s2[p]  # swap
    return abs(ndcg(s2) - ndcg(scores))


def ndcg_k(scores, k):
    scores_k = scores[:k]
    fenzi = dcg(scores_k)
    fenmu = dcg(sorted(scores)[::-1][:k])
    return fenzi/fenmu


def group_by(data, qid_index):
    """
    group documents by query-id
    :param data: input_data which contains multiple query and corresponding documents
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def get_pairs(scores):
    """
    compute the ordered pairs whose firth doc has a higher value than second one.
    :param scores: given score list of documents for a particular query
    :return: ordered pairs.  List of tuple, like [(1,2), (2,3), (1,3)]
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2 = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc


class Model(torch.nn.Module):
    """
    construct the RankNet
    """
    def __init__(self, n_feature, h1_units, h2_units):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            # h_1
            torch.nn.Linear(n_feature, h1_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # h_2
            torch.nn.Linear(h1_units, h2_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # output
            torch.nn.Linear(h2_units, 1),
        )
        self.output_sig = torch.nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.output_sig(s1-s2)
        return out

    def predict(self, input_):
        s = self.model(input_)
        n = s.data.numpy()[0]
        return n


class RankNet():
    """
    user interface
    """
    def __init__(self, n_feature, h1_units, h2_units, epoch, learning_rate, plot=True):
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = Model(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.plot = plot
        self.learning_rate = learning_rate

    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

    def fit(self, training_data):
        """
        train the RankNet based on training data.
        After training, save the parameters of RankNet, named 'parameters.pkl'
        :param training_data:
        """

        net = self.model
        qid_doc_map = group_by(training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        relevant_doc, irrelevant_doc = split_pairs(order_paris ,true_scores)
        relevant_doc = training_data[relevant_doc]
        irrelevant_doc = training_data[irrelevant_doc]

        X1 = relevant_doc[:, 2:]
        X2 = irrelevant_doc[:, 2:]
        y = np.ones((X1.shape[0], 1))

        # training......
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        y = torch.Tensor(y)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        loss_fun = torch.nn.BCELoss()

        loss_list = []

        if self.plot:
            plt.ion()

        print('Traning………………\n')
        for i in range(self.epoch):
            self.decay_learning_rate(optimizer, i, 0.95)

            net.zero_grad()
            y_pred = net(X1, X2)
            loss = loss_fun(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.numpy())
            if self.plot:
                    plt.cla()
                    plt.plot(range(i+1), loss_list, 'r-', lw=5)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.pause(1)
            if i % 10 == 0:
                print('Epoch:{}, loss : {}'.format(i, loss.item()))


        if self.plot:
            plt.ioff()
            plt.show()

        # save model parameters
        torch.save(net.state_dict(), 'parameters.pkl')

    def validate(self, test_data, k):
        """
        compute the average NDCG@k for the given test data.
        :param test_data: test data
        :param k: used to compute NDCG@k
        :return:
        """
        # load model parameters
        net = Model(self.n_feature, self.h1_units, self.h2_units)
        net.load_state_dict(torch.load('parameters.pkl'))

        qid_doc_map = group_by(test_data, 1)
        query_idx = qid_doc_map.keys()
        ndcg_k_list = []

        for q in query_idx:
            true_scores = test_data[qid_doc_map[q], 0]
            if sum(true_scores) == 0:
                continue
            docs = test_data[qid_doc_map[q]]
            X_test = docs[:, 2:]

            pred_scores = [net.predict(torch.Tensor(test_x).data) for test_x in X_test]
            pred_rank = np.argsort(pred_scores)[::-1]
            pred_rank_score = true_scores[pred_rank]
            ndcg_val = ndcg_k(pred_rank_score, k)
            ndcg_k_list.append(ndcg_val)
        print("Average NDCG@{} is {}".format(k, np.mean(ndcg_k_list)))

if __name__ == '__main__':
    print('Load training data...')
    training_data = np.load('./dataset/train.npy')
    print('Load done.\n\n')

    model1 = RankNet(46, 512, 256, 100, 0.01, True)
    model1.fit(training_data)

    print('Validate...')
    test_data = np.load('./dataset/test.npy')
    model1.validate(test_data)



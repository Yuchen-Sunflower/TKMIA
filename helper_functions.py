import math
import numpy as np
from torchvision import datasets as datasets
import torch
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import label_ranking_loss

def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def average(self):
        return self.avg
    
    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)


    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.clone().cpu().numpy()
        n, c = self.scores.size()
        scores = np.zeros((n, c))
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0.5 else 0
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            Ng[k] = np.sum(targets == 1)  # TP + FN 所有真实标签为1的数量
            Np[k] = np.sum(scores >= 0.5)  # TP + FP 所有预测为1的数量
            Nc[k] = np.sum(targets * (scores >= 0.5))  # TP 所有真实标签为1 且 预测为1的数量
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)  # TP / (TP + FP)
        OR = np.sum(Nc) / np.sum(Ng)  # TP / (TP + FN)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class  # TP[k] / (TP[k] + FP[k]) / num_class
        CR = np.sum(Nc / Ng) / n_class  # TP[k] / (TP[k] + FN[k]) / num_class 
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

    def ranking_loss(self):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        return label_ranking_loss(targets, scores)

    def precision_at_k(self, k):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        top_k_idx = np.argpartition(scores, kth=-k)[:, -k:]
        top_k_targets = targets[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx]
        top_k_scores = np.ones((targets.shape[0], k))

        return precision_score(top_k_targets, top_k_scores, average='micro')

    def recall_at_k(self, k):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        top_k_idx = np.argpartition(scores, kth=-k)

        top_k_scores = np.zeros((scores.shape[0], scores.shape[1]))
        top_k_scores[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx[:, -k:]] = 1

        return recall_score(targets, top_k_scores, average='micro')

    def precision_at_k_instance(self, targets, scores, top_k_scores, k):

        top_k_idx = np.argpartition(scores, kth=-k)[-k:]
        top_k_targets = targets[top_k_idx]
        # top_k_scores = np.ones(k)
        return precision_score(top_k_targets, top_k_scores[:k], average='micro')

    def mAP_at_K(self, K):
        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        top_k_idx = np.argsort(scores)[:, ::-1][:, :K]
        top_k_scores = np.ones(K)
        top_k_targets = targets[np.arange(targets.shape[0])[:, np.newaxis], top_k_idx]
        p_at_k = np.array([[self.precision_at_k_instance(t, p, top_k_scores, k) for k in range(1, K + 1)] for t, p in zip(targets, scores)])

        n_k = np.sum(targets, axis=1)
        tmp_k = np.ones(targets.shape[0]) * K
        tag = n_k <= tmp_k
        n_k[~tag] = K

        return np.average(np.sum(p_at_k * top_k_targets, axis=1) / n_k)

    def NDCG_at_K(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        return ndcg_score(y_true=targets, y_score=scores, k=K)

    def _tie_averaged_dcg(self, y_true, y_score, discount_cumsum):
        _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
        ranked = np.zeros(len(counts))
        np.add.at(ranked, inv, y_true)
        ranked /= counts
        groups = np.cumsum(counts) - 1
        discount_sums = np.empty(len(counts))
        discount_sums[0] = discount_cumsum[groups[0]]
        discount_sums[1:] = np.diff(discount_cumsum[groups])
        return (ranked * discount_sums).sum()

    def DCG_l_at_K(self, targets, scores, K, ignore_ties=False):
        # discount1 = 1 / (np.log(np.arange(targets.shape[1]) + 2) / np.log(2))
        # discount2 = 1 / (np.log(np.arange(targets.shape[1]) + 2) / np.log(2))
        discount = K - np.arange(targets.shape[1])

        if K is not None:
            discount[K:] = 0
        
        if ignore_ties:
            ranking = np.argsort(scores)[:, ::-1]
            ranked = targets[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
            cumulative_gains = discount.dot(ranked.T)

            # cumulative_gains = n_k * (n_k + 1) / 2
            # print(cumulative_gains.shape)
        else:
            discount_cumsum = np.cumsum(discount)
            cumulative_gains = [
                self._tie_averaged_dcg(y_t, y_s, discount_cumsum)
                for y_t, y_s in zip(targets, scores)
            ]
            cumulative_gains = np.asarray(cumulative_gains)

        return cumulative_gains

    def NDCG_l_at_K(self, K):
        targets = self.targets.clone().cpu().numpy()  # 0-1 array
        scores = self.scores.clone().cpu().numpy()
        gain = self.DCG_l_at_K(targets, scores, K, ignore_ties=False)
        normalizing_gain = self.DCG_l_at_K(targets, targets, K, ignore_ties=True)
        all_irrelevant = normalizing_gain == 0
        gain[all_irrelevant] = 0
        gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]

        return np.average(gain)

    def DCG_ln_at_K(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()

        n_k = np.sum(targets, axis=1)
        tmp_k = np.ones(targets.shape[0]) * K
        tag = n_k <= tmp_k
        n_k[~tag] = K
        gain = self.DCG_l_at_K(targets, scores, K) / n_k

        return np.average(gain)

    def AUTKC_M(self, K):

        targets = self.targets.clone().cpu().numpy()
        scores = self.scores.clone().cpu().numpy()
        gain = self.DCG_l_at_K(targets, scores, K) / K

        return np.average(gain)

    def AUTKC_L(self, K):

        return self.DCG_ln_at_K(K) / K

    def AUTKC_Q(self, K):

        return self.NDCG_l_at_K(K) / K


if __name__ == "__main__":
    # scores = np.array([[0.8, 0.7, 0.1, 0.75], [0.2, 0.5, 0.6, 0.9]])
    # target = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])
    
    scores = torch.tensor([[0.98, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]])
    targets = torch.tensor([[1, 1, 1, 1, 0, 1, 0, 0, 0, 0]])
    a = AveragePrecisionMeter()
    a.scores = scores
    a.targets = targets
    print(a.precision_at_k(5))
    print(a.mAP_at_K(5))
    print(a.NDCG_at_K(5))

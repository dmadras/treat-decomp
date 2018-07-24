import numpy as np
from sklearn.metrics import roc_auc_score

eps = 1e-8

def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)

def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)

def PR(Y):
    return pos(Y) / (pos(Y) + neg(Y))

def NR(Y):
    return neg(Y) / (pos(Y) + neg(Y))

def TP(Y, Ypred):
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)

def FP(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)

def TN(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)

def FN(Y, Ypred):
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)

def FP_soft(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), Ypred)).astype(np.float32)

def FN_soft(Y, Ypred):
    return np.sum(np.multiply(Y, 1 - Ypred)).astype(np.float32)

#note: TPR + FNR = 1; TNR + FPR = 1
def TPR(Y, Ypred):
    return TP(Y, Ypred) / pos(Y)

def FPR(Y, Ypred):
    return FP(Y, Ypred) / neg(Y)

def TNR(Y, Ypred):
    return TN(Y, Ypred) / neg(Y)

def FNR(Y, Ypred):
    return FN(Y, Ypred) / pos(Y)

def FPR_soft(Y, Ypred):
    return FP_soft(Y, Ypred) / neg(Y)

def FNR_soft(Y, Ypred):
    return FN_soft(Y, Ypred) / pos(Y)

def calibPosRate(Y, Ypred):
    return TP(Y, Ypred) / pos(Ypred)

def calibNegRate(Y, Ypred):
    return TN(Y, Ypred) / neg(Ypred)

def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])

def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)

def subgroup(fn, mask, args):# Y, Ypred=None, A=None):
    m = np.greater(mask, 0.5).flatten()
    flat_args = [x.flatten() if hasattr(x, 'flatten') else x for x in args]
    # print(m, flat_args)
    # print(m.shape, [x.shape for x in flat_args])
    return fn(*[x[m] for x in flat_args])
    # if Ypred is None and A is None:
    #     return fn(Yf[m])
    # elif not Ypred is None and A is None: #two-argument functions
    #     Ypredf = Ypred.flatten()
    #     return fn(Yf[m], Ypredf[m])
    # else: #three-argument functions
    #     Ypredf = Ypred.flatten()
    #     Af = A.flatten()
    #     return fn(Yf[m], Ypredf[m], Af[m])

def DI_FP(Y, Ypred, A):
    fpr1 = subgroup(FPR, A, [Y, Ypred])
    fpr0 = subgroup(FPR, 1 - A, [Y, Ypred])
    return abs(fpr1 - fpr0)

def DI_FN(Y, Ypred, A):
    fnr1 = subgroup(FNR, A, [Y, Ypred])
    fnr0 = subgroup(FNR, 1 - A, [Y, Ypred])
    return abs(fnr1 - fnr0)


def DI_FP_soft(Y, Ypred, A):
    fpr1 = subgroup(FPR_soft, A, [Y, Ypred])
    fpr0 = subgroup(FPR_soft, 1 - A, [Y, Ypred])
    return abs(fpr1 - fpr0)

def DI_FN_soft(Y, Ypred, A):
    fnr1 = subgroup(FNR_soft, A, [Y, Ypred])
    fnr0 = subgroup(FNR_soft, 1 - A, [Y, Ypred])
    return abs(fnr1 - fnr0)

def DI(Y, Ypred, A):
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5

def DI_soft(Y, Ypred, A):
    return (DI_FN_soft(Y, Ypred, A) + DI_FP_soft(Y, Ypred, A)) * 0.5

def DP(Ypred, A): #demographic disparity
    return abs(subgroup(PR, [A, Ypred]) - subgroup(PR, [1 - A, Ypred]))


def switch(x0, x1, s):
    return np.multiply(x0, 1. - s) + np.multiply(x1, s)

def ce(y, yp):
    return - (np.multiply(y, np.log(yp + eps)) + np.multiply(1 - y, np.log(1 - yp + eps)))

def avg_ce(y, yp):
    return np.mean(ce(y, yp))

def ece(yp):
    return ce(yp, yp)

def avg_ece(yp):
    return np.mean(ece(yp))

def calc_opt_idk_loss(y, yp, idk_num):
    ec = ece(yp)
    should_idk = np.greater(ec, idk_num)
    c = ce(y, yp)
    l = np.multiply(c, 1. - should_idk) + should_idk * idk_num
    return np.mean(l)

def roughCal(y, yp):
    diff = y - yp
    return np.mean(diff)

#### Causal inference

def ITE(y, cf_y, t):
    assert y.shape == cf_y.shape == t.shape
    treated_outcomes = switch(cf_y, y, t)
    untreated_outcomes = switch(cf_y, y, 1 - t)
    return treated_outcomes - untreated_outcomes

def ATE(y, cf_y, t):
    return np.mean(ITE(y, cf_y, t))

def PEHE(y, cf_y, t, y_pred, cf_y_pred):
    true_te = ITE(y, cf_y, t)
    est_te = ITE(y_pred, cf_y_pred, t)
    return np.mean(np.square(true_te - est_te))

def absErrITE(y, cf_y, t, y_pred, cf_y_pred):
    true_te = ITE(y, cf_y, t)
    est_te = ITE(y_pred, cf_y_pred, t)
    return np.mean(np.abs(true_te - est_te))

def absErrATE(y, cf_y, t, y_pred, cf_y_pred):
    true_ate = ATE(y, cf_y, t)
    est_ate = ATE(y_pred, cf_y_pred, t)
    return np.abs(true_ate - est_ate)

def AUC(y, ypred):
    try:
        return roc_auc_score(y, ypred)
    except:
        return 1. #all one class in y_true - can trivially get 100% accuracy


if __name__ == '__main__':
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Ypred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.9])
    A = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

    assert pos(Y) == 5
    assert neg(Y) == 5
    assert pos(Ypred) == 5
    assert neg(Ypred) == 5

    assert TP(Y, Ypred) == 3
    assert FP(Y, Ypred) == 2
    assert TN(Y, Ypred) == 3
    assert FN(Y, Ypred) == 2
    assert np.isclose(TPR(Y, Ypred), 0.6)
    assert np.isclose(FPR(Y, Ypred) , 0.4)
    assert np.isclose(TNR(Y, Ypred) , 0.6)
    assert np.isclose(FNR(Y, Ypred) , 0.4)
    assert np.isclose(calibPosRate(Y, Ypred) , 0.6)
    assert np.isclose(calibNegRate(Y, Ypred) , 0.6)
    assert np.isclose(errRate(Y, Ypred) , 0.4)
    assert np.isclose(accuracy(Y, Ypred) , 0.6)
    assert np.isclose(subgroup(TNR, A, [Y, Ypred]) , 0.5)
    assert np.isclose(subgroup(pos, 1 - A, [Ypred]) , 2)
    assert np.isclose(subgroup(neg, 1 - A, [Y]) , 3)
    assert np.isclose(DI_FP(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FP(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(subgroup(accuracy, A, [Y, Ypred]), 0.6)
    assert np.isclose(subgroup(errRate, 1 - A, [Y, Ypred]), 0.4)



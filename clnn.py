from data import get_binarized_mnist

import jax.numpy as jnp
from jax import random

rand_seed = 299792458
key = random.PRNGKey(rand_seed)

def n_wta(x, n=1):
    # Winner Take All Inhibition
    assert n == 1, "Only n=1 is supported"

    # x is a 1xq row vector of integers
    x_max = jnp.max(x)
    x_maxes = x == x_max
    first_max_idx = jnp.argmax(x_maxes)
    x_first_max_only = jnp.array([1 if i == first_max_idx else 0 for i in range(x.shape[1]) ])

    # print(x)
    # print(x_max)
    # print(x_maxes)
    # print(first_max_idx)
    # print(x_first_max_only)

    return x * x_first_max_only

def segment(x, theta):
    # x is a 1xp row of bits
    # theta is a 1x1 int
    # seg_w is a px1 column vector of ints
    p = x.shape[1]
    w = random.randint(key, (p, 1), 0, 10)
    tmp1 = x @ w
    tmp2 = tmp1 >= theta
    seg_out = tmp1 * tmp2
    # print(f"{seg_w=}, {tmp1=}, {tmp2=}, {seg_out=}")

    return seg_out

def dendrite_if(x, y, theta: int, q):
    # x is 1xp bit vector
    # y is a 1x1 bit
    # theta is a 1x1 int
    p = x.shape[1]
    x_y_min = jnp.minimum(x, y)
    w = random.randint(key, (p, q), 0, 10)
    pre_threshold = x_y_min @ w
    threshold_mask = pre_threshold >= theta
    thresholded = pre_threshold * threshold_mask
    return n_wta(thresholded)

def sdp_capture(weights, x, z, capture: int, w_max=8):
    # weights is pxq matrix of ints
    # x is 1xp bit vector
    # z is 1xq int vector
    # capture is int
    x = x.reshape(-1, 1)
    z_gz = z > 0
    r = x @ z_gz
    f = r * capture
    weights_new = weights + f
    return jnp.minimum(weights_new, w_max)

def sdp_backoff(weights, x, z, backoff: int, w_0=5):
    x = x.reshape(-1, 1)
    x_inv = 1 - 0
    z_gz = z > 0
    r = x_inv @ z_gz
    f = r * -backoff
    weights_new = weights + f
    return jnp.maximum(weights_new, w_0)


def sdp_search(weights, x, z, search: int, w_0=5):
    x = x.reshape(-1, 1)
    z_gz = z > 0
    z_gz_inv = 1 - z_gz
    r = x @ z_gz_inv
    f = r * search
    weights_new = weights + f
    return jnp.minimum(weights_new, w_0)

def dendrite_update(weights, x, z, capture: int, backoff: int, search: int, w_0=5, w_max=8):
    x = x.reshape(-1, 1)
    z_gz = z > 0
    x_inv = 1 - x
    z_gz_inv = 1 - z_gz
    r_capture = x @ z_gz
    r_backoff = x_inv @ z_gz
    r_search = x @ z_gz_inv
    delta_capture = r_capture * capture
    delta_backoff = r_backoff * -backoff
    delta_search = r_search * search
    weights_updated = weights + delta_capture + delta_backoff + delta_search
    weights_clipped = jnp.clip(weights_updated, w_0, w_max)
    return weights_clipped

def cv_unit(x, y, theta, q):
    # x is 1xp bit vector
    # y is 1x1 bit
    # theta is 1x1 int
    # q is int
    dend_out = dendrite_if(x, y, theta, q)
    dend_max = jnp.max(dend_out)
    if dend_max > 0:
        print(f"{dend_max.dtype=}, {dend_max.shape=}, {dend_max=}")
    return 1 if dend_max > 0 else 0

# if d is the number of units sampled in the receptive field
# D = 2d is the number of components of the vector after two-rail encoding
# C is the number of label classes for the classification problem, (for MNIST, C = 10)
def cv_group(rf_enc, label, theta):
    # rf_enc is 1xD bit vector
    # label is C bit vector
    C = label.shape[0]
    out = jnp.empty((1, C), dtype=jnp.uint8)
    for i in range(C):
        cvu_res = cv_unit(rf_enc, label[i], theta, C)
        out.at[0, i].set(cvu_res)
    return out




class Dendrite:
    def __init__(self, theta: float, p: int, q: int):
        self.theta = theta
        self.q = q

        self.weights = random.randint(key, (p, q), 0, 10)

    def inference(self, x, y):
        """
        x : 1xp bit vector, the distal inputs
        y: 1 bit, the proximal input
        """
        #min(x, y) between  bit vector and bit is just multiplication (bitwise AND)
        x_y_min = x * y
        pre_threshold = x_y_min @ self.weights
        threshold_mask = pre_threshold >= self.theta
        thresholded = pre_threshold * threshold_mask
        thresholded_first_max_idx = jnp.argmax(thresholded)
        thresholded_first_max_mask = jnp.array([1 if i == thresholded_first_max_idx else 0 for i in range(thresholded.shape[1])])
        return thresholded * thresholded_first_max_mask
        # return n_wta(thresholded)

    def update(self):
        pass

class CVGroup:
    def __init__(self, num_classes: int, theta: float, p: int, q: int):
        self.num_classes = num_classes
        self.theta = theta
        self.p = p
        self.q = q
        self.units = [Dendrite(theta, p, q) for _ in range(num_classes)]

    def inference(self, label, pat_enc):
        """
        label: C bit vector, one-hot encoding of label
        pat_enc: 1xD bit vector, the encoded version of one RF of the pattern
        """
        out = jnp.empty((self.num_classes,), dtype=jnp.uint8)
        for i in range(self.num_classes):
            dend_out = self.units[i].inference(pat_enc, label[i])
            out.at[i].set(1 if jnp.max(dend_out) > 0 else 0)
        return out


class CVGroupUnified:
    def __init__(self, num_classes: int, theta: float, D: int, Q: int):
        """
        num_classes: number of classes for the classification problem
        theta: threshold for the dendrites
        D: number of components of the vector after two-rail encoding
        Q: number of segments per dendrite (CV unit)
        """
        self.num_classes = num_classes
        self.theta = theta
        self.D = D
        self.Q = Q

        self.weights = random.randint(key, (num_classes, D, Q), 0, 10)

    def inference(self, label_C, pat_enc_1xD):
        """
        label: C bit vector, one-hot encoding of label
        pat_enc: 1xD bit vector, the encoded version of one RF of the pattern
        """
        label_Cx1 = label_C.reshape(-1, 1)
        min_results_CxD = label_Cx1 @ pat_enc_1xD
        pre_thresholds_CxQ = jnp.einsum('cd,cdq->cq', min_results_CxD, self.weights)
        threshold_masks_CxQ = pre_thresholds_CxQ >= self.theta
        thresholded_CxQ = pre_thresholds_CxQ * threshold_masks_CxQ
        thresholded_first_max_idx_Cx1 = jnp.argmax(thresholded_CxQ, axis=1)
        thresholded_first_max_mask_CxQ = jnp.eye(self.Q)[thresholded_first_max_idx_Cx1]
        dend_out_CxQ = thresholded_CxQ * thresholded_first_max_mask_CxQ
        dend_max_C = jnp.max(dend_out_CxQ, axis=1)
        out_C = jnp.array([1 if i > 0 else 0 for i in dend_max_C])
        return out_C


# Neuromorphic Online Clustering network

# from section 6.2:
# > To implement a supervised classifier, the designer determines the following.
# > 1) RF formation and encoding – these functions are both highly application dependent and crucial to the
# > quality (accuracy and cost) of the classifier.
# > 2) Network sizing: The number of RFs, the number of CV units per CV group (the number of labels), and
# > the number of segments per CV unit.
# > 3) Network tuning: For a representative data set, the parameters wmax, w0, threshold, capture, backoff, and
# > search are established via parameter sweeps. During normal operation, these global parameters are fixed
# > and are the same across all the segments in the network. To expedite the process a (small) representative
# > subset of RFs may be used.

class NOCNet:
    def __init__(self, params: dict):
        num_classes, thresh, num_rfs, rf_size, num_segs_per_dend = params['num_classes'], params['thresh'], params['num_rfs'], params['rf_size'], params['num_segs_per_dend']

        self.thresh = thresh
        self.C = num_classes
        self.R = num_rfs
        self.D = 2 * rf_size
        self.Q = num_segs_per_dend

        self.weights = random.randint(key, (self.R, self.C, self.D, self.Q), 0, 10, dtype=jnp.uint8)

    def inference(self, X, labels):
        """
        X: NxHxW tensor of bits, collectiion of binarized grayscale images
        label: Nx1 tensor of ints, the labels range from 0 to 9
        """

        N = X.shape[0]

        print(f"{X.shape=}, {labels.shape=}, {labels.min()=}, {labels.max()=}")

        out_size = jnp.sqrt(self.R).astype(int)

        print(f"{out_size=}")

        # form receptive fields
        rfs_NxOxOxD = jnp.zeros((N, out_size, out_size, self.D), dtype=jnp.uint8)
        for i in range(out_size):
            for j in range(out_size):
                rf_nz_idxs = jnp.array([0, 2, 4, 10, 12, 14, 20, 22, 24])
                rf_Nx5x5 = X[:, i:i+5, j:j+5]
                rf_Nx25 = rf_Nx5x5.reshape(N, -1)
                rf_Nx9 = rf_Nx25[:, rf_nz_idxs]

                # number of nonzero:
                # for u in range(N):
                #     num_nz = jnp.sum(jnp.where(rf_Nx9[u, :] > 0, 1, 0)).item()
                #     if num_nz > 0:
                #         print(f"{num_nz=}")

                # two-rail encode. 9x1 -> 18x1
                # 0 -> 01, 1 -> 10
                mapping = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8)
                enc_Nx18 = mapping[rf_Nx9].reshape(1000, -1)

                rfs_NxOxOxD.at[:, i, j, :].set(enc_Nx18)

        # dual-rail encode each RF output, giving  R x D tensor (or R 1xD tensors)
        rfs_NxRxD = rfs_NxOxOxD.reshape(N, -1, self.D)

        print(f"Formed RFs, {rfs_NxRxD.shape=}")

        labels_one_hot_NxC = jnp.eye(self.C)[labels]

        print(f"{labels_one_hot_NxC.shape=}")

        # there are R CV groups, each gets a D-bit distal input (RF) and C-bit proximal input (label)
        for i in range(N):
            print(f"Processing image {i}")
            # TODO: process through all R CV groups in parallel

            # need (C x 1) * (R x D) -> R x C x D
            min_results_RxCxD = jnp.einsum('c,rd->rcd', labels_one_hot_NxC[i, :], rfs_NxRxD[i, :, :])

            pre_thresholds_RxCxQ = jnp.einsum('rcd,rcdq->rcq', min_results_RxCxD, self.weights)
            threshold_masks_RxCxQ = pre_thresholds_RxCxQ >= self.thresh
            thresholded_RxCxQ = pre_thresholds_RxCxQ * threshold_masks_RxCxQ
            thresholded_first_max_idx_RxC = jnp.argmax(thresholded_RxCxQ, axis=2)

            thresholded_first_max_mask_RxCxQ = jnp.eye(self.Q)[thresholded_first_max_idx_RxC]
            dend_out_RxCxQ = thresholded_RxCxQ * thresholded_first_max_mask_RxCxQ
            dend_max_RxC = jnp.max(dend_out_RxCxQ, axis=2)
            out_RxC = jnp.array([[1 if i > 0 else 0 for i in dend_max_RxC[r, :]] for r in range(self.R)])

            # create C summation units, each taking R inputs
            sums_C = jnp.sum(out_RxC, axis=0)

            # winner take all mask
            sums_first_max_idx = jnp.argmax(sums_C)
            predicted_class_C = jnp.array([1 if i == sums_first_max_idx else 0 for i in range(sums_C.shape[0]) ])
            print(f"{predicted_class_C=}")

            self.update()

    def update(self):
        # TODO: update all
        # NOTE: single dendrite update code
        # x = x.reshape(-1, 1)
        # z_gz = z > 0
        # x_inv = 1 - x
        # z_gz_inv = 1 - z_gz
        # r_capture = x @ z_gz
        # r_backoff = x_inv @ z_gz
        # r_search = x @ z_gz_inv
        # delta_capture = r_capture * capture
        # delta_backoff = r_backoff * -backoff
        # delta_search = r_search * search
        # weights_updated = weights + delta_capture + delta_backoff + delta_search
        # weights_clipped = jnp.clip(weights_updated, w_0, w_max)
        # return weights_clipped
        raise NotImplementedError





def binary_mnist_net():
    X_train, y_train, X_test, y_test = get_binarized_mnist()

    rf_kernel = jnp.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ], dtype=jnp.uint8)

    # 28x28 w/ a 5x5 kernel yields 24x24 = 576 receptive fields
    out_size = 24

    num_nonzero = int(jnp.sum(rf_kernel))

    N = X_train.shape[0]
    O = out_size
    P = O*O
    D = 2 * num_nonzero

    X_train_rf_NxOxOxD = jnp.zeros((N, out_size, out_size, 2*num_nonzero), dtype=jnp.uint8)
    for i in range(out_size):
        for j in range(out_size):
            rf_nz_idxs = jnp.array([0, 2, 4, 10, 12, 14, 20, 22, 24])
            rf_Nx5x5 = X_train[:, i:i+5, j:j+5]
            rf_Nx25 = rf_Nx5x5.reshape(N, -1)
            rf_Nx9 = rf_Nx25[:, rf_nz_idxs]

            # number of nonzero:
            # for u in range(N):
            #     num_nz = jnp.sum(jnp.where(rf_Nx9[u, :] > 0, 1, 0)).item()
            #     if num_nz > 0:
            #         print(f"{num_nz=}")

            # two-rail encode. 9x1 -> 18x1
            # 0 -> 01, 1 -> 10
            mapping = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8)
            enc_Nx18 = mapping[rf_Nx9].reshape(1000, -1)

            X_train_rf_NxOxOxD.at[:, i, j, :].set(enc_Nx18)

    X_train_rf_NxPxD = X_train_rf_NxOxOxD.reshape(N, -1, 18)

    print(f"Formed RFs, {X_train_rf_NxPxD.shape=}")

    theta = 9

    # one-hot encode the labels
    n_values = 10
    y_train_one_hot = jnp.eye(n_values)[y_train]

    num_segments_per_dendrite = 16

    cvgroups = [CVGroupUnified(n_values, theta, D, num_segments_per_dendrite) for _ in range(P)]

    for i in range(N):
        print(f"Processing image {i}")
        # TODO: this is very slow, parallelize
        for u in range(P):
            # cv_group_res = cv_group(X_train_rf_NxPxD[i, u, :].reshape(1, -1), y_train_one_hot[i, :], theta)
            cvgroups[u].inference(y_train_one_hot[i, :], X_train_rf_NxPxD[i, u, :].reshape(1, -1))


def run():
    # inputs
    p = 5
    x = random.normal(key, (1, p))
    theta = random.randint(key, (1, 1), 0, 10)

    print(f"{x=}, {theta=}")

    # segment
    seg_w = random.randint(key, (p, 1), 0, 10)
    tmp1 = jnp.dot(x, seg_w)
    tmp2 = tmp1 >= theta
    seg_out = tmp1 * tmp2
    print(f"{seg_w=}, {tmp1=}, {tmp2=}, {seg_out=}")

    print("=============")

    q = 7
    # x = random.randint(key, (1, q), 0, 7)
    x = jnp.array([[1, 6, 2, 3, 1, 6, 0]], dtype=jnp.int8)
    x_wta = n_wta(x)
    print(x)
    print(x_wta)

    print("=============")

    y = 1
    theta = 29
    q = 7
    dif_out = dendrite_if(x, y, theta, q)
    print(f"{dif_out=}")

    print("=============")


    # binary_mnist_net()

    X_train, y_train, X_test, y_test = get_binarized_mnist()

    num_classes = 10
    thresh = 7
    num_rfs = 576
    rf_size = 9
    num_segs_per_dend = 18
    params = {
        'num_classes': num_classes,
        'theta': thresh,
        'num_rfs': num_rfs,
        'rf_size': rf_size,
        'num_segs_per_dend': num_segs_per_dend
    }
    nocnet = NOCNet(params)
    nocnet.inference(X_train, y_train)


if __name__ == "__main__":
    run()

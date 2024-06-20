from data import get_binarized_mnist

from operator import itemgetter

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

rand_seed = 299792458
key = random.PRNGKey(rand_seed)

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
        attrs = ['num_classes', 'thresh', 'num_rfs', 'rf_size', 'num_segs_per_dend', 'capture', 'backoff', 'search', 'w_0', 'w_max']
        num_classes, thresh, num_rfs, rf_size, num_segs_per_dend, capture, backoff, search, w_0, w_max = itemgetter(*attrs)(params)


        self.thresh = thresh
        self.C = num_classes
        self.R = num_rfs
        self.D = 2 * rf_size
        self.Q = num_segs_per_dend
        self.capture = capture
        self.backoff = backoff
        self.search = search
        self.w_0 = w_0
        self.w_max = w_max

        self.weights_RxCxDxQ = random.randint(key, (self.R, self.C, self.D, self.Q), 0, 10, dtype=jnp.uint8)

        # RF
        self.rf_nonzero_idxs = jnp.array([0, 2, 4, 10, 12, 14, 20, 22, 24])

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
        # TODO: empty instead of zeros?
        rfs_NxOxOxD = jnp.zeros((N, out_size, out_size, self.D), dtype=jnp.uint8)
        for i in range(out_size):
            for j in range(out_size):
                rf_Nx5x5 = X[:, i:i+5, j:j+5]
                rf_Nx25 = rf_Nx5x5.reshape(N, -1)
                rf_Nx9 = rf_Nx25[:, self.rf_nonzero_idxs]

                # number of nonzero:
                # for u in range(N):
                #     num_nz = jnp.sum(jnp.where(rf_Nx9[u, :] > 0, 1, 0)).item()
                #     if num_nz > 0:
                #         print(f"{num_nz=}")

                # two-rail encode. 9x1 -> 18x1
                # assuming (0 -> 01, 1 -> 10), paper doesn't clarify
                mapping = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8)
                enc_Nx18 = mapping[rf_Nx9].reshape(1000, -1)

                rfs_NxOxOxD.at[:, i, j, :].set(enc_Nx18)

        # dual-rail encode each RF output, giving  N (R x D) encoded tensors
        rfs_NxRxD = rfs_NxOxOxD.reshape(N, -1, self.D)

        print(f"Formed RFs, {rfs_NxRxD.shape=}")

        labels_one_hot_NxC = jnp.eye(self.C)[labels]

        print(f"{labels_one_hot_NxC.shape=}")

        # there are R CV groups, each gets a D-bit distal input (RF) and C-bit proximal input (label)
        # this is online learning, no batching of inputs allowed
        for i in range(N):
            print(f"Processing image {i}, label = {labels[i]}")

            # need (C x 1) * (R x D) -> R x C x D
            rf_patterns_RxD = rfs_NxRxD[i, :, :]
            min_results_RxCxD = jnp.einsum('c,rd->rcd', labels_one_hot_NxC[i, :], rf_patterns_RxD)

            pre_thresholds_RxCxQ = jnp.einsum('rcd,rcdq->rcq', min_results_RxCxD, self.weights_RxCxDxQ)
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

            weights_delta_RxCxDxQ = self.update_weights(rf_patterns_RxD, dend_out_RxCxQ)
            weights_abs_diff_RxC = jnp.sum(jnp.abs(weights_delta_RxCxDxQ), axis=(2, 3))
            weights_net_delta_RxC = jnp.sum(weights_delta_RxCxDxQ, axis=(2,3))

            O = int(jnp.sqrt(self.R))
            weights_abs_diff_OxOxC = weights_abs_diff_RxC.reshape(O, O, self.C)

            print(jnp.sum(weights_abs_diff_OxOxC))

            halfC = int(self.C / 2)
            plt.figure(figsize=(halfC * 3 + 2, 6))
            for i in range(2):
                for j in range(halfC):
                    idx = i * halfC + j
                    plt.subplot(2, halfC, idx + 1)
                    plt.imshow(weights_abs_diff_OxOxC[:, :, idx], cmap="binary")
                    plt.title(f"Σ|Δ|, Class = {idx}")

            plt.suptitle(f"Weight update sum of absolute deviances, for all {self.C} classes")
            plt.show()


    # from NOCAC Fig. 6 caption: "Note that for the update function (Figure 8), the int output A is binarized to bits, i.e., spikes."
    # so the output of dendrite inference function is a Q-bit vector. there are R * C dendrites (1-1 correspondence between CV units and dendrites,
    # R CV groups, each group has C CV units))
    def update_weights(self, X_RxD, Z_RxCxQ):
        """
        X: RxD bit matrix, the receptive fields for a given image
        Z: RxCxQ bit tensor, the output of the CV units
        """
        Z_bin_RxCxQ = jnp.where(Z_RxCxQ > 0, 1, 0).astype(jnp.uint8)

        X_inv_RxD = (1 - X_RxD)
        Z_bin_inv_RxCxQ = 1 - Z_bin_RxCxQ

        r_capture = jnp.einsum('rd,rcq->rcdq', X_RxD, Z_bin_RxCxQ)
        r_backoff = jnp.einsum('rd,rcq->rcdq', X_inv_RxD, Z_bin_RxCxQ)
        r_search = jnp.einsum('rd,rcq->rcdq', X_RxD, Z_bin_inv_RxCxQ)

        delta_capture = r_capture * self.capture
        delta_backoff = r_backoff * -self.backoff
        delta_search = r_search * self.search
        weights_updated_RxCxDxQ = self.weights_RxCxDxQ + delta_capture + delta_backoff + delta_search
        weights_clipped_RxCxDxQ = jnp.clip(weights_updated_RxCxDxQ, self.w_0, self.w_max)
        weights_delta_RxCxDxQ = weights_clipped_RxCxDxQ - self.weights_RxCxDxQ
        self.weights_RxCxDxQ = weights_clipped_RxCxDxQ
        return weights_delta_RxCxDxQ




def run():
    rf_kernel = jnp.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ], dtype=jnp.uint8)

    rf_size = int(jnp.sum(rf_kernel))

    # X_train, y_train, X_test, y_test = get_binarized_mnist()
    X_train, y_train, X_test, y_test = get_binarized_mnist(restricted_labels=[0, 1], train_size=1000, test_size=1000)

    # num_classes = 10
    num_classes = 2 # binary, binarized MNIST
    thresh = 7
    num_rfs = 576
    num_segs_per_dend = 16
    # search << backoff, capture
    capture = 10
    backoff = 10
    search = 1
    w_0 = 5
    w_max = 8
    params = {
        'num_classes': num_classes,
        'thresh': thresh,
        'num_rfs': num_rfs,
        'rf_size': rf_size,
        'num_segs_per_dend': num_segs_per_dend,
        'capture': capture,
        'backoff': backoff,
        'search': search,
        'w_0': w_0,
        'w_max': w_max
    }
    nocnet = NOCNet(params)
    nocnet.inference(X_train, y_train)


if __name__ == "__main__":
    run()

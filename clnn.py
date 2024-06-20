from data import get_binarized_mnist

import jax.numpy as jnp
from jax import random

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
        # TODO: empty instead of zeros?
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
        for i in range(N):
            print(f"Processing image {i}")
            # TODO: process through all R CV groups in parallel

            # need (C x 1) * (R x D) -> R x C x D
            rf_patterns = rfs_NxRxD[i, :, :]
            min_results_RxCxD = jnp.einsum('c,rd->rcd', labels_one_hot_NxC[i, :], rf_patterns)

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

            self.update_weights(rf_patterns, dend_out_RxCxQ)

    # from NOCAC Fig. 6 caption: "Note that for the update function (Figure 8), the int output A is binarized to bits, i.e., spikes."
    # so the output of dendrite inference function is a Q-bit vector. there are R * C dendrites (1-1 correspondence between CV units and dendrites,
    # R CV groups, each group has C CV units))
    def update_weights(self, X_RxD, Z_RxCxQ):
        """
        X: RxD bit matrix, the receptive fields for a given image
        Z: RxCxQ bit tensor, the output of the CV units
        """
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





def run():
    rf_kernel = jnp.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ], dtype=jnp.uint8)

    rf_size = int(jnp.sum(rf_kernel))

    X_train, y_train, X_test, y_test = get_binarized_mnist()

    num_classes = 10
    thresh = 7
    num_rfs = 576
    num_segs_per_dend = 16
    params = {
        'num_classes': num_classes,
        'thresh': thresh,
        'num_rfs': num_rfs,
        'rf_size': rf_size,
        'num_segs_per_dend': num_segs_per_dend
    }
    nocnet = NOCNet(params)
    nocnet.inference(X_train, y_train)


if __name__ == "__main__":
    run()

from data import load_data, make_binarized_mnist

from functools import reduce
from operator import itemgetter
import pathlib
import random

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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
    def __init__(self, params: dict, key):
        print(f"Initializing NOCNet with params: {params}")
        attrs = ['num_classes', 'thresh', 'num_rfs', 'rf_size', 'num_segs_per_dend', 'capture', 'backoff', 'search', 'w_max']
        num_classes, thresh, num_rfs, rf_size, num_segs_per_dend, capture, backoff, search, w_max = itemgetter(*attrs)(params)


        self.thresh = thresh
        self.C = num_classes
        self.R = num_rfs
        self.D = 2 * rf_size
        self.Q = num_segs_per_dend
        self.capture = capture
        self.backoff = backoff
        self.search = search
        self.w_max = w_max

        self.weights_RxCxDxQ = jax.random.randint(key, (self.R, self.C, self.D, self.Q), 0, 10, dtype=jnp.uint8)
        print(f"Number of parameters: {self.weights_RxCxDxQ.size}")

        # RF
        self.rf_nonzero_idxs = jnp.array([0, 2, 4, 10, 12, 14, 20, 22, 24])

    def form_receptive_fields(self, X):
        N = X.shape[0]
        out_size = jnp.sqrt(self.R).astype(int)

        rfs_NxOxOxD = jnp.empty((N, out_size, out_size, self.D), dtype=jnp.uint8)
        for i in range(out_size):
            for j in range(out_size):
                rf_Nx5x5 = X[:, i:i+5, j:j+5]
                rf_Nx25 = rf_Nx5x5.reshape(N, -1)
                rf_Nx9 = rf_Nx25[:, self.rf_nonzero_idxs]

                # two-rail encode. 9x1 -> 18x1
                # assuming (0 -> 01, 1 -> 10), paper doesn't clarify
                mapping = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint8)
                enc_Nx18 = mapping[rf_Nx9].reshape(N, -1)

                rfs_NxOxOxD = rfs_NxOxOxD.at[:, i, j, :].set(enc_Nx18)

        rfs_NxRxD = rfs_NxOxOxD.reshape(N, -1, self.D)
        return rfs_NxRxD


    def inference(self, X):
        return self.supervised_learning(X, jnp.ones(X.shape[0], dtype=jnp.uint8))


    def supervised_learning(self, X, labels):
        """
        X: NxHxW tensor of bits, collectiion of binarized grayscale images
        label: NxC tensor of labels (one-hot)
        """

        N = X.shape[0]
        rfs_NxRxD = self.form_receptive_fields(X)

        # there are R CV groups, each gets a D-bit distal input (RF) and C-bit proximal input (label)
        # this is online learning, no batching of inputs allowed
        predictions = []

        for i in range(N):
            print(f"\nProcessing image {i}, label = {jnp.argmax(labels[i])}\n------------------------------")

            # need (C x 1) * (R x D) -> R x C x D
            rf_patterns_RxD = rfs_NxRxD[i, :, :]

            min_results_RxCxD = jnp.einsum('c,rd->rcd', labels[i, :], rf_patterns_RxD)
            pre_thresholds_RxCxQ = jnp.einsum('rcd,rcdq->rcq', min_results_RxCxD, self.weights_RxCxDxQ)

            threshold_masks_RxCxQ = pre_thresholds_RxCxQ >= self.thresh
            thresholded_RxCxQ = pre_thresholds_RxCxQ * threshold_masks_RxCxQ
            thresholded_first_max_idx_RxC = jnp.argmax(thresholded_RxCxQ, axis=2)

            thresholded_first_max_mask_RxCxQ = jnp.eye(self.Q)[thresholded_first_max_idx_RxC]
            dend_out_RxCxQ = thresholded_RxCxQ * thresholded_first_max_mask_RxCxQ
            dend_max_RxC = jnp.max(dend_out_RxCxQ, axis=2)

            cvu_out_RxC = jnp.array([[1 if i > 0 else 0 for i in dend_max_RxC[r, :]] for r in range(self.R)])

            # create C summation units, each taking R inputs
            sums_C = jnp.sum(cvu_out_RxC, axis=0)

            # winner take all mask
            predicted_digit = jnp.argmax(sums_C)
            predicted_class_C = jnp.array([1 if i == predicted_digit else 0 for i in range(sums_C.shape[0]) ])
            predictions.append(predicted_class_C)

            # perform update
            weights_delta_RxCxDxQ = self.update_weights(rf_patterns_RxD, dend_out_RxCxQ)

            def plot_stuff():
                # print(f"average weight delta = {weights_delta_RxCxDxQ.mean().item()}")
                weights_abs_diff_RxC = jnp.sum(jnp.abs(weights_delta_RxCxDxQ), axis=(2, 3))
                weights_net_delta_RxC = jnp.sum(weights_delta_RxCxDxQ, axis=(2,3))

                O = int(jnp.sqrt(self.R))
                halfC = int(self.C / 2)

                # display cvu_out
                cvu_out_OxOxC = cvu_out_RxC.reshape(O, O, self.C)

                plt.figure(figsize=(halfC * 3 + 2, 6))
                for i in range(self.C):
                    for j in range(halfC):
                        idx = i * halfC + j
                        plt.subplot(2, halfC, idx + 1)
                        plt.imshow(cvu_out_OxOxC[:, :, idx], cmap="binary")
                        plt.title(f"CVU out, Class = {idx}")

                plt.suptitle(f"CVU output, for all {self.C} classes")
                # plt.show()
                plt.close()


                # display sums_C
                plt.figure(figsize=(6, 6))
                plt.imshow(sums_C.reshape(1, -1), cmap="binary")
                plt.title("Summation units")
                # plt.show()
                plt.close()


                # display weights_abs_diff
                weights_abs_diff_OxOxC = weights_abs_diff_RxC.reshape(O, O, self.C)
                # print("sum of absolute differences: ", jnp.sum(weights_abs_diff_OxOxC))
                plt.figure(figsize=(halfC * 3 + 2, 6))
                for i in range(2):
                    for j in range(halfC):
                        idx = i * halfC + j
                        plt.subplot(2, halfC, idx + 1)
                        plt.imshow(weights_abs_diff_OxOxC[:, :, idx], cmap="binary")
                        plt.title(f"Σ|Δ|, Class = {idx}")

                plt.suptitle(f"Weight update sum of absolute deviances, for all {self.C} classes")
                # plt.show()
                plt.close()

            # plot_stuff()

        return predictions

    # from NOCAC Fig. 6 caption: "Note that for the update function (Figure 8), the int output A is binarized to bits, i.e., spikes."
    # so the output of dendrite inference function is a Q-bit vector. there are R * C dendrites (1-1 correspondence between CV units and dendrites,
    # R CV groups, each group has C CV units))
    def update_weights(self, X_RxD, Z_RxCxQ):
        """
        X: RxD bit matrix, the receptive fields for a given image
        Z: RxCxQ bit tensor, the output of the CV units
        """
        # print("----------------------------------")
        # print(f"{self.weights_RxCxDxQ[0, 0, :, :]=}")

        # num_active_X = jnp.sum(jnp.where(X_RxD > 0, 1, 0)).item()
        # num_active_Z = jnp.sum(jnp.where(Z_RxCxQ > 0, 1, 0)).item()

        # if num_active_X > 0 or num_active_Z > 0:
        #     print("--- update weights, input ---")
        #     print(f"{num_active_X=}")
        #     print(f"{num_active_Z=}")
        #     print()

        Z_bin_RxCxQ = jnp.where(Z_RxCxQ > 0, 1, 0).astype(jnp.uint8)

        X_inv_RxD = (1 - X_RxD)
        Z_bin_inv_RxCxQ = 1 - Z_bin_RxCxQ

        r_capture = jnp.einsum('rd,rcq->rcdq', X_RxD, Z_bin_RxCxQ)
        r_backoff = jnp.einsum('rd,rcq->rcdq', X_inv_RxD, Z_bin_RxCxQ)
        r_search = jnp.einsum('rd,rcq->rcdq', X_RxD, Z_bin_inv_RxCxQ)

        delta_capture = r_capture * self.capture
        delta_backoff = r_backoff * -self.backoff
        delta_search = r_search * self.search

        # num_active_delta_capture = jnp.sum(jnp.where(delta_capture > 0, 1, 0)).item()
        # num_active_delta_backoff = jnp.sum(jnp.where(delta_backoff < 0, 1, 0)).item()
        # num_active_delta_search = jnp.sum(jnp.where(delta_search > 0, 1, 0)).item()

        # if num_active_delta_capture > 0 or num_active_delta_backoff > 0 or num_active_delta_search > 0:
        #     print("--- update weights, delta ---")
        #     print(f"{num_active_delta_capture=}")
        #     print(f"{num_active_delta_backoff=}")
        #     print(f"{num_active_delta_search=}")
        #     print()

        weights_updated_RxCxDxQ = self.weights_RxCxDxQ + delta_capture + delta_backoff + delta_search
        weights_clipped_RxCxDxQ = jnp.clip(weights_updated_RxCxDxQ, 0, self.w_max)
        weights_delta_RxCxDxQ = weights_clipped_RxCxDxQ - self.weights_RxCxDxQ
        self.weights_RxCxDxQ = self.weights_RxCxDxQ.at[:, :, :, :].set(weights_clipped_RxCxDxQ)
        # print(f"{self.weights_RxCxDxQ[0, 0, :, :]=}")
        # print(f"{weights_delta_RxCxDxQ[0, 0, :, :]=}")

        # print(f"weights updated, mean absolute magnitude = {jnp.mean(jnp.abs(weights_delta_RxCxDxQ)).item()}")
        return weights_delta_RxCxDxQ




def set_seed(seed):
    random.seed(seed)


def run():
    rand_seed = 299792458
    set_seed(rand_seed)
    key = jax.random.PRNGKey(rand_seed)

    print(f"{jax.devices()=}")

    rf_kernel = jnp.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ], dtype=jnp.uint8)

    rf_size = int(jnp.sum(rf_kernel))

    labels_01 = [0, 1]
    labels_012 = [0, 1, 2]
    labels_01234 = list(range(5))
    labels_all = list(range(10))

    labels = labels_01
    num_classes = len(labels)
    train_valid_test_per_class = [500, 100, 100]

    data_file_name = 'bin-mnist-01.npz'
    data_dir_path = 'data'
    data_file_path = f'{data_dir_path}/{data_file_name}'

    pathlib.Path(data_dir_path).mkdir(parents=True, exist_ok=True)

    try:
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(data_file_path)
    except:
        make_binarized_mnist(train_valid_test_per_class, key, data_file_path, restricted_labels=labels)
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(data_file_path)

    # plot first few images
    num_plots = (5, 5)
    plt.figure(figsize=(15, num_plots[0] * 3.4))
    plt.suptitle(f"First {reduce(lambda x, y: x*y, num_plots)} images")
    for i in range(num_plots[0]):
        for j in range(num_plots[1]):
            idx = i * num_plots[1] + j
            plt.subplot(num_plots[0], num_plots[1], idx + 1)
            plt.axis("off")
            plt.imshow(X_train[idx].reshape(28, 28), cmap="binary")
            plt.title(f"Label: {y_train[idx]}")
    plt.show()


    # one-hot encode
    I_C = jnp.eye(num_classes)
    y_train_oh = I_C[y_train]
    y_valid_oh = I_C[y_valid]
    y_test_oh = I_C[y_test]

    thresh = 5
    num_rfs = 576
    num_segs_per_dend = 16
    # search << backoff, capture
    capture = 10
    backoff = 8
    search = 3
    # w_0 doesnt actually show up in the unified dendrite update circuit, so unclear where
    # it comes into play
    # w_0 = 5
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
        # 'w_0': w_0,
        'w_max': w_max
    }
    nocnet = NOCNet(params, key)
    results_train = nocnet.supervised_learning(X_train, y_train_oh)

    results_test = nocnet.inference(X_test)


if __name__ == "__main__":
    run()

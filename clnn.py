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

    for i in range(N):
        print(f"Processing image {i}")
        # TODO: this is very slow, parallelize
        for u in range(P):
            cv_group_res = cv_group(X_train_rf_NxPxD[i, u, :].reshape(1, -1), y_train_one_hot[i, :], theta)




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

    binary_mnist_net()


if __name__ == "__main__":
    run()

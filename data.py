from datasets import load_dataset
import jax.numpy as jnp
import matplotlib.pyplot as plt

def get_binarized_mnist():
    ds = load_dataset("mnist")
    ds = ds.with_format("jax")

    ds["train"] = ds["train"][:1000]
    ds["test"] = ds["test"][1000:2000]

    X_train = ds["train"]["image"] / 255.0
    y_train = ds["train"]["label"].astype("uint8")
    X_test = ds["test"]["image"] / 255.0
    y_test = ds["test"]["label"].astype("uint8")

    # Binarize the images
    threshold = 0.4
    X_train = (X_train > threshold).astype("uint8")
    X_test = (X_test > threshold).astype("uint8")

    print(f"{X_train.dtype=}, {X_train.shape=}")
    print(f"{X_test.dtype=}, {X_test.shape=}")
    print(f"{y_train.dtype=}, {y_train.shape=}")
    print(f"{y_test.dtype=}, {y_test.shape=}")

    # plot first few images
    # plt.figure(figsize=(12, 5))
    # num_plots = 5
    # plt.suptitle(f"First {num_plots} images")
    # for i in range(num_plots):
    #     plt.subplot(1, num_plots, i+1)
    #     plt.imshow(X_train[i].reshape(28, 28), cmap="binary")
    #     plt.title(f"Label: {y_train[i]}")
    # plt.show()

    return X_train, y_train, X_test, y_test

from functools import reduce

from datasets import load_dataset
import jax.numpy as jnp
import matplotlib.pyplot as plt

# TODO: enforce same number of labels per class
def get_binarized_mnist(restricted_labels=None, train_size=1000, test_size=1000):
    ds = load_dataset("mnist")
    ds = ds.with_format("jax")

    train_data = ds["train"]
    test_data = ds["test"]

    if restricted_labels is not None:
        train_data = train_data.filter(lambda x: x["label"] in restricted_labels)
        test_data = test_data.filter(lambda x: x["label"] in restricted_labels)

    train_data = train_data.take(train_size)
    test_data = test_data.take(test_size)

    X_train = train_data["image"] / 255.0
    y_train = train_data["label"].astype("uint8")
    X_test = test_data["image"] / 255.0
    y_test = test_data["label"].astype("uint8")

    # Binarize the images
    threshold = 0.45
    X_train = (X_train > threshold).astype("uint8")
    X_test = (X_test > threshold).astype("uint8")

    print(f"{X_train.dtype=}, {X_train.shape=}")
    print(f"{X_test.dtype=}, {X_test.shape=}")
    print(f"{y_train.dtype=}, {y_train.shape=}")
    print(f"{y_test.dtype=}, {y_test.shape=}")

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

    return X_train, y_train, X_test, y_test

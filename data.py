from functools import reduce

from datasets import load_dataset
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def get_binarized_mnist(train_valid_test_per_class, key, restricted_labels=None):
    ds = load_dataset("mnist")
    ds = ds.with_format("jax")

    if restricted_labels is not None:
        num_classes = len(restricted_labels)
    else:
        num_classes = np.argmax(ds["train"]["label"]) + 1
        restricted_labels = list(range(num_classes))

    train_data = ds["train"]
    test_data = ds["test"]

    train_per_class, valid_per_class, test_per_class = train_valid_test_per_class
    train_valid_per_class = train_per_class + valid_per_class

    train_indices = []
    valid_indices = []
    for label in restricted_labels:
        indices = jnp.where(train_data["label"] == label)[0]
        key, subkey = jax.random.split(key)
        sampled_indices = jax.random.choice(subkey, indices, shape=(train_valid_per_class,), replace=False)
        sampled_train_indices = sampled_indices[:train_per_class]
        sampled_valid_indices = sampled_indices[train_per_class:]
        train_indices.extend(sampled_train_indices.tolist())
        valid_indices.extend(sampled_valid_indices.tolist())

    test_indices = []
    for label in restricted_labels:
        indices = jnp.where(test_data["label"] == label)[0]
        key, subkey = jax.random.split(key)
        sampled_indices = jax.random.choice(subkey, indices, shape=(test_per_class,), replace=False)
        test_indices.extend(sampled_indices.tolist())

    train_indices = jnp.array(train_indices)
    valid_indices = jnp.array(valid_indices)
    test_indices = jnp.array(test_indices)

    key, subkey = jax.random.split(key)
    train_indices = jax.random.permutation(key, train_indices)
    key, subkey = jax.random.split(key)
    valid_indices = jax.random.permutation(key, valid_indices)
    key, subkey = jax.random.split(key)
    test_indices = jax.random.permutation(key, test_indices)

    valid_data = {
        "image": train_data["image"][valid_indices],
        "label": train_data["label"][valid_indices]
    }

    train_data = {
        "image": train_data["image"][train_indices],
        "label": train_data["label"][train_indices]
    }

    test_data = {
        "image": test_data["image"][test_indices],
        "label": test_data["label"][test_indices]
    }

    X_train = train_data["image"] / 255.0
    y_train = train_data["label"].astype("uint8")
    X_valid = valid_data["image"] / 255.0
    y_valid = valid_data["label"].astype("uint8")
    X_test = test_data["image"] / 255.0
    y_test = test_data["label"].astype("uint8")

    # Binarize the images
    threshold = 0.45
    X_train = (X_train > threshold).astype("uint8")
    X_valid = (X_valid > threshold).astype("uint8")
    X_test = (X_test > threshold).astype("uint8")

    print(f"{X_train.dtype=}, {X_train.shape=}")
    print(f"{X_valid.dtype=}, {X_valid.shape=}")
    print(f"{X_test.dtype=}, {X_test.shape=}")
    print(f"{y_train.dtype=}, {y_train.shape=}")
    print(f"{y_valid.dtype=}, {y_valid.shape=}")
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

    return X_train, y_train, X_valid, y_valid, X_test, y_test

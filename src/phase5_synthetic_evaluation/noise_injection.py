import numpy as np


def inject_label_noise(X, y, noise_rate=0.1, random_state=42):
    """
    Inject synthetic label noise

    Parameters:
    X : Features
    y : Labels
    noise_rate : percentage of labels flipped

    Returns:
    X
    y_noisy
    noisy_indices
    """

    np.random.seed(random_state)

    y_noisy = y.copy()

    n_samples = len(y)

    n_noisy = int(noise_rate * n_samples)

    noisy_indices = np.random.choice(
        n_samples,
        n_noisy,
        replace=False
    )

    classes = np.unique(y)

    for idx in noisy_indices:

        current_label = y[idx]

        other_labels = classes[
            classes != current_label
        ]

        new_label = np.random.choice(other_labels)

        y_noisy[idx] = new_label

    return X, y_noisy, noisy_indices

"""Your three tasks. Start with docs/WORKSHOP.md; keep these signatures unchanged.

Everything uses NumPy arrays, not MNE objects, after loading the recording.
"""
import numpy as np
from scipy.signal import welch  # Hint for Task 1
from sklearn.linear_model import LogisticRegression  # Hint for Task 2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(window: np.ndarray, sfreq: float, bands) -> np.ndarray:
    """TASK 1: return channel-major log band powers as a 1-D array.

    window: (n_samples, n_channels), already causally bandpass-filtered, in volts.
    bands: (("mu", 8.0, 12.0), ("beta", 13.0, 30.0)).
    Return: shape (n_channels * len(bands),), usually (6,).

    Steps:
      1. Welch PSD along axis=0, using a 1-second segment (nperseg=int(sfreq)).
      2. For each channel, then each band, select low <= frequency < high.
      3. Integrate PSD: sum(selected PSD) * frequency spacing.
      4. Take natural log after adding 1e-20 to avoid log(0).
      5. Return a flat float array in the documented order.
    """
    # Welch splits the window into overlapping 1-second segments, estimates a
    # spectrum for each, and averages them. axis=0 because axis 0 is TIME;
    # frequencies[k] labels row k of psd, and psd[:, c] is channel c.
    frequencies, psd = welch(window, fs=sfreq, nperseg=int(sfreq), axis=0)

    # Spacing between neighbouring frequency bins, in Hz. At 160 Hz with
    # nperseg=160 this is exactly 1.0 Hz.
    df = frequencies[1] - frequencies[0]

    features = []
    # Channel-major order: all bands of C3, then all bands of Cz, then C4.
    for channel in range(window.shape[1]):
        for _name, low, high in bands:
            # Half-open [low, high) so mu and beta can never claim the same bin.
            mask = (frequencies >= low) & (frequencies < high)
            # PSD is in V^2/Hz; summing bins and scaling by Hz approximates the
            # integral, giving band power in V^2.
            power = psd[mask, channel].sum() * df
            # Log compresses the huge dynamic range of EEG power into something
            # a linear model can use. The epsilon keeps log(0) from blowing up
            # on an all-zero channel.
            features.append(np.log(power + 1e-20))

    return np.asarray(features, dtype=float)


def fit_classifier(X: np.ndarray, y: np.ndarray):
    """TASK 2: fit and return StandardScaler → LogisticRegression.

    X: (n_training_windows, n_features); y: (n_training_windows,).
    Labels: 0=feet, 1=hands. Only training data reaches this function.
    Hint: make_pipeline(...); use max_iter=1000 and random_state=42.
    The returned object must support predict, predict_proba, and classes_.
    """
    # The pipeline is what keeps the evaluation honest: calling .fit() here
    # learns the scaler's mean/std from THESE windows only. Later, .predict on
    # run 14 reuses those statistics instead of recomputing them from test data.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X, y)
    return model


def probability_to_velocity(p_hands: float, speed: float,
                            low: float, high: float) -> float:
    """TASK 3: map an already smoothed P(hands) to vertical velocity in m/s.

    p_hands >= high → +speed (up)
    p_hands <= low  → -speed (down)
    Otherwise      → 0.0 (hover)

    The provided controller handles rest gating and smoothing before calling you.
    """
    # Thresholds are inclusive on both ends, and the gap between them is a
    # deadband: when the decoder is unsure, do nothing rather than twitch.
    if p_hands >= high:
        return speed
    if p_hands <= low:
        return -speed
    return 0.0

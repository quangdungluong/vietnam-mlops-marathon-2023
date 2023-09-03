from scipy import stats
import numpy as np
import pandas as pd

def _wasserstein_distance_norm(
    reference_data: pd.Series, current_data: pd.Series, threshold: float
):
    """Compute the first Wasserstein distance between two arrays normed by std of reference data
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: all values above this threshold means data drift
    Returns:
        wasserstein_distance_norm: normed Wasserstein distance
        test_result: whether the drift is detected
    """
    norm = max(np.std(reference_data), 0.001)
    wd_norm_value = stats.wasserstein_distance(reference_data, current_data) / norm
    return wd_norm_value, wd_norm_value >= threshold

def _ks_stat_test(
    reference_data: pd.Series, current_data: pd.Series, threshold_down: float, threshold_up: float
):
    """Run the two-sample Kolmogorov-Smirnov test of two samples. Alternative: two-sided
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: level of significance
    Returns:
        p_value: two-tailed p-value
        test_result: whether the drift is detected
    """
    p_value = stats.ks_2samp(reference_data, current_data)[1]
    return p_value, threshold_down <= p_value <= threshold_up

def get_unique_not_nan_values_list_from_series(current_data: pd.Series, reference_data: pd.Series) -> list:
    """Get unique values from current and reference series, drop NaNs"""
    return list(set(reference_data.dropna().unique()) | set(current_data.dropna().unique()))

def get_binned_data(
    reference_data: pd.Series, current_data: pd.Series, feature_type: str, n: int, feel_zeroes: bool = True
):
    """Split variable into n buckets based on reference quantiles
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        n: number of quantiles
    Returns:
        reference_percents: % of records in each bucket for reference
        current_percents: % of records in each bucket for current
    """
    n_vals = reference_data.nunique()

    if feature_type == "num" and n_vals > 20:
        bins = np.histogram_bin_edges(pd.concat([reference_data, current_data], axis=0).values, bins="sturges")
        reference_percents = np.histogram(reference_data, bins)[0] / len(reference_data)
        current_percents = np.histogram(current_data, bins)[0] / len(current_data)

    else:
        keys = get_unique_not_nan_values_list_from_series(current_data=current_data, reference_data=reference_data)
        ref_feature_dict = {**dict.fromkeys(keys, 0), **dict(reference_data.value_counts())}
        current_feature_dict = {**dict.fromkeys(keys, 0), **dict(current_data.value_counts())}
        reference_percents = np.array([ref_feature_dict[key] / len(reference_data) for key in keys])
        current_percents = np.array([current_feature_dict[key] / len(current_data) for key in keys])

    if feel_zeroes:
        np.place(
            reference_percents,
            reference_percents == 0,
            min(reference_percents[reference_percents != 0]) / 10**6
            if min(reference_percents[reference_percents != 0]) <= 0.0001
            else 0.0001,
        )
        np.place(
            current_percents,
            current_percents == 0,
            min(current_percents[current_percents != 0]) / 10**6
            if min(current_percents[current_percents != 0]) <= 0.0001
            else 0.0001,
        )

    return reference_percents, current_percents

def _psi(
    reference_data: pd.Series, current_data: pd.Series, feature_type: str, threshold: float, n_bins: int = 30
):
    """Calculate the PSI
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        threshold: all values above this threshold means data drift
        n_bins: number of bins
    Returns:
        psi_value: calculated PSI
        test_result: whether the drift is detected
    """
    reference_percents, current_percents = get_binned_data(reference_data, current_data, feature_type, n_bins)

    psi_values = (reference_percents - current_percents) * np.log(reference_percents / current_percents)
    psi_value = np.sum(psi_values)

    return psi_value, psi_value >= threshold

def detect_drift_prob1(df, test_df):
    feature = "feature24"
    return _psi(df[feature], test_df[feature], "num", 0.016)[1]

def detect_drift_prob2(df, test_df):
    feature = "feature20"
    return _psi(df[feature], test_df[feature], "num", 0.026)[1]
"""DP-Transformer module.

This module contains a DataTransformer with a privacy-friendlier option for
continuous columns, intended for use with DP-CTGAN and DP-TVAE.
"""

import numpy as np

from ctgan.data_transformer import ColumnTransformInfo, DataTransformer, SpanInfo


class DPDataTransformer(DataTransformer):
    """Data Transformer with a privacy-friendlier option for continuous columns.

    This class inherits from the original ``DataTransformer`` and overrides the
    continuous column transformation to use simple linear binning instead of a
    Variational Gaussian Mixture Model (VGM). This is useful because the VGM
    learns a detailed distribution of the data, which can be a source of privacy
    leakage.
    """

    def __init__(self, n_bins=10):
        """Create a DP-friendly data transformer.

        Args:
            n_bins (int):
                Number of bins to use for linear binning of continuous columns.
        """
        self._n_bins = n_bins

    def _fit_continuous(self, data):
        """Fit a linear binner for a continuous column."""
        # NOTE: This method leaks the true min and max of the column, which is a
        # privacy violation. In a truly DP setting, these would need to be
        # computed with a DP mechanism (e.g., using the Laplace mechanism) or
        # be pre-defined based on public knowledge of the data domain.
        column_name = data.columns[0]
        min_val = data[column_name].min()
        max_val = data[column_name].max()
        transform = {'min': min_val, 'max': max_val, 'bins': self._n_bins}

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=transform,
            output_info=[SpanInfo(self._n_bins, 'softmax')], output_dimensions=self._n_bins
        )

    def _transform_continuous(self, column_transform_info, data):
        """Transform continuous values into one-hot encoded bins."""
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        min_val = column_transform_info.transform['min']
        max_val = column_transform_info.transform['max']
        n_bins = column_transform_info.transform['bins']
        bins = np.linspace(min_val, max_val, n_bins)
        binned_data = np.digitize(flattened_column, bins)
        output = np.zeros((len(flattened_column), n_bins))
        output[np.arange(len(flattened_column)), binned_data - 1] = 1.0
        return output

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        """Inverse transform one-hot encoded bins back to continuous values."""
        min_val = column_transform_info.transform['min']
        max_val = column_transform_info.transform['max']
        n_bins = column_transform_info.transform['bins']
        binned_data = np.argmax(column_data, axis=1)
        bin_width = (max_val - min_val) / n_bins
        bin_starts = min_val + binned_data * bin_width
        return bin_starts + np.random.uniform(0, bin_width, size=len(binned_data))


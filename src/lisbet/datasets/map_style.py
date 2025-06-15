"""
Map-style dataset for extracting windows of frames from records.
"""

from torch.utils.data import Dataset

from lisbet.datasets.common import AnnotatedWindowSelector, WindowSelector


class WindowDataset(Dataset):
    """
    Map-style dataset for extracting windows of frames from records.

    This dataset generates windows of frames from a collection of records. It is
    intended for inference (no labels, ordered windows) or as a base class for tasks
    requiring labeled windows (e.g., classification, regression). Windows can be
    centered, causal, or anticausal with respect to the reference frame, depending on
    the window_offset parameter. Padding and interpolation are applied as needed.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
    ):
        """
        Initialize a WindowDataset instance.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        """
        super().__init__()

        self.window_selector = WindowSelector(
            records, window_size, window_offset, fps_scaling
        )

        self.transform = transform

    def __len__(self):
        """
        Returns the total number of available windows in the dataset.

        Returns
        -------
        int
            Number of windows (frames) in the dataset.
        """
        return self.window_selector.n_frames

    def __getitem__(self, global_idx):
        """
        Retrieve a window of frames corresponding to the given global index.

        Parameters
        ----------
        global_idx : int
            Global index of the window to retrieve.

        Returns
        -------
        x : xarray.Dataset or torch.Tensor
            The window of frames, possibly transformed.
        """
        # Map global index to (record_index, frame_index)
        rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

        # Extract corresponding window
        x = self.window_selector.select(rec_idx, frame_idx)

        if self.transform:
            x = self.transform(x)

        return x


class AnnotatedWindowDataset(Dataset):
    """
    Map-style dataset for extracting labeled windows of frames from records.

    This dataset generates labeled windows of frames from a collection of records,
    suitable for evaluation or supervised learning tasks. It supports different label
    formats, including binary, multiclass, and multilabel, for various classification
    tasks. Labels are extracted according to the specified annotation format.
    """

    def __init__(
        self,
        records,
        window_size,
        window_offset=0,
        fps_scaling=1.0,
        transform=None,
        annot_format="multiclass",
    ):
        """
        Initialize an AnnotatedWindowDataset instance.

        Parameters
        ----------
        records : list
            List of records containing the data.
        window_size : int
            Size of the window in frames.
        window_offset : int, optional
            Offset for the window in frames (default is 0).
        fps_scaling : float, optional
            Scaling factor for the frames per second (default is 1.0).
        transform : callable, optional
            A function/transform to apply to the data (default is None).
        annot_format : str, optional
            Format of the labels. Valid options are 'binary', 'multiclass', or
            'multilabel' for the respective classification tasks (default is
            'multiclass').
        """
        super().__init__()

        self.window_selector = AnnotatedWindowSelector(
            records, window_size, window_offset, fps_scaling, annot_format
        )

        self.transform = transform

    def __len__(self):
        """
        Returns the total number of available labeled windows in the dataset.

        Returns
        -------
        int
            Number of labeled windows (frames) in the dataset.
        """
        return self.window_selector.n_frames

    def __getitem__(self, global_idx):
        """
        Retrieve a window of frames and its label corresponding to the given global
        index.

        Parameters
        ----------
        global_idx : int
            Global index of the window to retrieve.

        Returns
        -------
        x : xarray.Dataset or torch.Tensor
            The window of frames, possibly transformed.
        y : int, np.ndarray, or torch.Tensor
            The label(s) for the window, format depends on annot_format.
        """
        # Map global index to (record_index, frame_index)
        rec_idx, frame_idx = self.window_selector.global_to_local(global_idx)

        # Extract corresponding window
        x, y = self.window_selector.select(rec_idx, frame_idx)

        if self.transform:
            x = self.transform(x)

        return x, y

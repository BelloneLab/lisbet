"""An HDF5 Archive dataset."""

import logging
from collections import defaultdict
from pathlib import Path

import h5py
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def load(filename, seed=None, test_ratio=None):
    """Load dataset from an HDF5 file."""
    records = defaultdict(dict)

    def load_dataset(name, node):
        if isinstance(node, h5py.Dataset):
            record_name = str(Path(name).parent)
            dataset_name = str(Path(name).name)

            records[record_name][dataset_name] = node[()]

    with h5py.File(filename, "r") as h5file:
        h5file.visititems(load_dataset)

    # Convert to the expected data structure
    records = list(tuple(records.items()))

    if test_ratio is not None:
        # Split safety check
        assert test_ratio < 1

        # Split sets randomly
        rec_train, rec_test = train_test_split(
            records, test_size=test_ratio, random_state=seed
        )

        logging.info("Test set size = %d videos", len(rec_test))
        logging.debug([key for key, val in rec_test])
    else:
        rec_train = records
        rec_test = None

    logging.info("Training set size =  %d videos", len(rec_train))
    logging.debug([key for key, val in rec_train])

    return rec_train, rec_test


def dump(filename, records):
    """Dump list of records to an HDF5 file."""
    with h5py.File(filename, "w") as h5file:
        for grp_key, grp_data in tqdm(records, desc="Dumping to H5"):
            grp = h5file.create_group(grp_key)
            for ds_name, ds_data in grp_data.items():
                grp.create_dataset(ds_name, data=ds_data, compression="gzip")

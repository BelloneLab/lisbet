"""Preprocessing module for loading and splitting records for multiple tasks."""

import logging
import re
from collections import defaultdict
from itertools import repeat

from sklearn.model_selection import train_test_split

from lisbet.datasets import load_records


def load_multi_records(
    data_format,
    data_path,
    data_scale,
    data_filter,
    select_coords,
    rename_coords,
):
    """Internal helper. Loads and splits records for all tasks."""
    datasets = data_format.split(",")
    datapaths = data_path.split(",")
    if len(datasets) == len(datapaths):
        datasources = list(zip(datasets, datapaths))
    elif len(datapaths) == 1:
        datasources = list(zip(datasets, repeat(datapaths[0])))
    else:
        raise ValueError(
            "Input arguments datasets and datapaths must have the same length, or"
            " datapath must be a single element."
        )
    logging.debug(datasources)

    # Load records
    multi_records = [
        load_records(
            dataset,
            datapath,
            data_scale=data_scale,
            data_filter=data_filter,
            select_coords=select_coords,
            rename_coords=rename_coords,
        )
        for dataset, datapath in datasources
    ]

    # Sanity check: All posetracks must have the same 'features' coordinate across
    #               datasets. As consistency within a dataset is already checked, we
    #               only need to check the first record of each dataset against the
    #               others.
    main_features = [
        recs[0][1]["posetracks"].coords["features"].values.tolist()
        for recs in multi_records
    ]
    ref_features = main_features[0]
    for i, features in enumerate(main_features):
        if features != ref_features:
            raise ValueError(
                f"Inconsistent posetracks coordinates in loaded records, dataset {i}:\n"
                f"Reference features:\n{ref_features}\n"
                f"Record features:\n{features}"
            )

    return multi_records


def split_multi_records(
    multi_records,
    dev_ratio,
    dev_seed,
    task_ids,
    task_data,
):
    """Split records into train and dev sets."""
    # Build task to data mapping, by default use all data for every task
    task_data_map = {task_id: list(range(len(multi_records))) for task_id in task_ids}

    # Update task to data mapping, if requested
    if task_data is not None:
        logging.debug("Updating task to data mapping")
        pattern = r"(\b(?:" + r"|".join(task_ids) + r")\b):(\[(.*?)\])"
        matches = re.findall(pattern, task_data)
        task_data_map |= {
            key: [int(x) for x in vals.split(",")] for key, _, vals in matches
        }
    logging.debug(task_data_map)

    # Create the lists of records for each task
    train_rec = defaultdict(list)
    dev_rec = defaultdict(list)

    # Assign records
    for task_id, dataidx_lst in task_data_map.items():
        for dataidx in dataidx_lst:
            # Locate records for the current task
            records = multi_records[dataidx]

            # Split records
            if dev_ratio is not None:
                train_rec_task, dev_rec_task = train_test_split(
                    records,
                    test_size=dev_ratio,
                    random_state=dev_seed,
                )

                # Assign records to train and dev sets
                train_rec[task_id].extend(train_rec_task)
                dev_rec[task_id].extend(dev_rec_task)

            else:
                # Assign all records to train sets
                train_rec[task_id].extend(records)

            logging.info(
                "Assigning records from dataset no. %d to task %s", dataidx, task_id
            )

        logging.info("Final training set size = %d", len(train_rec[task_id]))
        logging.debug([key for key, _ in train_rec[task_id]])

        logging.info("Final dev set size = %d", len(dev_rec[task_id]))
        logging.debug([key for key, _ in dev_rec[task_id]])

    return train_rec, dev_rec

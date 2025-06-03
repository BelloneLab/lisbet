"""Preprocessing functions for model training."""

import logging
import re
from collections import defaultdict

from sklearn.model_selection import train_test_split


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
        logging.debug([rec.id for rec in train_rec[task_id]])

        logging.info("Final dev set size = %d", len(dev_rec[task_id]))
        logging.debug([rec.id for rec in dev_rec[task_id]])

    return train_rec, dev_rec

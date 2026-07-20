.. _fine-tuning:

Fine tuning a classification model on a custom dataset
======================================================

After pre-training the LISBET encoder on a large unlabeled dataset (see :ref:`model-training`), the model can be fine-tuned to reproduce the annotation style and preferences of the user using a smaller labeled dataset.
We demonstrate this process on the CalMS21 dataset - Task 1 (Sun et al., 2021).
This dataset contains key points tracking for 70 training videos and 19 testing videos of mice pairs in free interaction, annotated with 4 classes: *attack*, *investigation*, *mount*, and *other*.

Step 1: Load the dataset
------------------------

The CalMS21 dataset - Task 1 can be loaded using the ``betman fetch_dataset`` command as follows.

.. code-block:: bash

   betman fetch_dataset CalMS21_Task1

The dataset is stored in the ``datasets/CalMS21/task1_classic_classification`` directory.

CalMS21 Task 2 data
-------------------

CalMS21 Task 2 contains the same four behavior classes as Task 1, annotated in
the individual styles of five additional annotators. Download and convert it with:

.. code-block:: bash

   betman fetch_dataset CalMS21_Task2

The converted data is stored under
``datasets/CalMS21/task2_annotation_styles``. Each video remains an independent
movement record, organized using the hierarchy provided by CalMS21:

.. code-block:: text

   task2/annotator1/train/<video-id>/
   task2/annotator1/test/<video-id>/
   ...
   task2/annotator5/train/<video-id>/
   task2/annotator5/test/<video-id>/

Each leaf directory contains ``tracking.nc`` and ``annotations.nc``. The hierarchy
also enables selection through ``--data_filter``, for example
``--data_filter=annotator1/train``.

The source archive is approximately 912 MB and the two extracted keypoint JSON
files require several gigabytes of disk space. Conversion uses the standard Python
JSON loader, so its peak memory use can be substantially larger than the compressed
download. The command prepares the dataset in LISBET's standard format; it does not
train an annotator-conditioned model or implement the Task 2 benchmark protocol.

Step 2: Fine-tune the model
---------------------------

Fine-tuning the model on the CalMS21 dataset - Task 1 can be done using the ``betman train_model`` command as follows:

.. code-block:: bash

    betman train_model \
        -v \
        --data_format=movement \
        --data_filter=train \
        --run_id=lisbet32x4-calms21UftT1 \
        --seed=42 \
        --learning_rate=1e-6 \
        --epochs=15 \
        --emb_dim=32 \
        --num_layers=4 \
        --num_heads=4 \
        --hidden_dim=128 \
        --load_backbone_weights=models/lisbet32x4-calms21U/weights/weights_last.pt \
        --window_offset=99 \
        --save_history \
        datasets/CalMS21/task1_classic_classification

References
----------
Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

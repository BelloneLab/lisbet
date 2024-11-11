.. _fine-tuning:

Fine tuning a classification model on a custom dataset
======================================================

After pre-training the LISBET encoder on a large unlabeled dataset (see :ref:`model-training`), the model can be fine-tuned to reproduce the annotation style and prefereces of the user using a smaller labeled dataset. We demonstrate this process on the CalMS21 dataset - Task 1 (Sun et al., 2021). This dataset contains key points tracking for 70 training videos and 19 testing videos of mice pairs in free interaction, annotated with 4 classes: *attack*, *investigation*, *mount*, and *other*.

Step 1: Load the dataset
------------------------

The CalMS21 dataset - Task 1 can be loaded using the ``betman fetch_dataset`` command as follows.

.. code-block:: bash

   betman fetch_dataset CalMS21_Task1

The dataset is stored in the ``datasets/CalMS21/task1_classic_classification`` directory.

Step 2: Fine-tune the model
---------------------------

Fine-tuning the model on the CalMS21 dataset - Task 1 can be done using the ``betman train_model`` command as follows.

.. code-block:: bash

    betman train_model \
        -v \
        --data_format=h5archive \
        --run_id=lisbet32x8-calms21UftT1 \
        --seed=42 \
        --learning_rate=1e-6 \
        --epochs=15 \
        --emb_dim=32 \
        --num_layers=8 \
        --num_heads=8 \
        --hidden_dim=128 \
        --load_backbone_weights=models/lisbet32x8-calms21U/weights/weights_last.pt \
        --window_offset=99 \
        --save_history \
        datasets/CalMS21/task1_classic_classification/train_records.h5

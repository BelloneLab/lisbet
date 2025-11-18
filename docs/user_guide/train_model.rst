.. _model-training:

Model training using self-supervised learning
=============================================

Training a model from scratch is a time-consuming process and generally not required to work with LISBET.
However, if your tracking data does not follow the conventions of any of the provided models, you want to use a different data source, or simply you want to experiment, you can train a new model using the self-supervised learning approach.
This approach requires a large amount of tracking data and a GPU to train the model.
The training process is described in the following steps.

Step 1: Prepare the data
------------------------

Please check :ref:`data-preparation` to learn how to prepare your the data for training.
Alternatively, you can use ``betman fetch_dataset`` to download any of the available datasets directly from the command line.
Here we are going to use the latter method to download the CalMS21 dataset (Sun et al., 2021).
This dataset contains a large corpus of tracking data of mice pairs in free social interaction ("Unlabeled" group).

.. code-block:: console

    $ betman fetch_dataset CalMS21_Unlabeled

This command will download the dataset to the datasets directory in the current working directory.
The dataset is preprocessed for training and stored in the `movement` format, which is a binary format that can store large amounts of data efficiently (currently NetCDF).

Step 2: Train the model
-----------------------

To train a model, you can use the ``betman train_model`` command.
The following command trains a model using the CalMS21 dataset with all the available self-supervised tasks.

.. note::
    For a detailed explanation of the self-supervised tasks available in LISBET, see :doc:`self_supervised_tasks`.

The model is a transformer model with 8 layers, 8 heads, and a hidden dimension of 256.
The embedding dimension is 64.
The training process is limited to 100 epochs, randomly sampling 5% of the data at the beginning of each epoch.
The training history is saved to a file for later review.

.. code-block:: console

    $ betman train_model \
        -v \
        --task_ids=cons,order,shift,warp \  # Use all the available self-supervised tasks
        --data_format=movement \
        --run_id=lisbet64x8-calms21U \
        --seed=1234 \
        --epochs=100 \
        --emb_dim=64 \
        --num_layers=8 \
        --num_heads=8 \
        --hidden_dim=256 \
        --train_sample=0.05 \
        --save_history \
        datasets/CalMS21/unlabeled_videos

Step 3: Use data augmentation (optional)
-----------------------------------------

Data augmentation can improve model robustness and generalization by introducing variations during training.
LISBET supports several augmentation techniques that can be combined and applied with configurable probabilities.

Available augmentation techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **all_perm_id**: Randomly permute individual identities across all frames in a window
    - Use this to make the model invariant to individual labels (e.g., "mouse1" vs "mouse2")
    - Particularly useful for self-supervised tasks where identity labels are arbitrary

* **all_perm_ax**: Randomly permute spatial axes (x, y, z) across all frames in a window
    - Use this to make the model invariant to coordinate system orientation
    - **⚠️ Important**: Only suitable for top-down view datasets (typical laboratory setups)
    - **Not recommended** for front-view, side-view, or 3D datasets where axes have semantic meaning

* **blk_perm_id**: Randomly permute individual identities within a contiguous block of frames
    - Creates temporal identity confusion within part of the window
    - More challenging augmentation than ``all_perm_id``
    - Requires ``frac`` parameter to specify the fraction of frames to permute

Usage examples
~~~~~~~~~~~~~~

.. note::
    Use the ``--data_augmentation=`` format (with equals sign) to clearly separate the argument from its value. While this makes quoting optional in most shells, it's still recommended for consistency and to prevent any potential shell interpretation of special characters (commas, colons).

**Basic usage with default probability (1.0):**

.. code-block:: console

    $ betman train_model \
        --data_augmentation="all_perm_id" \
        ... # other parameters

**With custom probability:**

.. code-block:: console

    $ betman train_model \
        --data_augmentation="all_perm_id:p=0.5" \
        ... # other parameters

This applies identity permutation to 50% of training samples.

**Multiple augmentations:**

.. code-block:: console

    $ betman train_model \
        --data_augmentation="all_perm_id:p=0.5,all_perm_ax:p=0.7" \
        ... # other parameters

**Block permutation with fraction:**

.. code-block:: console

    $ betman train_model \
        --data_augmentation="blk_perm_id:p=0.3:frac=0.2" \
        ... # other parameters

This applies identity permutation to a random 20% block of frames, with 30% probability.

**Combined augmentations for top-down view datasets:**

.. code-block:: console

    $ betman train_model \
        -v \
        --task_ids=cons,order,shift,warp \
        --data_augmentation="all_perm_id:p=0.5,all_perm_ax:p=0.7,blk_perm_id:p=0.3:frac=0.2" \
        --data_format=movement \
        --run_id=lisbet64x8-calms21U-aug \
        --seed=1234 \
        --epochs=100 \
        --emb_dim=64 \
        --num_layers=8 \
        --num_heads=8 \
        --hidden_dim=256 \
        --train_sample=0.05 \
        --save_history \
        datasets/CalMS21/unlabeled_videos

Important considerations
~~~~~~~~~~~~~~~~~~~~~~~~

* **View-dependent augmentations**: The ``all_perm_ax`` augmentation assumes symmetry across spatial axes and should only be used for top-down view datasets common in laboratory mouse experiments. For human datasets or non-overhead camera angles, this augmentation may hurt performance as axes have semantic meaning (e.g., up/down gravity, left/right lateral).

* **Task compatibility**: Identity permutations (``all_perm_id``, ``blk_perm_id``) are most beneficial for self-supervised tasks and datasets where individual identities are interchangeable.

* **Probability tuning**: Start with moderate probabilities (0.3-0.7) and adjust based on validation performance. Higher probabilities increase variability but may make training less stable.

* **Computational cost**: Augmentations are applied on-the-fly during training and add minimal overhead. Block permutations (``blk_perm_id``) are slightly more expensive than full permutations.

The training process can take a long time depending on the size of the dataset and the complexity of the model.
For reference, running the command above required approximatively 1h15 per epoch on a MacBook Pro (Apple M1 Pro), or approximatively 12 minutes on a Linux compute node with GPU (AMD EPYC-7742, NVIDIA RTX A5500).
The training process can be monitored using the ``-v`` flag.
The model configuration, weights and training history are saved in the ``models`` directory in the current working directory, under the given ``run_id`` (i.e., ``lisbet64x8-calms21U`` in this case).

[OPTIONAL] Step 4: Export embedding model
-----------------------------------------

The trained model can be exported to a standalone embedding model using the following command.

.. code-block:: console

    $ betman export_embedder \
        models/lisbet64x8-calms21U/model_config.yml \
        models/lisbet64x8-calms21U/weights/weights_last.pt

Unless otherwise specified using the --output_path option, the embedding model is saved in the ``models directory`` in the current working directory, under models using the same run_id of the original model with the "-embedder" suffix (i.e., ``lisbet64x8-calms21U-embedder`` in this case).

References
----------

Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

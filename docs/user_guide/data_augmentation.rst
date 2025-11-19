.. _data_augmentation:

Data augmentation
=================

Data augmentation can improve model robustness and generalization by introducing variations during training.
LISBET supports several augmentation techniques that can be combined and applied with configurable probabilities.

Available augmentation techniques
---------------------------------

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
--------------

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

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
The dataset is preprocessed for training and stored in the hdf5 format, which is a binary format that can store large amounts of data efficiently.

Step 2: Train the model
-----------------------

To train a model, you can use the ``betman train_model`` command.
The following command trains a model using the CalMS21 dataset with all the available self-supervised tasks.
The model is a transformer model with 8 layers, 8 heads, and a hidden dimension of 256.
The embedding dimension is 64.
The training process is limited to 100 epochs, randomly sampling 5% of the data at the beginning of each epoch.
The training history is saved to a file for later review.

.. code-block:: console

   $ betman train_model \
      -v \
      --task_ids=nwp,smp,vsp,dmp \  # Use all the available self-supervised tasks
      --data_format=h5archive \
      --run_id=lisbet64x8-calms21U \
      --seed=1234 \
      --epochs=100 \
      --emb_dim=64 \
      --num_layers=8 \
      --num_heads=8 \
      --hidden_dim=256 \
      --train_sample=0.05 \
      --save_history \
      datasets/CalMS21/unlabeled_videos/all_records.h5

The training process can take a long time depending on the size of the dataset and the complexity of the model.
For reference, running the command above required approximatively 1h15 per epoch on a MacBook Pro (Apple M1 Pro), or approximatively 12 minutes on a Linux compute node with GPU (AMD EPYC-7742, NVIDIA RTX A5500).
The training process can be monitored using the ``-v`` flag.
The model configuration, weights and training history are saved in the ``models`` directory in the current working directory, under the given ``run_id`` (i.e., ``lisbet64x8-calms21U`` in this case).

[OPTIONAL] Step 3: Export embedding model
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

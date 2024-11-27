.. _social-behavior-discovery:

Social behavior discovery using a pre-trained model
===================================================

One of the most common applications of LISBET is using a pre-trained model to automatically discover social behaviors in your dataset.

Requirements: - A pre-trained LISBET embedding model - A keypoints dataset in the standard LISBET format

Step 0: Prepare the data and model
----------------------------------
Please check :ref:`data-preparation` to learn how to prepare your key point data for embedding.
Alternatively, you can use ``betman fetch_dataset`` to download any of the available datasets directly from the command line.
Similarly, you can look at :ref:`model-training` to fit a new model, or use ``betman fetch_model`` to download any of the available pre-trained models.

Step 1: Embedding
-----------------

The embedding step transforms the input keypoints into a set features related to social behavior.
For each input sequence (i.e., keypoints extracted from a given recording) of shape (nframes, nbodyparts), LISBET generates a CSV embedding file of shape (nframes, nfeatures).

To generate the embedding files for your dataset you can use **betman**, a command line tool designed to interact with LISBET:

.. code-block:: console

   $ betman compute_embeddings \
      --data_format=DATAFORMAT \      # Format used by your keypoints dataset
      --window_size=WINDOW_SIZE \     # Number of frames in the past to consider
      --fps_scaling=FPS_SCALING \     # Your dataset FPS / model training FPS
      -v \                            # Enable verbose mode
      DATA_PATH \                     # Path to the keypoints dataset
      MODEL_PATH \                    # Path to pre-trained embedding model config
      MODEL_WEIGHTS                   # Path to pre-trained embedding model weights

where DATAFORMAT indicates the format used to store the keypoints (e.g., **GenericDLC** if your dataset was obtained using DeepLabCut by Mathis et al. 2018), WINDOW_SIZE is the number of past frames the model is allowed to use to compute the embedding of each frame, and FPS_SCALING is the ratio between the frame rate (FPS) of your dataset and the one used by the pre-trained model (e.g., **0.833** if your dataset was acquired at 25 FPS and you are using a model pre-trained on the 30 FPS CalMS21 dataset by Sun et al. 2021).

For example, your ``betman predict`` command might look something like this:

.. code-block:: console

   $ betman compute_embeddings \
      --data_format=maDLC \
      --window_size=200 \
      --fps_scaling=0.833 \
      -v \
      datasets/MyDataset \
      models/lisbet32x8-calms21U-embedder/model_config.yml \
      models/lisbet32x8-calms21U-embedder/weights/weights_last.pt

Please use ``betman compute_embeddings --help`` for a list of all available option.

After runninng betman, you should find the embeddings for your dataset in the OUTPUT_PATH directory and be ready to proceed with **Step 2: HMM fitting**.

Step 2: HMM fitting
-------------------

The HMM fitting step transforms the embeddings into behavioral motifs (i.e., labels), using a Gaussian Hidden Markov Model (HMM).
For each embedding sequence of shape (nframes, nfeatures), LISBET generates a CSV annotation file of shape (nframes, nmotifs).

To generate the embedding files for your dataset you can again use \**betman\*:

.. code-block:: console

   $ betman segment_motifs \
      --output_path=OUTPUT_PATH \       # Path to store annotation files (i.e., labels)
      --num_states=N_MOTIFS \           # Desired number of motifs (i.e., HMM states)
      --num_iter=N_ITER \               # Max number of steps for the fitting algorithm
      -v \                              # Enable verbose mode
      EMBEDDING_PATH                    # Path to LISBET embeddings

where EMBEDDING_PATH is the location of the LISBET embeddings obtained in Step 1, N_MOTIFS is the desired number of behavioral motifs to identify in the data (corresponding to the number of hidden states in the HMM), and N_ITER is the maximum number of iterations allowed before stopping the HMM fitting algorithm before convergence.

For example, your ``betman segment_motifs`` command might look something like this:

.. code-block:: console

   $ betman segment_motifs \
      --output_path=hmm_predictions \
      --num_states=8 \
      --num_iter=500 \
      -v \
      bet_predictions/maDLC

Please use ``betman segment_motifs --help`` for a list of all available option.

After running unsupman, you should find the annotations (i.e., labels) for your dataset in the OUTPUT_PATH directory.

As discussed in Chindemi et al. 2023, choosing the number of behaviors N_MOTIFS a priori is a limitation imposed by most clustering algorithms.
If you want to avoid doing so, we propose a "scan-and-select" procedure which allows to specify an upper limit to the number of behaviors in your dataset, rather than the exact number, and automatically determine the actual number of behaviors in the dataset.
This procedure is described below in **Step 3: Prototype selection**.
Before proceeding with Step 3, you need to generate multiple sets of HMM annotations by running unsupman, each time with a different N_MOTIFS, as described above.
In case you are using a SLURM cluster, this can be easily done by running ``betman segment_motifs`` in a JOB ARRAY.

[OPTIONAL] Step 3: Prototype selection
--------------------------------------

The prototype selection step transforms multiple sets of behavioral motifs into a single one, by clustering similar motifs and selecting one of them as a prototype representing the whole group.

For each set of motifs of shape nsets x (nframes, nmotifs), LISBET generates a CSV annotation file of shape (nframes, nprototypes), where nprototypes is automatically computed to maximixe a clustering metric (i.e., the silhouette score).

To generate the embedding files for your dataset you can use **betman**:

.. code-block:: console

   $ betman select_prototypes \
      --hmm_range LOW HIGH \       # Smallest and largest annotation set to consider
      --output_path=OUTPUT_PATH \  # Path to store annotation files (i.e., labels)
      --method=METHOD \            # Prototype selection method
      -v \                         # Enable verbose mode
      ANNOT_PATH                   # Path to the root of the annotation sets

where ANNOT_PATH is the location of the LISBET annotations obtained in Step 2, MIN_STATES (MAX_STATES) is the smallest (largest) annotation set to consider (corresponding to the number of states in the HMM models), and METHOD determines how the prototype for a motif group is chosen (i.e., **best** will select the prototype with the highest silhouette coefficient).

For example, your ``unsupman select_prototypes`` command might look something like this:

.. code-block:: console

   $ betman select_prototypes \
      --hmm_range 6 32 \
      --output_path=proto_predictions \
      --method=best \
      -v \
      hmm_predictions/maDLC

Please use ``betman select_prototypes --help`` for a list of all available option.

After running unsupman, you should find the annotations (i.e., labels) for your dataset in the OUTPUT_PATH directory.

References
----------

Mathis, A., Mamidanna, P., Cury, K. M., Abe, T., Murthy, V. N., Mathis, M. W., & Bethge, M. (2018).
DeepLabCut: Markerless pose estimation of user-defined body parts with deep learning.
Nature Neuroscience, 21(9), Article 9.
https://doi.org/10.1038/s41593-018-0209-y

Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

Chindemi, G., Girard, B., & Bellone, C. (2023). LISBET: a machine learning model for the automatic segmentation of social behavior motifs (arXiv:2311.04069).
arXiv.
https://doi.org/10.48550/arXiv.2311.04069

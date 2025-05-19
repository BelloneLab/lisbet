.. _social-behavior-classification:

Social behavior classification using a pre-trained model
========================================================

LISBET enables automated annotation of social behaviors in animal datasets using pre-trained or custom-trained classification models.
This workflow is ideal when you have a set of human-defined behaviors and annotated examples, and you want to automate the scoring of large datasets with high consistency and speed.

Requirements:

- A pre-trained LISBET classification model (or a model you have fine-tuned)
- A keypoints dataset in a supported format (e.g., DeepLabCut, SLEAP, or movement)

Step 0: Prepare the data and model
----------------------------------
Ensure your keypoint data is organized according to LISBET’s requirements (see :ref:`data-preparation`).
You can use your own data or fetch a public dataset, for example CalMS21 (Sun et al. 2021), using:

.. code-block:: console

   $ betman fetch_dataset CalMS21_Task1

To use a pre-trained model, download one from the LISBET model zoo:

.. code-block:: console

   $ betman fetch_model lisbet32x4-calms21UftT1-classifier

Step 1: Annotate behaviors
--------------------------
Run the LISBET classifier on your dataset to generate behavior annotations.
The main command is:

.. code-block:: console

   $ betman annotate_behavior \
      --data_format=DATAFORMAT \      # Format of your keypoints dataset
      --fps_scaling=FPS_SCALING \     # Your dataset FPS / model training FPS
      --data_scale=DATA_SCALE \       # Original video frames dimensions in pixels
      -v \                            # Enable verbose mode
      DATA_PATH \                     # Path to the keypoints dataset
      MODEL_PATH \                    # Path to model config (YAML)
      MODEL_WEIGHTS                   # Path to model weights (.pt)

where DATAFORMAT indicates the format used to store the keypoints, FPS_SCALING is the ratio between the frame rate (FPS) of your dataset and the one used by the pre-trained model (e.g., **0.833** if your dataset was acquired at 25 FPS and you are using a model pre-trained on the 30 FPS CalMS21 dataset by Sun et al. 2021), and DATA_SCALE is the original video frames dimensions in pixels (e.g., **1024x570** for CalMS21).
If `--data_scale` is not provided, LISBET will scale the data in the required `(0, 1)` range by computing the min and max values of the keypoints in the dataset.

For example:

.. code-block:: console

   $ betman annotate_behavior \
      --data_format=movement \
      --data_scale="1024x570" \
      -v \
      datasets/CalMS21/task1_classic_classification \
      models/lisbet32x4-calms21UftT1-classifier/model_config.yml \
      models/lisbet32x4-calms21UftT1-classifier/weights/weights_last.pt

The output will be CSV files with predicted behavior labels for each frame, saved in the specified output directory.

Example: Adapting Keypoints from a New Dataset
----------------------------------------------

LISBET models require the input keypoints to match exactly (in both order and naming) the configuration used during model training.
If your dataset uses different keypoint names, contains extra keypoints, or uses a different order, you can adapt it using the `--select_coords` and `--rename_coords` options.

**1. Inspect the Model’s Expected Keypoints**

Before running inference, check which keypoints and order the model expects:

.. code-block:: console

   $ betman model_info models/lisbet32x4-calms21U-embedder/model_config.yml

This will print the list of required keypoints (see the `input_features` field).

**2. Adapt Your Dataset**

Suppose your dataset has the following keypoints: `body, nose, earL, earR, neck, hipsL, hipsR, tail`, and the individuals are named `experimental` and `stimuli`.
This is the configuration used in the SampleData dataset provided with LISBET.
The model expects only `nose, left_ear, right_ear, neck, left_hip, right_hip, tail` for each individual, named `resident` and `intruder`.

You can adapt your dataset as follows:

.. code-block:: console

   $ betman compute_embeddings \
       --data_format=DLC \
       --fps_scaling=0.833 \
       --select_coords="*;*;nose,earL,earR,neck,hipsL,hipsR,tail" \
       --rename_coords="experimental:resident,stimuli:intruder;*;earL:left_ear,earR:right_ear,hipsL:left_hip,hipsR:right_hip" \
       datasets/sample_keypoints \
       models/lisbet32x4-calms21U-embedder/model_config.yml \
       models/lisbet32x4-calms21U-embedder/weights/weights_last.pt

- `--select_coords` drops the extra keypoint `body` and ensures the order matches the model.
- `--rename_coords` maps your dataset’s individual and keypoint names to those expected by the model.

**3. What If You Include Extra Keypoints?**

If you try to include a keypoint that the model was not trained on (e.g., `body`), LISBET will raise an error:

.. code-block:: console

   $ betman compute_embeddings \
       --data_format=DLC \
       --fps_scaling=0.833 \
       --select_coords="*;*;body,nose,earL,earR,neck,hipsL,hipsR,tail" \
       --rename_coords="experimental:resident,stimuli:intruder;*;earL:left_ear,earR:right_ear,hipsL:left_hip,hipsR:right_hip" \
       datasets/sample_keypoints \
       models/lisbet32x4-calms21U-embedder/model_config.yml \
       models/lisbet32x4-calms21U-embedder/weights/weights_last.pt

This will fail with an error about incompatible input features, because the model does not expect the `body` keypoint.

**4. Summary**

- Always match the keypoints (names and order) to the model’s `input_features`.
- Use `--select_coords` to drop extra keypoints and reorder as needed.
- Use `--rename_coords` to map your dataset’s names to the model’s expected names.
- Use `betman model_info` to inspect the model’s required keypoints before running inference.

Advanced: Fine-tuning a classifier
----------------------------------
If you want to adapt LISBET to your own annotation style or behaviors, you can fine-tune a model using your labeled data. See :ref:`fine-tuning` for a step-by-step guide.

References
----------
Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

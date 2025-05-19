.. _prototype-inference:

Annotating new data using selected prototypes
=============================================

In the original design of the discovery-driven pipeline, prototype selection was intended as the final step following the HMM scan.
However, some use cases may require labeling a new dataset using previously selected prototypes, without re-running the HMM scan or re-analyzing all prototypes.

Re-running the HMM scan on new data, even when it includes the original dataset, typically results in a permutation of the prototype labels.
This permutation can, in principle, be corrected through automated label alignment, but the process would add complexity and slow down workflows.
More importantly, the inclusion of new data can introduce new prototypes if novel behavioral patterns are detected.
While this dynamic discovery is a valuable feature of the pipeline, it complicates analyses where the goal is specifically to track known prototypes across additional datasets, for example to study behavior consistency or investigate associated brain circuit dynamics.

To address this need, we have implemented two methods for annotating new data using the prototypes selected in the original dataset.
The first method is to train a LISBET classifier on the selected prototypes and then use it to label new data.
The second method uses cached HMMs to annotate new data, but it is not recommended for most users due to its complexity and potential safety issues.

It is worth noting that the cached HMMs method provides an exact match to the original prototype labels, whereas the the classifier approach only offers an approximation.
In our experience, the advantages of the classifier approach outweigh its drawbacks, as it is less error-prone and yields a reusable model that can be shared for future use.

Recommended Approach: Train a LISBET Classifier on Prototypes
-------------------------------------------------------------

The recommended and most robust way to annotate new data with previously selected prototypes is to train a LISBET classifier using the prototype labels as ground truth. This approach is simple, reproducible, and produces a reusable model.

Prepare a Labeled Dataset with Prototype Annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After running prototype selection, you will have CSV files with prototype labels for each sequence.
You need to convert these into a dataset format suitable for LISBET training (e.g., directory structure with `tracking` and `annotations`).

Example Python snippet to patch the CalMS21 dataset (Sun et al. 2021) with prototype labels, please adapt to your dataset:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xarray as xr

    from lisbet.datasets import dump_records, load_records


    def extract_labels(csv_path):
        df = pd.read_csv(csv_path, index_col=0)

        # Rows that already have at least one positive label
        covered = df.eq(1).any(axis=1)

        # Create / update the fallback class
        df["Other"] = (~covered).astype(int)

        # Keep only the first 1 in every row
        first_mask = df.eq(1).cumsum(axis=1).eq(1)

        # Apply the mask – everything that isn’t the first 1 becomes 0
        df &= first_mask

        return df.values


    def patch_dataset():
        records = load_records(
            data_format="movement",
            data_path="datasets/CalMS21/task1_classic_classification",
            data_scale="1024x570",
            data_filter="train",
        )["main_records"]

        patched_records = []
        for key, data in records:
            posetracks = data["posetracks"].unstack("features")

            labels = extract_labels(f"prototypes/{key}/machineAnnotation_hmmbest_6_32.csv")

            assert labels.shape[0] == posetracks.sizes["time"]

            # Convert to xarray Dataset
            annotations = xr.Dataset(
                data_vars=dict(
                    label=(
                        ["time", "behaviors", "annotators"],
                        labels[..., np.newaxis],
                    )
                ),
                coords=dict(
                    time=posetracks.time,
                    behaviors=[f"motif_{motif_id}" for motif_id in range(labels.shape[1])],
                    annotators=["LISBET"],
                ),
                attrs=dict(
                    source_software=posetracks.source_software,
                    ds_type="annotations",
                    fps=posetracks.fps,
                    time_unit=posetracks.time_unit,
                ),
            )

            patched_record = (
                key,
                {"posetracks": posetracks, "annotations": annotations},
            )

            patched_records.append(patched_record)

        dump_records("datasets/proto_CalMS21", patched_records)


    if __name__ == "__main__":
        patch_dataset()

This will create a new dataset with prototype labels as annotations.

Train a Classifier on the Prototype Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the LISBET CLI to train a classifier on your new labeled dataset. For example:

.. code-block:: console

    $ betman train_model \
        --run_id=proto_classifier \
        --data_format=movement \
        --data_scale="1x1" \
        --data_filter=train \
        --learning_rate=1e-4 \
        --epochs=10 \
        --load_backbone_weights=models/lisbet32x4-calms21U-embedder/weights/weights_last.pt \
        --freeze_backbone_weights \
        --save_history \
        -v \
        datasets/proto_CalMS21

- Use `--freeze_backbone_weights` to ensure the classifier matches the embedding model used for prototype discovery.
- Adjust `--data_format` and paths as needed for your dataset.

Annotate New Data Using the Trained Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once trained, use the classifier to annotate new datasets:

.. code-block:: console

    $ betman annotate_behavior \
        --data_format=movement \
        --data_scale="1024x570" \
        --data_filter=test \
        -v \
        datasets/CalMS21/task1_classic_classification \
        models/proto_classifier/model_config.yml \
        models/proto_classifier/weights/weights_last.pt

The output will be CSV files with predicted prototype labels for each frame.

.. note::
  - This approach provides an approximation of the original prototype labels
  - Overlapping prototypes are currently resolved by assigning the first label; multi-label support is planned.
  - Always ensure your new data matches the keypoint configuration expected by the model (see :ref:`data-preparation`).

Alternative: Using Cached HMMs
------------------------------

For advanced users, LISBET allows you to use cached HMM models to annotate new data. This method is not recommended for most users due to complexity and potential safety issues with loading pickle files.

If you wish to proceed:

1. Ensure you have the original HMM `.joblib` files saved from the initial scan.
2. Run:

   .. code-block:: console

      $ betman segment_motifs \
          --pretrained_path=PATH_TO_HMM_MODELS \
          --output_path=NEW_OUTPUT_PATH \
          datasets/NewDataset

   You can then extract the relevant prototype columns from the output annotation files.

.. warning::
   Loading pickle/joblib files can be unsafe if the source is untrusted. Only use this method with files you generated yourself, DO NOT LOAD PICKLE FILES FROM UNTRUSTED SOURCES.

References
----------

Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

Chindemi, G., Girard, B., & Bellone, C. (2023). LISBET: a machine learning model for the automatic segmentation of social behavior motifs (arXiv:2311.04069).
arXiv.
https://doi.org/10.48550/arXiv.2311.04069

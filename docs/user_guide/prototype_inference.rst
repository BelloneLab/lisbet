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

[RECOMMENDED] Method 1: Train a LISBET classifier using prototypes as labels
----------------------------------------------------------------------------------

In this guide we are going to use Task1 training records in the CalMS21 dataset (Sun et al. 2021) as an example, and suppose that you have already run the prototype selection process on it.
To train a LISBET classifier on the selected prototypes, you need to create a new dataset with the prototypes as labels.
This can easily be done using functions available in the LISBET API, for example:

.. code-block:: python

    """
    Patch the CalMS21 dataset (Sun et al. 2021) to use selected prototypes as labels.
    """
    from lisbet.datasets.h5archive import load, dump
    import pandas as pd


    def extract_labels(csv_path):
        df = pd.read_csv(csv_path, index_col=0)

        # Find indices of covered (i.e., labeled) rows
        covered = (df == 1).any(axis=1)

        # Create a dummy class "Other" for all rows not covered
        df["Other"] = 1
        df.loc[covered, "Other"] = 0

        # Make labels for multi-class classification
        labels = df.values.argmax(axis=1)

        return labels


    def patch_dataset():
        records, _ = load("datasets/CalMS21/task1_classic_classification/train_records.h5")

        patched_records = []
        for key, data in records:
            labels = extract_labels(f"prototypes/{key}/machineAnnotation_hmmbest_6_32.csv")

            assert (
                labels.shape[0] == data["keypoints"].shape[0]
            ), f"Labels length {labels.shape[0]} does not match keypoints length {data['keypoints'].shape[0]}"

            patched_record = (key, {"keypoints": data["keypoints"], "annotations": labels})

            patched_records.append(patched_record)

        dump("proto_train.h5", patched_records)


    if __name__ == "__main__":
        patch_dataset()

This script loads the original dataset, extracts the labels from the HMM annotations, and creates a new dataset with the prototypes as labels.

The new dataset is saved in the HDF5 format, which is compatible with LISBET and can be directly used for training.
To train the classifier, for example one based on the standard ``lisbet32x4-calms21U-embedder`` (see :ref:`social-behavior-discovery-step0`), you can use the ``betman train_model`` command as usual (see :ref:`model-training`):

.. code-block:: console

   $ betman train_model \
      --run_id=lisbet32x4-calms21UftProto-classifier \
      --data_format=h5archive \
      --learning_rate=1e-4 \
      --epochs=10 \
      --load_backbone_weights=models/lisbet32x4-calms21U-embedder/weights/weights_last.pt \
      --freeze_backbone_weights \
      --save_history \
      -v \
      proto_train.h5

Please note the use of ``--freeze_backbone_weights``, which is required to match as closely as possible the input of the original prototype selection process.

As mentioned above, the training approach produces an approximation of the original prototype labels.
In particular, compared to HMMs, classification models operate on regular windows rather than embedding stacks.
This slightly change the information available to the two models, and can potentially affect the performance of the classifier.
Moreover, overlapping prototype labels are currently not supported (multi-label classification) and resolved by assigning the label of the first prototype in the list.
Support for multi-label classification in LISBET is planned, though it is not expected to significantly impact results, as most prototypes overlap only briefly (typically for just a few frames in our experience).

Method 2: Using cached HMMs to annotate new data
------------------------------------------------

.. warning::
   This method is not recommended for most users, as it requires caching HMM fitting results as ``pickle`` files.
   It is primarily intended for local use by advanced users who are comfortable with the underlying assumptions of the prototype selection algorithm and its implementation in LISBET.

   Pickle serialization has known limitations, including potential safety issues (arbitrary code execution) and compatibility across different Python versions.

   DO NOT LOAD PICKLE FILES FROM UNTRUSTED SOURCES.

   For most users, we recommend using the fine-tuning method described above, as it provides a more straightforward and user-friendly approach to annotating new data.

PENDING: This section is not yet complete and requires further development.

References
----------

Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q., Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (arXiv:2104.02710).
arXiv.
https://doi.org/10.48550/arXiv.2104.02710

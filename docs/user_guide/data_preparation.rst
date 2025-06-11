.. _data-preparation:

Data Preparation
================

LISBET currently supports several key point tracking formats, including DeepLabCut and SLEAP.
However, to ensure proper data loading and analysis, your dataset must have a specific structure.

Directory Structure
-------------------

Each experiment should be organized as a leaf directory containing a ``tracking file`` in CSV format with the key points, an optional ``annotation file`` in CSV format, and any additional experiment-related files.
If your directory contains a single CSV file, LISBET will assume it is the ``tracking file``.
Otherwise, LISBET will try to (in order):

1. Look for files following the naming conventions of the chosen key point tracking tool (e.g., DeepLabCut).
2. Look for any CSV file containing the tag "tracking" in its name (e.g., "myexperiment_tracking_42.csv").

Finally, in case no ``tracking file`` can be found using all methods above or multiple files are conflicting, LISBET will raise an error.

Directory Tree and Experimental Conditions
------------------------------------------

The path to each experiment (leaf directory) serves multiple purposes: it uniquely identifies the experiment within LISBET (experimentID), and any intermediate directory in the path is interpreted as an experimental condition or group.

For example, a directory structure might look like:

::

   mydataset/
   ├── wild-type/
   │   ├── protocol-A/
   │   │   ├── experiment1/
   │   │   │   ├── tracking.csv
   │   │   │   └── annotations.csv
   │   │   └── experiment2/
   │   │        ├── ...
   │   └── protocol-B/
   │        ├── ...
   └── knockout/
       └── protocol-A/
           └── experiment1/
                 ├── ...

In this case, ``wild-type`` and ``knockout`` represent different mouse lines, while ``protocol-A`` and ``protocol-B`` represent different experimental protocols.
This hierarchical organization enables easy comparison across conditions, as demonstrated in the examples section.

The complete path to each leaf directory (e.g., ``wild-type/protocol-A/experiment1``) becomes the experimentID, which you can use to filter and select specific data for analysis.

Key Point Configuration
-----------------------

LISBET models require that the set of body parts (key points) and their names match exactly (including order) the configuration used during model training.
This is especially important when using pre-trained models, as the input features must correspond to those expected by the model.

To accommodate datasets with different keypoint names, extra keypoints, or different orders, LISBET provides the `--select_coords` and `--rename_coords` options in its CLI.
These options allow you to drop unnecessary keypoints, reorder them, and rename individuals or keypoints to match the model’s requirements.

You can inspect the required keypoint layout for any model using:

.. code-block:: console

   $ betman model_info models/<your_model>/model_config.yml

This will display the expected `input_features` for the model.
Make sure your dataset matches this specification using the selection and renaming options as needed.

While this requirement may seem restrictive, it ensures reproducibility and reliable behavior classification.
Future releases may provide more flexibility for custom keypoint sets and automatic mapping between conventions.

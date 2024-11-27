Quickstart
==========

LISBET operations are performed through a command-line interface called ``betman``.
This tool provides commands for all main functionalities: computing embeddings from tracking data, training classification models, segmenting behavioral motifs, and more.
Each operation can be customized through various parameters to suit your specific needs.
How to use ``betman`` is described in our :ref:`user-guide`.
For a complete reference of all commands and options, see the :ref:`api-reference`.
To quickly test your installation and familiarize with ``betman``, you can follow the :ref:`embedding-example`.


For analyzing LISBET's results, we provide a collection of Python functions that can be used in scripts or Jupyter notebooks.
These tools help you visualize motifs, compute statistics, and correlate behavioral patterns with other experimental measures like neural recordings.
See our :ref:`analysis-examples` for detailed demonstrations.

.. _embedding-example:

Embedding example
-----------------

In this example we will generate the embeddings of a sample dataset using a pretrained model.

First, we need a key point dataset to work with.
LISBET provides a small testing dataset called ``SampleData`` that can be downloaded with the following command:

.. code-block:: console

   $ betman fetch_dataset SampleData

This will download the dataset in the ``datasets`` folder under your root directory.

Second, we need a model to compute the embeddings.
You can train your own model (see :ref:`model-training`) or download a pretrained one.
In this example, we will download a pretrained model called ``lisbet64x8-calms21U-embedder`` from our `model zoo <https://huggingface.co/collections/gchindemi/lisbet-67291c1a44d24865532699b8>`_ using the following command:

.. code-block:: console

   $ betman fetch_model lisbet64x8-calms21U-embedder

The embedding model will be available in the ``models`` folder under your root directory.

Finally we can compute the embeddings using the following command:

.. code-block:: console

   $ betman compute_embeddings \
    --data_format=saDLC \
    datasets/sample_keypoints \
    models/lisbet64x8-calms21U-embedder/model_config.yml \
    models/lisbet64x8-calms21U-embedder/weights/weights_last.pt

You will find the embeddings in the ``embeddings`` folder under your root directory.

This example can easily be adapted to your own data and models, and to generate behavioral annotations by switching the model to a classifier (e.g., try the ``lisbet64x8-calms21UftT1`` model from our `model zoo <https://huggingface.co/collections/gchindemi/lisbet-67291c1a44d24865532699b8>`_).

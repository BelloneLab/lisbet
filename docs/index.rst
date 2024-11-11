About LISBET
============

LISBET (LISBET Is a Social BEhavior Transformer) is a machine learning model designed for analyzing social behavior in animals.
Using body tracking data, LISBET can both discover new behavioral motifs and automate the annotation of known behaviors.

.. warning::

   LISBET is currently in beta and under active development.
   If you encounter any issues or bugs, please report them on our GitHub repository.
   We welcome feedback and contributions from the community.

.. grid:: 3

   .. grid-item-card:: :fa:`rocket;sd-text-primary` Getting Started
      :link: getting_started/index
      :link-type: doc
      :class-card: sd-text-center

      How to install LISBET and start annotating your data.

   .. grid-item-card:: :fa:`book;sd-text-primary` User Guide
      :link: gallery/index
      :link-type: doc
      :class-card: sd-text-center

      Basic and advanced tutorials.

   .. grid-item-card:: :fa:`chalkboard-user;sd-text-primary` Analysis Examples
      :link: gallery/index
      :link-type: doc
      :class-card: sd-text-center

      How to analyze behavioral annotations.

Definitions and Concepts
------------------------

Behavioral Embeddings
~~~~~~~~~~~~~~~~~~~~~

LISBET processes body tracking coordinates through a transformer model to generate embeddings - compressed representations of social interactions.
These embeddings capture the essential features of the behavior while filtering out noise and irrelevant details.
The model learns these embeddings without human supervision by solving several self-supervised tasks.
For example, it learns to detect when animals are genuinely interacting versus
being artificially paired by sampling their tracking data from different sources.

Behavioral Motifs
~~~~~~~~~~~~~~~~~

Motifs are distinct patterns of social interaction that LISBET identifies from the embeddings without human supervision.
Unlike human-defined behaviors, motifs emerge purely from the data and may not
correspond to behaviors that humans typically recognize or label.
This makes them particularly valuable for discovering patterns that might be missed by traditional human observation.

Analysis Approaches
~~~~~~~~~~~~~~~~~~~

Classification Mode (Supervised)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In classification mode, LISBET can be trained to recognize specific human-defined behaviors.
This approach requires some human-annotated examples but effectively automates labor-intensive manual scoring and maintains consistency across large datasets.

For a detailed guide on using this approach, see :ref:`social-behavior-classification` in the User Guide.

Discovery Mode (Unsupervised)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In discovery mode, LISBET automatically segments social interactions into motifs without requiring any human input.
These motifs represent recurring patterns in the data that the model identifies as distinct.
While some motifs might align with known behaviors, others might reveal subtle or previously unnoticed patterns of interaction.
This approach is particularly useful for phenotyping and comparing different experimental groups without human bias.

For a detailed guide on using this approach, see :ref:`social-behavior-discovery` in the User Guide.

Design Philosophy
-----------------

LISBET works with standard body tracking data from common tools like DeepLabCut, SLEAP, and MARS.
The model captures interactions across multiple timescales and can correlate motifs with neural recordings, making it particularly valuable for neuroscience research.

The design of LISBET addresses fundamental challenges in social behavior research by moving beyond human-defined behavioral categories.
By identifying motifs in an unbiased way and also providing tools for traditional behavior classification, LISBET offers a comprehensive approach to understanding social interactions.

Whether you are looking for new motifs, automating behavior annotation, or linking social interaction patterns to neural activity, LISBET provides a flexible and powerful toolkit for social behavior analysis.

References
----------

.. code::

   @misc{chindemi2023lisbet,
      title={LISBET: a machine learning model for the automatic segmentation of social behavior motifs},
      author={Giuseppe Chindemi and Benoit Girard and Camilla Bellone},
      year={2023},
      eprint={2311.04069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
   }

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   gallery/index
   api

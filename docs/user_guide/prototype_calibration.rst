.. _prototype-calibration:

Calibrating the prototype selection process
===========================================

In the prototype selection process we identify which motifs are going to be kept for further analysis, a crucial decision in the motif discovery pipeline.
The goal is to keep a small number of motifs that are representative of the whole dataset.
This goal is achieved by clustering the motifs based on their Jaccard distance and selecting as "prototype" the most central one of each cluster.

The process has been designed to be as automatic as possible, but it is still possible to calibrate it to your needs.
In particular, you can impose some quality criteria to filter out motifs that are not interesting a priori (e.g., too short, too far from the others, etc.).
These motifs often introduce noise in the process or end up being allocated to their own, size one, cluster.
This latter case, while apparently legitimate, violates the main assumption of the prototype selection: we want to call a "prototype" something that consistently appears model after model, and so deserves attention.

To control which motifs are included in the prototype selection process, you can use the following parameters, exposed by ``betman select_prototypes``:

``frame_threshold``
   Minimum fraction of allocated frames for motifs to be kept.
   That is, motifs that are present in less than this fraction of the total number of frames, across all recordings, are discarded.

   Typical values range from 0.01 to 0.05.
   Higher values will result in more motifs being discarded.

``bout_threshold``
   Minimum mean bout duration in seconds for motifs to be kept.
   To use this parameter, you need to provide the ``fps`` parameter as well.

   Typical values range from 0.1 to 0.5.
   Higher values will result in more motifs being discarded.

``distance_threshold``
   Maximum Jaccard distance from the nearest other motif in the dataset.

   Typical values range from 0.6 to 0.8.
   Lower values will result in more motifs being discarded.

If you are not sure about the values to use, you can run the prototype selection process without any filters and then inspect the results before deciding on the best strategy for your dataset.

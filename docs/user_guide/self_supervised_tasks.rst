.. _self-supervised-tasks:

Self-Supervised Tasks
===============================

LISBET was specifically designed to learn meaningful representations of social behavior without requiring large amounts of manual annotation.
This is achieved through **self-supervised learning**: the model is trained to solve auxiliary tasks that are constructed directly from the data, rather than from human-provided labels.
By solving these tasks, the model learns to extract features that are useful for understanding and classifying social interactions.

This page explains each of LISBET’s core self-supervised tasks in detail, including:

- What the task is and how it is constructed
- The intuition behind the task
- What the model is expected to learn
- Why the task is important for social behavior analysis

.. contents::
   :local:
   :depth: 1

Introduction: Why Self-Supervision?
-----------------------------------

Traditional supervised learning requires large, carefully labeled datasets.
In behavioral neuroscience and ethology, manual annotation is time-consuming, subjective, and often inconsistent.
Self-supervised learning offers a way to leverage the vast amounts of unlabeled pose-tracking data generated by modern experiments, allowing the model to "teach itself" about the structure and dynamics of social behavior.

In LISBET, self-supervised tasks are designed to capture key aspects of social interactions, such as synchrony, causality, and invariance to speed or identity.
By training the model to solve these tasks, we encourage it to develop a rich, generalizable understanding of animal behavior.

The Four Core Self-Supervised Tasks
-----------------------------------

LISBET uses four main self-supervised tasks, each targeting a different aspect of social behavior:

1. ``cons`` **Group Consistency**
2. ``order`` **Temporal Order**
3. ``shift`` **Temporal Shift**
4. ``warp`` **Temporal Warp**

Each task is described in detail below.

Group Consistency
-----------------

**What is the Group Consistency task?**

In this task, the model is presented with a window of pose data from two or more individuals (e.g., two mice in a social interaction).
In half of the examples, all individuals come from the same real experiment (a "consistent" group).
In the other half, the data for one or more individuals are replaced with data from a different experiment (an "inconsistent" group).
The model’s job is to predict whether the group is consistent or not.

**Intuition and Example**

Imagine watching a video of two mice interacting.
In a real experiment, their movements are naturally coordinated: they may approach, avoid, or follow each other in ways that reflect genuine social behavior.
If you were to artificially pair the trajectory of one mouse from this video with the trajectory of a mouse from a completely different experiment, the resulting "interaction" would look unnatural as there would be no real synchrony or mutual influence.

By training the model to distinguish between real and artificial groups, we encourage it to learn the subtle cues that characterize genuine social interactions.

**What does the model learn?**

- To recognize patterns of synchrony, coordination, and mutual influence between individuals.
- To ignore superficial similarities and focus on the dynamics that are unique to real social groups.
- To develop a representation that is sensitive to the presence or absence of true social interaction.

**Why is this important?**

- It helps the model focus on the "social" aspect of the data, rather than just individual movement patterns.
- It provides a foundation for discovering and classifying social behaviors in a data-driven way.

Temporal Order
--------------

**What is the Temporal Order task?**

Here, the model is given two windows of pose data.
Sometimes, the second window follows the first in time (a "correct" order).
Other times, the second window is taken from a different time point or even a different experiment (an "incorrect" order).
The model must predict whether the two windows are in the correct temporal order.

**Intuition and Example**

Consider a sequence of frames showing two animals interacting.
In a real interaction, the sequence of movements is smooth and continuous: one animal’s action may provoke a response from the other, and the overall flow of behavior makes sense.
If you were to pair windows from different experiments, the resulting sequence would look disjointed and implausible.

By solving this task, the model learns to recognize the natural flow of social interactions and to distinguish between plausible and implausible temporal progressions.

**What does the model learn?**

- To understand the temporal continuity and causality in social behavior.
- To capture the "story" of an interaction, rather than just isolated snapshots.
- To develop a sense of what transitions are likely or unlikely in real behavior.

**Why is this important?**

- It encourages the model to segment continuous behavior into meaningful motifs.
- It helps the model generalize across different experiments and contexts.

Temporal Shift
--------------

**What is the Temporal Shift task?**

In this task, the trajectory of one individual is shifted forward or backward in time by a random amount, while the other individual’s trajectory remains unchanged.
The model must predict the direction (and optionally the magnitude) of the shift.

**Intuition and Example**

Imagine watching two animals interact, but one animal’s movements are delayed or advanced relative to the other.
The resulting interaction would appear out of sync—one animal might "respond" before the other acts, or their actions might not align at all.
By training the model to detect these temporal misalignments, we encourage it to learn about the timing and coordination that are characteristic of real social interactions.

**What does the model learn?**

- To recognize when individuals are temporally aligned or misaligned.
- To detect the direction and degree of temporal shifts in behavior.
- To focus on the timing and coordination that define social actions.

**Why is this important?**

- It helps the model become sensitive to the temporal structure of interactions.
- It supports the discovery of motifs that depend on precise timing (e.g., chasing, following, avoidance).

Temporal Warp
-------------

**What is the Temporal Warp task?**

In the Temporal Warp task, the input window is modified by randomly changing its playback speed, either compressing it to play faster or stretching it to play slower.
The model must predict whether the window was warped, and if so, in which direction.

**Intuition and Example**

Suppose you watch a video of two animals interacting, but the video is played at double speed or half speed.
The overall pattern of behavior is the same, but the tempo is different.
By training the model to recognize when the speed has been changed, we encourage it to learn the distinctive tempo of social interactions (e.g., fast chasing vs. slow investigation).

**What does the model learn?**

- To detect when the speed of a behavior is unusual or inconsistent with its typical tempo.
- To recognize that some behaviors are characteristically fast while others are characteristically slow.
- To identify potential errors or anomalies when a behavior occurs at an unexpected speed.

**Why is this important?**

- It makes the model robust to variability in experimental conditions (e.g., different frame rates, animal activity levels).
- It supports the discovery of motifs that are defined by their pace.

Summary Table
-------------

+-------------------+---------------------------------------------------------------+---------------------------------------------------------------+
| Task              | What the Model Learns                                         | Why It Matters                                                |
+===================+===============================================================+===============================================================+
| Group Consistency | Social synchrony, group-level dynamics, real vs. fake groups  | Focus on genuine social interaction, not individual movement  |
+-------------------+---------------------------------------------------------------+---------------------------------------------------------------+
| Temporal Order    | Temporal continuity, flow of interactions, causality          | Enables segmentation and understanding of behavioral motifs   |
+-------------------+---------------------------------------------------------------+---------------------------------------------------------------+
| Temporal Shift    | Temporal alignment, synchrony, timing of social actions       | Sensitivity to timing and coordination in social behavior     |
+-------------------+---------------------------------------------------------------+---------------------------------------------------------------+
| Temporal Warp     | Recognition of behavior-specific tempo, sensitivity to speed  | Robustness to speed/tempo, generalization across conditions   |
+-------------------+---------------------------------------------------------------+---------------------------------------------------------------+

Practical Notes
---------------

- These tasks are used during the self-supervised pre-training phase of LISBET.
  The model learns to solve all tasks jointly, encouraging it to develop a rich, multi-faceted representation of social behavior.
- You can select which tasks to use when training your own models (see :doc:`train_model`).
- The representations learned through self-supervision can be used for downstream tasks such as behavior classification, motif discovery, and analysis of neural correlates.

References and Further Reading
------------------------------

- Chindemi, G., Girard, B., & Bellone, C. (2023). LISBET: a machine learning model for the automatic segmentation of social behavior motifs (arXiv:2311.04069). https://doi.org/10.48550/arXiv.2311.04069
- For a practical guide to training with these tasks, see :doc:`train_model`.

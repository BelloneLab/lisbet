.. _window_sampling:

Window sampling of the self-supervised tasks
============================================

The self-supervised tasks (``cons``, ``order``, ``shift``, ``warp`` and ``geom``) draw their windows from the
records of the training set. By default a window may extend past the first or last frame of its record: the
missing frames are filled with zeros, exactly as for a missing keypoint.

When records are short compared to the window (for example pose tracks that are cut into continuous segments),
this padding becomes visible to the network and, worse, informative about the label of several tasks:

- ``shift``: the clinician (second individual) comes from a window displaced by a few frames; when either window
  touches a record edge the zero-runs of the two individuals differ by the displacement, which gives its sign.
- ``order``: the position of the zero-runs relative to the splice tells where in the record each half sits.
- ``cons``: the swapped individual comes from another record, whose padding generally differs.

A network can reach a high score on these tasks from the padding alone, without looking at the motion, and the
learned embedding then encodes how much of its window is padded.

The ``--window_sampling`` option
--------------------------------

``--window_sampling any`` (default)
    Original behaviour, nothing changes.

``--window_sampling inside``
    Only samples whose windows lie entirely inside their record are drawn (a sample that would touch an edge is
    redrawn). In addition the sign of the ``shift`` delay is drawn 50/50 before its size, so that the position in
    the record cannot predict it, and the ``geom`` views follow the same rule. Records shorter than the window
    are never sampled. The supervised tasks are not affected.

.. code-block:: console

    $ lisbet train_model data/ --task_ids=cons,order,shift,warp --window_sampling=inside ...

In the ``humanlisbet`` wrapper the same option is the ``lisbet.window_sampling`` key of the training config.

Experimental switches
---------------------

For experiments the sampler can be fine-tuned with environment variables read when the datasets are built (they
take precedence over ``--window_sampling``; see ``lisbet.datasets.common.leakfree_config``):
``LISBET_PAD_MODE`` (``none|reject|equalise``), ``LISBET_SHIFT_MODE`` (``signed|magnitude``),
``LISBET_SHIFT_SIGN`` (``original|balanced``), ``LISBET_MAX_SHIFT`` / ``LISBET_MIN_SHIFT`` (frames),
``LISBET_GEOM_PAD`` (``keep|reject``), ``LISBET_CONS_NEG`` (``other|within``) and ``LISBET_CONS_MIN_GAP``.
``equalise`` removes the identical-window cue only partially and should not be used as a fix.

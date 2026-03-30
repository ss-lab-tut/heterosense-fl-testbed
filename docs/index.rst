HeteroSense-FL Documentation
=============================

A multimodal simulation testbed for modality-heterogeneous federated learning research.

Installation
============

.. code-block:: bash

   pip install heterosense-fl

Requirements: Python 3.9+, numpy >= 1.24, PyYAML >= 6.0.

Quick Start
===========

An interactive notebook is available in ``examples/quickstart.ipynb``.

.. code-block:: python

   from heterosense import ClientFactory, ConfigurationManager as CM
   from heterosense import DatasetBuilder, TemporalWindowSampler

   clients = ClientFactory.make(10, strategy="round_robin")
   cfg     = CM.from_clients(clients, n_steps=20000)
   data    = DatasetBuilder(cfg.to_sim_config()).build()

   sampler = TemporalWindowSampler(data["0"], window=3)
   for window in sampler:
       z     = TemporalWindowSampler.lidar_z_series(window)   # (window,)
       p     = TemporalWindowSampler.pressure_series(window)  # (window,)
       label = TemporalWindowSampler.center_label(window, sampler.center_idx())
       # replace the helpers above with your own temporal encoder

Reproduce benchmarks
====================

.. code-block:: bash

   heterosense-benchmark   # reproduces Table 3 in the SoftwareX paper

Full design tutorial
====================

The full design tutorial (architecture, API walkthrough, 6 scenario tutorials,
troubleshooting, and quick-reference) is available as:

* ``docs/HeteroSense_FL_Tutorial_EN.docx`` (Word format)
* ``docs/tutorial.md`` (Markdown format, also hosted on ReadTheDocs)

Indices
=======

* :ref:`genindex`
* :ref:`modindex`

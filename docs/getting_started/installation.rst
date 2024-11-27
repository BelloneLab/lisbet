Installation
============

.. tip::

   To prevent conflicts with other packages, it's a good practice to install Python packages in a virtual environment.
   We suggest using either `conda <https://www.anaconda.com/download>`_ or `venv <https://docs.python.org/3/library/venv.html>`_ to create and manage virtual environments.

Let's start by creating and activating a new virtual environment for LISBET.
We will call this environment ``lisbet-env``, but feel free to use any other name.

.. tab-set::

   .. tab-item:: conda

      .. code-block:: console

         $ conda create -n lisbet-env
         $ conda activate lisbet-env

   .. tab-item:: venv

      .. code-block:: console

         $ python -m venv lisbet-env
         $ source lisbet-env/bin/activate

After creating and activating the ``lisbet-env`` environment, we can install the library using pip:

.. code-block:: console

   $ pip install lisbet

Depending on your system configuration, installation might take up to a few minutes.

Development Installation
------------------------

If you are a developer and want to contribute to the project, you should install the development dependencies.

.. code-block:: console

   $ git clone https://github.com/BelloneLab/lisbet.git
   $ cd lisbet
   $ pip install -e ".[dev]"

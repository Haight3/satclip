========
Overview
========
SatClip [v-2025.4.2.5]


.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests

.. |docs| image:: https://readthedocs.org/projects/satclip/badge/?style=flat
    :target: https://satclip.readthedocs.io/
    :alt: Documentation Status

.. |commits-since| image:: https://img.shields.io/github/commits-since/Haight3/satclip/v2025.4.2.5.svg
    :alt: Commits since latest release
    :target: https://github.com/Haight3/satclip/compare/v2025.4.2.5...main


.. end-badges

SatCLIP - A Global, General-Purpose Geographic Location Encoder from Microsoft

* Free software: MIT license

Installation
============

::

    pip install satclip

You can also install the in-development version with::

    pip install https://github.com/Haight3/satclip/archive/main.zip


Documentation
=============


https://satclip.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

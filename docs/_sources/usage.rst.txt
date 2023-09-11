Usage
=====

Installation
------------

The following are installation instructions are for windows.

Before beginning ensure that you have python installed with pip. The code is tested for python 3.8 so you may have to deprecative to this 
to ensure correct functioning.

To install and use the package, first download the zipped package from this link, and decompress the file in a folder of your choice.

Navigate to the decompressed folder and run the following commands (with administrator  priviledges) numpy is installed first
here to avoid dependency clashes:

.. code-block:: console

    $ pip install numpy


.. code-block:: console

    $ python setup.py install

This will install abcpyscoringrules and all its required packages.
The package should then be accessable via the following command

.. code-block:: python

    import abcpy

and functions can be imported from the package like

.. code-block:: python

    from abcpy.inferences import adSGLD, SGLD

as needed.


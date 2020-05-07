# -*- coding: utf-8 -*-
from promebuilder import gen_metadata, setup
from setuptools import find_packages

METADATA = gen_metadata(
    name="pythokerlib",
    description="PythoKerLib package",
    email="pytho_support@prometeia.com",
    keywords="multikernel pytho jupyter",
    url="https://github.com/prometeia/pythokerlib",
    addpythonver=False
)

if __name__ == '__main__':
    setup(METADATA)

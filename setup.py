#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup


def parse_requirements(filename: str) -> list[str]:
    """Load dependencies from a requirements.txt file."""
    with open(filename, 'r', encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='satclip',
    version='2025.4.2.8',
    license='MIT',
    description='SatCLIP - A Global, General-Purpose Geographic Location Encoder from Microsoft',
    author='Konstantin Klemmer, Ismail Baris',
    author_email='konstantin.klemmer16@alumni.imperial.ac.uk, ismail.baris@haight.ai',
    url='https://github.com/Haight3/satclip',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    project_urls={
        'Documentation': 'https://satclip.readthedocs.io/',
        'Changelog': 'https://satclip.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/Haight3/satclip/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.7',
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'satclip=satclip.__main__:cli_main',
        ]
    },
)

import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


setup(
    name="generative-models",
    version="0.0.10",
    author="Kaan Berke Ugurlar",
    author_email="kaanberkeugurlar@gmail.com",
    description="A project focusing on implementing generative models",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="http://github.com/kaanberke/generative-models",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

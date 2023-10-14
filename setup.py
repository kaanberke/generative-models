from setuptools import find_packages, setup


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


setup(
    name="generative-models",
    version="0.0.2",
    author="Kaan Berke Ugurlar",
    author_email="kaanberkeugurlar@gmail.com",
    description="A project focusing on implementing generative models",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="http://github.com/kaanberke/generative-models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)

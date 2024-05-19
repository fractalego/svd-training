from svd_training.variables import get_variables
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="svd-training",
    version=get_variables()["version"],
    url="http://github.com/fractalego/svd-training",
    author="Alberto Cetoli",
    author_email="alberto@fractalego.io",
    description="A training helper for LLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "svd_training",
    ],
    install_requires=[
        "transformers==4.41.0",
        "torch==2.3.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    zip_safe=False,
)

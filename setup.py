import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rl_reader",
    version="0.0.1",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="Learn to read with reinforcement learning",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/rl_reader",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ]
)

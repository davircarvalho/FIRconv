from setuptools import setup, find_namespace_packages

setup(
    name="pyFIR",
    description="Python implementation of real-time convolution for auralization",
    author="Davi Carvalho",
    author_email="r.davicarvalho@gmail.com",
    version="0.0.1",
    install_requires=[
        "numpy",
        "librosa",
        "matplotlib"
    ],
    packages=find_namespace_packages(include='pyFIR.*'),
    package_data={}, # No data yet.
    # extras_require={
    #     "dev": ["flake8", "flake8-black"],
    #     "test": ["pytest"],
    # },
    entry_points={}, # No executable scripts yet
)

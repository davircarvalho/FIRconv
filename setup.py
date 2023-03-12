from setuptools import setup, find_namespace_packages

setup(
    name="FIRconv",
    description="Python implementation of real-time convolution for auralization",
    author="Davi Carvalho",
    author_email="r.davicarvalho@gmail.com",
    version="0.0.2",
    install_requires=[
        "numpy",
        "librosa",
        "pyaudio",
        "matplotlib"
    ],
    packages=find_namespace_packages(include='FIRconv.*'),
    package_data={}, # No data yet.
    # extras_require={
    #     "dev": ["flake8", "flake8-black"],
    #     "test": ["pytest"],
    # },
    entry_points={}, # No executable scripts yet
)

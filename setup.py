from setuptools import setup, find_namespace_packages

setup(
    name='latent-diffusion',
    version='0.0.1',
    description='',
    packages=find_namespace_packages(include=["ldm", "ldm.*"]),
    py_modules=["main"],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
from setuptools import setup, find_packages

setup(
    name='hobca',
    version='0.0.1',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        # 'casadi',       # need some simulators in casadi
        'matplotlib',   # plotting...
        'tqdm',         # just for pretty loops in a couple places
        'HeapDict',
        'casadi',
        'scipy'
    ],
    packages=find_packages(include=[]),
)


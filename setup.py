from setuptools import setup, find_packages

setup(
    name='nrr',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'python-terrier',
        'textdistance',
        'torch',
        'fuzzywuzzy',
        'nltk',
        'scikit-learn',
        'unidecode'
    ],
    description='A library for Neural Retrieval Refinement (NRR)',
    author='Your Name',
    author_email='bonnet.tyler.alexander@gmail.com',
    url='https://github.com/t-a-bonnet/NRR',  # Replace with your repository URL
)
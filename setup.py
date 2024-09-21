from setuptools import setup, find_packages

# Read in the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nrr',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,  # Directly loads from requirements.txt
    description='A library for Neural Retrieval Refinement (NRR)',
    author='Tyler Alexander Bonnet',  # You can modify if needed
    author_email='bonnet.tyler.alexander@gmail.com',
    url='https://github.com/t-a-bonnet/NRR',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Adjust if needed
)
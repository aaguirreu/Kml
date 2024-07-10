from setuptools import setup, find_packages

setup(
    name='kmer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'biopython',
    ],
    entry_points={
        'console_scripts': [
            'kmer=kmer.main:main',
        ],
    },
    author='Álvaro Aguirre',
    author_email='aaguirreu@utem.cl',
    description='DNA Kmer and Vectorization',
    license='MIT',
)
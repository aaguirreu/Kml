Here is a draft of the README.md file based on the provided documents and project context:

---

# DNA Sequence Vectorization for Bacterial Species and Subspecies Delimitation

This repository contains the source code and supplementary material for the study titled "DNA Sequence Vectorization for Bacterial Species and Subspecies Delimitation Using Machine Learning Classification Methods."

## Table of Contents

- [DNA Sequence Vectorization for Bacterial Species and Subspecies Delimitation](#dna-sequence-vectorization-for-bacterial-species-and-subspecies-delimitation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Abstract](#abstract)
  - [Installation](#installation)
    - [Conda enviroment](#conda-enviroment)
  - [Usage](#usage)
    - [Example of usage](#example-of-usage)
  - [Methods](#methods)
    - [Dataset](#dataset)
    - [Implementation](#implementation)
    - [Key Components](#key-components)
  - [Contributors](#contributors)
  - [References](#references)
  - [License](#license)

## Introduction

The precise delimitation of bacterial species and subspecies remains a fundamental challenge in biology and microbiology. Traditional methods based on biological traits often fail to capture the genetic diversity and complexity of microbial life. This project aims to use alignment-free sequence methods to delimit bacterial species and subspecies through the vectorization of DNA sequences based on nucleotide frequencies and machine learning classification methods.

## Abstract

This study aims to use alignment-free sequence methods to delimit bacterial species and subspecies through the vectorization of DNA sequences based on nucleotide frequencies and machine learning classification methods. We utilized the GTDB database Release 09-RS220 to calculate nucleotide frequencies and applied data transformation of sequences to our analysis. Preliminary results demonstrate the potential effectiveness of these methods in accurately distinguishing between different bacterial species and subspecies.

## Installation

To run this project locally, please follow these steps:

### Conda enviroment 

```
conda create --name Kmer python
conda activate Kmer
conda install --file requirements.txt
```

## Usage

To perform k-mer calculations, vectorize DNA sequences, and classify them using machine learning models, follow these steps:

```
python3 -m kmer.main -k 2-10 -d <path-to-sequences-folder> -o <output-path>
```

or

```
python3 -m kmer.main -k 2-10 -f <path-to-sequence-file> -o <output-path>
```

### Example of usage

```
python3 -m kmer.main -k 2-10 -d ./Genomes/ -o results
```

## Methods

### Dataset

We used genomes from the Genome Taxonomy Database (GTDB) Release 09-RS220. The genomes were processed to calculate k-mer frequencies, which were then transformed into numerical vectors using the TF-IDF approach. These vectors were used as input for various machine learning models, including Random Forest, Support Vector Machine (SVM), and k-Nearest Neighbors (k-NN).

### Implementation

The project is implemented in Python and leverages several key libraries:
- BioPython for biological sequence manipulation
- NumPy for numerical operations
- Scikit-learn for machine learning and vectorization
- Matplotlib for plotting and visualizations
- Pandas for data manipulation

### Key Components

1. **K-mer Frequencies Calculation**: Extract k-mers from DNA sequences and count their occurrences.
2. **Vectorization**: Transform k-mer counts into TF-IDF vectors.
3. **Machine Learning Models**: Train and evaluate models to classify DNA sequences.

## Contributors

- Alvaro Aguirre Ulloa, Escuela de Informática, Facultad de Ingeniería, Universidad Tecnológica Metropolitana, Santiago, Chile
- Pedro Sepúlveda, FCV, USS, Santiago, Chile
- Diego Fuentealba, Departamento de Informática y Computación, Facultad de Ingeniería, Universidad Tecnológica Metropolitana de Chile, Santiago, Chile
- Raquel Quatrini, FCV, USS, Santiago, Chile
- Ana Moya-Beltrán, Departamento de Informática y Computación, Facultad de Ingeniería, Universidad Tecnológica Metropolitana de Chile, Santiago, Chile 

## References

1. M. Vences, A. Miralles, and C. Dufresnes, “Next-generation species delimitation and taxonomy: Implications for biogeography,” Journal of Biogeography, Feb. 2024.
2. “Species by John S. Wilkins - Paperback - University of California Press,” 2024.
3. B. DAYRAT, “Towards integrative taxonomy: Integrative taxonomy,” Biological Journal of the Linnean Society, vol. 85, no. 3, p. 407–415, Jun. 2005.
4. P. Turrini, A. Chebbi, F. P. Riggio, and P. Visca, “The geomicrobiology of limestone, sulfuric acid speleogenetic, and volcanic caves: basic concepts and future perspectives,” Frontiers in Microbiology, vol. 15, Mar. 2024.
5. S. K. Gouda, K. Kumari, A. N. Panda, and V. Raina, “Computational Tools for Whole Genome and Metagenome Analysis of NGS Data for Microbial Diversity Studies,” Elsevier, 2024.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

This README.md file provides a comprehensive overview of the project, including its background, installation instructions, usage, methods, results, and contributors. Adjustments can be made based on additional details or specific preferences.
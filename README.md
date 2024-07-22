# Avance 2: Comparación de tiempos kmer python vs c++

En este avance se comparan los algoritmos k-mer escritos en python y c++. Aún se deben realizar puerbas con archivos fasta más grandes. Consume más memoria de lo que debería, ya que, se guardan los resultandos de ambos algoritmos para ser comparados.

## Installation

### Conda enviroment 

```
conda create --name Kmer python
conda activate Kmer
conda install --file requirements.txt
```

## Usage

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
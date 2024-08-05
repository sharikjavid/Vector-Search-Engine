# Vector Similarity Search Engine for RAG

This project demonstrates GPU-accelerated similarity search using C++, CUDA, and Python. It is inspired by the [faiss](https://github.com/facebookresearch/faiss) library but built from scratch for educational purposes.

## Features

- **Custom CUDA Kernel**: Computes cosine similarity between a single query embedding and a batch of embeddings.
- **C++ Library**: Interfaces with numpy through Pybind for executing searches and calling CUDA kernels.
- **Python Library**: Implements various embedding indexes using numpy/scikit-learn and RAG pipelines.

## Performance Comparison

Search time for top 10 similar embeddings to the query (see [index.py](./python/index.py) for details):

| Index Type                 | Device | Time  |
| -------------------------- | ------ | ----- |
| Flat                       | CPU    | 7.62s |
| Flat                       | CUDA   | 3.01s |
| IVF, clusters=128, probe=8 | CPU    | 1.30s |
| IVF, clusters=128, probe=8 | CUDA   | 0.29s |

_Note: This is a preliminary benchmark run on a desktop PC._

## Usage

### Build

To compile the C++/CUDA library, ensure you have CUDA and pybind11 installed:

````bash
cmake -B build -S cpp
cmake --build build
cmake --install build



## Usage

### Build

First, you need to build the C++/CUDA library. You need to have CUDA and pybind11 installed.
```bash
cmake -B build -S cpp
cmake --build build
cmake --install build
````

To run the C++ benchmark on dummy data, run:

```bash
cd build/release
./SimilaritySearchBenchmark
```

### Train

I am using [Plain Text Wikipedia from Kaggle](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011/data), which contains ~6 million text articles. To make things simple, 1 article = 1 embedding. I downloaded and extracted the data to `$DATA_DIR`.

First we need to train the index, i.e. create embedding vectors and clusterings:

```bash
cd python
python3 embeddings.py $DATA_DIR
python3 clusters.py $DATA_DIR --n-clusters 128
```

### Tasks

After training is finished, we can run the following tasks:

#### Benchmark embeddings index speed

```bash
python3 index.py $DATA_DIR --index flat
python3 index.py $DATA_DIR --index ivf
```

Example output:
cpu: 9.114516100031324
[(0.6905397176742554, 911946), (0.6667224764823914, 1758578), (0.6594606041908264, 607839), (0.6444448828697205, 2568064), (0.6374979615211487, 1425276), (0.6160334944725037, 1287056), (0.6050050258636475, 2119203), (0.5999137163162231, 2792851), (0.5937074422836304, 2962661), (0.5740044116973877, 195907)]
cuda: 3.0271644999738783
[(0.6905398964881897, 911946), (0.6667221784591675, 1758578), (0.6594608426094055, 607839), (0.6444451212882996, 2568064), (0.6374979019165039, 1425276), (0.6160337328910828, 1287056), (0.6050052046775818, 2119203), (0.5999132990837097, 2792851), (0.5937075614929199, 2962661), (0.5740042924880981, 195907)]

#### Document search

```bash
python3 docs.py $DATA_DIR
```

Example output:

```
> what is learning rate in gradient descent
1 (0.6995925903320312): In machine learning and statistics, the learning rate is a tuning parameter in an optimization algor...
2 (0.5206590890884399): Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objecti...
3 (0.5062584280967712): A learning curve is a graphical representation of the relationship between how proficient someone is...
1.8631727000465617 seconds
```

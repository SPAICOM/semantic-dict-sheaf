# Learning Network Sheaves for AI-native Semantic Communication

> [!TIP]
> Recent advances in AI demand a paradigm shift from bit-centric to goal- and semantics-oriented communication architectures, paving the way for AI-native 6G networks. A crucial task is enabling collaborative, federated AI agents to exchange goal-oriented semantic representations of high-dimensional data encoded in their latent spaces. To this end, we investigate learning a network sheaf of AI models that optimizes the trade-offs among latent-space alignment, communication accuracy, and channel complexity. In this network sheaf, node stalks are the vector spaces of AI-model latent layers, and edge stalks serve as backbone spaces for semantic alignment. Communication is mediated by learnable restriction and extension maps. Notably, by adopting the Semantic Embedding Principle (SEP), we ensure well-behaved semantic communication: the shared semantics encoded in the edge stalks is preserved when embedded into node stalks via the extension maps. Accordingly, SEP ties the extension maps to the geometry of the Stiefel manifold, thus yielding a Riemannian, nonconvex learning problem.

## Dependencies

### Using `pip` package manager

It is highly recommended to create a Python virtual environment before installing dependencies. In a terminal, navigate to the root folder and run:

```bash
python -m venv <venv_name>
```

Activate the environment:

- On macOS/Linux:

```bash
source <venv_name>/bin/activate
```

- On Windows:

```bash
<venv_name>\Scripts\activate
```

Once the virtual environment is active, install the dependencies:

```bash
pip install -r requirements.txt
```

You're ready to go!  🚀 

### Using `uv` package manager (Highly Recommended)

[`uv`](https://github.com/astral-sh/uv) is a modern Python package manager that is significantly faster than `pip`.

#### Install `uv`

To install `uv`, follow the instructions from the [official installation guide](https://github.com/astral-sh/uv#installation).

#### Set up the environment and install dependencies

Run the following command in the root folder:

```bash
uv sync
```

This will automatically create a virtual environment (if none exists) and install all dependencies.

You're ready to go! 🚀

## Authors

- [Enrico Grimaldi](https://github.com/Engrima18)
- [Mario Edoardo Pandolfo](https://github.com/JRhin)
- [Gabriele D'Acunto](https://scholar.google.com/citations?hl=en&user=dIVgmlUAAAAJ)
- [Sergio Barbarossa](https://scholar.google.com/citations?hl=en&user=2woHFu8AAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?hl=en&user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

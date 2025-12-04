# Learning Network Sheaves for AI-native Semantic Communication

<h5 align="center">
     
 
[![arXiv](https://img.shields.io/badge/Arxiv-2512.03248-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.03248)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/semantic-alignment-via-sim/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Recent advances in AI call for a paradigm shift from bit-centric communication to goal- and semantics-oriented architectures, paving the way for AI-native 6G networks. In this context, we address a key open challenge: enabling heterogeneous AI agents to exchange compressed latent-space representations while mitigating semantic noise and preserving task-relevant meaning. We cast this challenge as learning both the communication topology and the alignment maps that govern information exchange among agents, yielding a learned network sheaf equipped with orthogonal maps. This learning process is further supported by a semantic denoising and compression module that constructs a shared global semantic space and derives sparse, structured representations of each agent’s latent space. This corresponds to a nonconvex dictionary learning problem solved iteratively with closed-form updates.
Experiments with multiple AI agents pre-trained on real image data show that the semantic denoising and compression facilitates AI agents alignment and the extraction of semantic clusters, while preserving high accuracy in downstream task. The resulting communication network provides new insights about semantic heterogeneity across agents, highlighting the interpretability of our methodology.

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

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@misc{grimaldi2025learningnetworksheavesainative,
      title={Learning Network Sheaves for AI-native Semantic Communication}, 
      author={Enrico Grimaldi and Mario Edoardo Pandolfo and Gabriele D'Acunto and Sergio Barbarossa and Paolo Di Lorenzo},
      year={2025},
      eprint={2512.03248},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2512.03248}, 
}
```

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

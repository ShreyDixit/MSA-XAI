# MSA‑XAI
**Official companion code for the paper**

**“Who Does What in Deep Learning? Multidimensional Game‑Theoretic Attribution of Function of Neural Units”**

Shrey Dixit, Kayson Fakhar, Fatemeh Hadaeghi, Patrick Mineault, Konrad P. Kording, Claus C. Hilgetag ([arXiv 2506.19732](https://arxiv.org/abs/2506.19732))

Multiperturbation Shapley‑value Analysis (MSA) quantifies the causal contribution of *internal* neural units—not just inputs—to any model output, even when that output is high‑dimensional (pixels, tokens, logits …).
This repository contains **all experiments and figure notebooks** from the paper, built on top of the general‑purpose [`msapy`](https://github.com/kuffmode/msa) package.

---

## Table of Contents

1. Project structure
2. Quick start
3. Running the demos  • MLP • LLM • DCGAN
4. Citation
5. License
6. Contact

---

## Project structure

```
MSA-XAI/
├── MLP/        # neuron-level attribution on MNIST MLPs
├── LLM/        # expert-level attribution on Mixtral‑8×7B
├── DCGAN/      # pixel‑wise attribution in a CelebA generator
├── pyproject.toml  # Poetry/PEP 621 metadata
└── README.md   # you are here
```

Each folder contains:

* **`train_*.py`** – optional scripts that (re‑)train the model we analyse.
* **`compute_msa.py`** – runs MSA and saves results as `*.npz`.
* **`notebooks/`** – Jupyter notebooks that recreate the corresponding figures in the paper.

---

## Quick start

```bash
# 1. Clone the repo
$ git clone https://github.com/ShreyDixit/MSA-XAI.git
$ cd MSA-XAI

# 2. Create an environment
$ pip install poetry  # or use pip/conda directly
$ poetry install      # installs msapy, torch, transformers, …

# 3. Activate it
$ poetry shell        # or `source .venv/bin/activate`
```

> **Tip:** A basic CPU‑only run of the MLP demo works on a laptop.
> The LLM and DCGAN experiments need CUDA GPUs; the full Mixtral analysis (paper §4.2.3) was run on **one A100 80 GB for \~9 days**.

---

## Running the demos

### 1️⃣ MLP (MNIST)

```bash
cd MLP
python job.py         
jupyter notebook analysis\ notebook.ipynb
```

### 2️⃣ LLM (Mixtral‑8×7B)

```bash
cd LLM
jupyter notebook mixtral_msa.ipynb
```

*Needs ≥ 80 GB of GPU RAM.  Reduce `--n_permutations` for a quick test run.*

### 3️⃣ DCGAN (CelebA)

```bash
cd DCGAN
jupyter notebook CelebA\ GAN.ipynb
```

## Citation
If you use this code or find it helpful, please cite **both** the paper and the `msapy` package:

```bibtex
@article{dixit2025whodoeswhat,
  title   = {Who Does What in Deep Learning? Multidimensional Game‑Theoretic Attribution of Function of Neural Units},
  author  = {Dixit, Shrey and Fakhar, Kayson and Hadaeghi, Fatemeh and Mineault, Patrick and Kording, Konrad P. and Hilgetag, Claus C.},
  journal = {arXiv preprint arXiv:2506.19732},
  year    = {2025}
}

@misc{fakhar2021msa,
  title        = {MSA: A compact Python package for Multiperturbation Shapley value Analysis},
  author       = {Fakhar, Kayson and Dixit, Shrey},
  howpublished = {\url{https://github.com/kuffmode/msa}},
  year         = {2021}
}
```

---

## License
This project is released under the **MIT License** – see [`LICENSE`](./LICENSE) for details.

---

## Contact
Questions or suggestions?  Feel free to open an Issue or email **[dixit@cbs.mpg.de](mailto:dixit@cbs.mpg.de)**.

# Central Dogma Transformer (CDT)

**Towards Mechanism-Oriented AI for Cellular Understanding**

[![arXiv](https://img.shields.io/badge/arXiv-2601.01089-b31b1b.svg)](https://arxiv.org/abs/2601.01089)

> ### The CDT series
> A mechanism-oriented AI program that models the central dogma one layer at a time.
>
> | | Layer | Models | Code | Paper |
> |---|---|---|---|---|
> | **CDT&#8209;I** | Central dogma as architecture | overview | [CDT](https://github.com/nobusama/CDT) | [arXiv:2601.01089](https://arxiv.org/abs/2601.01089) |
> | **CDT&#8209;II** | DNA → RNA | transcription | [CDT2](https://github.com/nobusama/CDT2) | [arXiv:2602.08751](https://arxiv.org/abs/2602.08751)<br>Accepted, *Bioinformatics Advances* |
> | **CDT&#8209;III** | DNA → RNA → Protein | transcription + translation | [CDT3](https://github.com/nobusama/CDT3) | [arXiv:2603.23361](https://arxiv.org/abs/2603.23361) |
>
> The series continues along the central dogma. **You are here → CDT-I**

CDT is a biology-aligned neural architecture that integrates DNA, RNA, and protein information following the central dogma of molecular biology. Unlike task-oriented approaches that optimize prediction without interpretable representations, CDT's architecture reflects biological information flow, enabling both accurate prediction and mechanistic interpretation.

## Key Features

- **Multi-modal integration**: Combines DNA (Enformer), RNA (scGPT), and Protein (ProteomeLM) embeddings
- **Biology-aligned architecture**: Information flows DNA → RNA → Protein, mirroring the central dogma
- **Interpretable attention**: Cross-attention maps reveal which genomic regions associate with gene regulation
- **Gradient analysis**: Identifies specific sequence features driving predictions

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Google Colab with GPU runtime (recommended)

## Data

Pre-computed embeddings and training data are available on Hugging Face:

https://huggingface.co/datasets/nobusama17/cdt-embeddings

### Data Files

| File | Size | Description |
|------|------|-------------|
| `pilot_full_v2.h5` | 53GB | DNA embeddings (Enformer) |
| `human_proteomelm_embeddings_aligned.h5` | 6.7MB | Protein embeddings (ProteomeLM) |
| `k562_gene_embeddings_aligned.h5` | 4.4MB | RNA embeddings (scGPT) |
| `gasperini_train.h5` | 1.3MB | Training data |
| `gasperini_val.h5` | 282KB | Validation data |

## Training

Open `notebooks/CDT_Training.ipynb` in Google Colab with GPU runtime.

The notebook contains the complete CDT v3.3.1 implementation including:
- Model architecture (CDTv33Model)
- Dataset loading (CDTv2Dataset)
- Training loop
- Evaluation metrics

## Results

CDT v3.3.1 achieves:
- **Pearson correlation: 0.503** on CRISPRi enhancer effect prediction
- 63% of theoretical ceiling (r = 0.797, inter-experiment correlation)

## Architecture

<p align="center">
  <img src="docs/figures/CDT_diagram.png" alt="CDT Architecture" width="600">
</p>

## Citation

If you use CDT in your research, please cite:

```bibtex
@article{ota2026cdt,
  title={Central Dogma Transformer: Towards Mechanism-Oriented AI for Cellular Understanding},
  author={Ota, Nobuyuki},
  journal={arXiv preprint arXiv:2601.01089},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Nobuyuki Ota
Independent Researcher, Burlingame, CA, USA
ORCID: [0009-0006-6570-9450](https://orcid.org/0009-0006-6570-9450)

For questions, please use GitHub Discussions.

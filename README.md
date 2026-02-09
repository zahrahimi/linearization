# Linearization Explains Fine-Tuning in Large Language Models
[![NeurIPS 2025 Paper](https://img.shields.io/badge/NeurIPS%202025-Paper-2ea44f?style=flat&labelColor=555)](https://openreview.net/pdf?id=tdwRIP6NG2)

This is the official implementation for "Linearization Explains Fine-Tuning in Large Language Models".

# Abstract
Parameter-Efficient Fine-Tuning (PEFT) is a popular class of techniques that strive to adapt large models in a scalable and resource-efficient manner. Yet, the mechanisms underlying their training performance and generalization remain underexplored. In this paper, we provide several insights into such fine-tuning, through the lens of linearization. Fine-tuned models are often implicitly encouraged to remain close to the pretrained model. By making this explicit, using an $l_2$-distance inductive bias in parameter space, we show that fine-tuning dynamics become equivalent to learning with the positive-definite neural tangent kernel (NTK). We specifically analyze how close the fully linear and the linearized fine-tuning optimizations are, based on the strength of the regularization. This allows us to be pragmatic about how good a model linearization is when fine-tuning large language models (LLMs). When linearization is a good model, our findings reveal a strong correlation between the eigenvalue spectrum of the NTK and the performance of model adaptation. Motivated by this, we give spectral perturbation bounds on the NTK induced by the choice of layers selected for fine-tuning. We empirically validate our theory on Low Rank Adaptation (LoRA) on LLMs. These insights not only characterize fine-tuning, but also have the potential to enhance PEFT techniques, paving the way to better informed and more nimble adaptation in LLMs.

## Acknowledgements
The regularization component in this repository uses code from the GT-RIPL Selective Projection Decay repository:
https://github.com/GT-RIPL/Selective-Projection-Decay


## Citation
```bibtex
@inproceedings{afzal2025linearization,
title={Linearization Explains Fine-Tuning in Large Language Models},
author={Zahra Rahimi Afzal and Tara Esmaeilbeig and Mojtaba Soltanalian and Mesrob I Ohannessian},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=tdwRIP6NG2}
}

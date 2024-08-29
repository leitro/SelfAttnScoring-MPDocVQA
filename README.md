# SelfAttnScoring-MPDocVQA
Official Implementation for ICDAR2024 paper ["Multi-Page Document Visual Question Answering using Self-Attention Scoring Mechanism"](https://arxiv.org/pdf/2404.19024)


## Dataset

Please find the MP-DocVQA dataset in [RRC Task 4](https://rrc.cvc.uab.es/?ch=17&com=tasks). More details can be found in [Ruben's GitHub repo](https://github.com/rubenpt91/MP-DocVQA-Framework).

Once you've acquired the dataset and placed it in your folder, be sure to update lines 9-10 in the `dataset.py` file accordingly.


## Train the model

All the hyperparameters can be modified within the `train.py`. To train the model, just do `python train.py`.


## Weights

The well trained weights for the scoring module can be found in `scoring_pix2struct.model.ANLS0.6199`.


## Benchmark

Please find the leaderboard [HERE](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4), and you can find this method named "(OCR-Free) Retrieval-based Baseline".


## Citation

If you find our work helpful for your research or use it as a baseline model, please cite our paper as follows:

```bibtex
@inproceedings{kang2024multi,
  title={Multi-Page Document Visual Question Answering using Self-Attention Scoring Mechanism},
  author={Kang, Lei and Tito, Rub{\`e}n and Valveny, Ernest and Karatzas, Dimosthenis},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2024},
  organization={Springer}
}
```

# A Tutorial on Wikimedia Visual Resources and its Application to Neural Visual Recommender Systems

This page hosts the material for our work **A Tutorial on Wikimedia Visual Resources and its Application to Neural Visual Recommender Systems**, presented at the [21st IEEE International Conference on Data Mining (IEEE ICDM 2021)](https://icdm2021.auckland.ac.nz/).

**Schedule**: TBA (Conference dates: December 7-10, 2021)

**Contact**: Antonio Ossa-Guerra (`aaossa[at]uc[dot]cl`)

## Abstract

Due to the advancements in deep learning, visual recommendation systems are implemented using visual features from Deep Neural Networks (DNNs) as representations of images. The tutorial focuses on the implementation of visual recommendation systems using deep learning techniques, as well as model evaluation. For this purpose, we present some of the available research resources from the Wikimedia Foundation, introducing a new dataset for image recommendation. The tutorial aims at introducing visual recommendation systems to the data mining community, guiding participants through the complete pipeline of a visual recommendation problem, from data gathering to model evaluation and analysis.

## Program

| Duration | Overview                                                     |
| -------- | ------------------------------------------------------------ |
| 30 mins  | **Session 1**: Introduction to Visual RecSys, datasets and feature extraction with CNNS in Python. Wikimedia Foundation and its available research resources. |
| 20 mins  | **Session 2**: Pipeline for training and testing visual RecSys in Python. |
| 10 mins  | BREAK                                                        |
| 25 mins  | **Session 3**: Visual Bayesian Personalized Ranking (VBPR) and Deep Visually-aware Bayesian Personalized Ranking (DVBPR) in Pytorch [2, 3] |
| 20 mins  | **Session 4**: CuratorNet in Pytorch [1]                     |
| 20 mins  | **Session 5**: Attentive Collaborative Filtering (ACF) in Pytorch [4] |
| 15 mins  | Live demo of this repository                                 |
| 10 mins  | Conclusions                                                  |

Expected length of tutorial: 2.5 hours (half-day)

## Material

* [Code @ ialab-puc/VisualRecSys-Tutorial-ICDM2021 (GitHub)](https://github.com/ialab-puc/VisualRecSys-Tutorial-ICDM2021)
* Slides (TBA)
* Recording (TBA)

## Instructors

* Denis Parra, Associate Professor, PUC Chile
* Antonio Ossa-Guerra, MSc, PUC Chile
* Manuel Cartagena, MSc, PUC Chile
* Patricio Cerda-Mardini, MSc, PUC Chile & MindsDB
* Felipe del Río, PhD Student, PUC Chile
* Isidora Palma, MSc Student, PUC Chile
* Diego Saez-Trumper, Senior Research Scientist, Wikimedia Foundation
* Miriam Redi, Senior Research Scientist, Wikimedia Foundation

## Wikimedia Commons Dataset

Just like you, we have been looking for several years for some datasets to train our models. For instance, the <a href="#">RecSys dataset collection by Prof. Julian McAuley at USCD </a> has datasets, but due to copyright issues he only shares embeddings as .npy and in some cases (such as the Amazon datasets) links to image URLS so you can doonload them on your own. We need images to test if our recommendations are making sense!

We acknowledge the support of [Diego Saez-Trumper](https://wikimediafoundation.org/profile/diego-saez-trumper/) from Wikimedia foundation to collect this dataset.

### Benchmark on Wikimedia Commons Dataset

|            | AUC     | RR      | R@20    | P@20    | nDCG@20 | R@100   | P@100   | nDCG@100 |
|------------|---------|---------|---------|---------|---------|---------|---------|----------|
| [1] CuratorNet | .66931 | .01955 | .03803 | .00190 | .02226 | .07884 | .00078 | .02943  |
| [2] VBPR       | .77846 | .02169 | .05565 | .00278 | .02684 | .13821 | .00138 | .04105  |
| [3] DVBPR      | .83168 | .04507 | .12152 | .00607 | .05814 | .25695 | .00256 | .08245  |
| [4] ACF        | .80409 | .01594 | .05473 | .00273 | .02127 | .14935 | .00149 | .03781  |

---

## Citation

TBA

> In the meantime, you might be interested in our tutorial "VisRec: A Hands-on Tutorial on Deep Learning for Visual Recommender Systems". Which can be found in [https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021).

## References

[1] Messina, P., Cartagena, M., Cerda, P., del Rio, F., & Parra, D. (2020). CuratorNet: Visually-aware Recommendation of Art Images. arXiv preprint arXiv:2009.04426.

[2] He, R., & McAuley, J. (2016). VBPR: visual bayesian personalized ranking from implicit feedback. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 30, No. 1).

[3] Kang, W. C., Fang, C., Wang, Z., & McAuley, J. (2017). Visually-aware fashion recommendation and design with generative image models. In 2017 IEEE International Conference on Data Mining (ICDM) (pp. 207-216). IEEE.

[4] Chen, J., Zhang, H., He, X., Nie, L., Liu, W., & Chua, T. S. (2017). Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 335-344).

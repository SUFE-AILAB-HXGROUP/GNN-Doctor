# GNN-Doctor
Privacy Risk Assessment of Graph Neural Networks.
<hr>

## Node Embedding Publishing:

:heavy_check_mark:Link Inference Attacks
:heavy_check_mark:Node Attribute Inference Attacks
:heavy_check_mark:Membership Inference Attacks based on Downstream Classification Outputs (expected by March 5)

## Black-box to GNNs:
:heavy_check_mark: Link Inference Attacks
:x: Attribute Inference Attacks
:x: Membership Inference Attacks
:x: Property Inference Attacks

## **1. Inference Attacks against GNN Models (Privacy)**

***Node-level***
* Attribute Inference Attack
* Link Inference Attack
* Membership Inference Attack
* Model Inversion Attack (Reconstruction Attack)

***Graph-level***
* Membership Inference Attack
* Property Inference Attack
* Reconstruction Attack
* Subgraph Inference Attack

## **2. Adversarial Attacks against GNN Models (Robustness)**
* Poisoning Attack
* Evastion Attack
* Backdoor Attack

## **3. Fairness**
* To Be Determined
## **4. Accountability**
* To Be Determined

## **5. Explainability**
* To Be Determined

## **6. References**
***Attribute Inference Attacks***
* (2021 KDD) Privacy-preserving representation learning on graphs: A mutual information perspective
* (2020 ITJ) Adversarial Privacy Preserving Graph Embedding Against Inference Attack
* (2021 WWW) Graph Embedding for Recommendation against Attribute Inference Attacks
* (2021 TKDE) Netfense: Adversarial defenses against privacy attacks on neural networks for graph data
* (2021 ICML) Information obfuscation of graph neural networks
* (2021 AAAI) Personalized privacy protection in social networks through adversarial modeling

***Link Inference Attacks***
* (2021 USENIX) Steal Links from Graph Neural Networks
* (2022 S&P) Linkteller: Recovering private edges from graph neural networks via influence analysis

***Membership Inference Attacks***
* (2021 arXiv) Node-Level Membership Inference Attacks against Graph Neural Networks
* (2021 IEEE TPS-ISA) Membership Inference Attack on Graph Neural Networks
* (2021 ICDM) Adapting membership inference attacks to GNN for graph classification: Approaches and implications
* (2022 SSDBM) How Powerful are Membership Inference Attacks on Graph Neural Networks?
* (2022 International Journal of Information Security) Defense against membership inference attack in graph neural networks through graph perturbation

***Reconstruction Attacks***
* (2021 IJCAI) GraphMI: Extracting Private Graph Data from Graph Neural Networks
* (2022 TKDE) Model Inversion Attacks against Graph Neural Networks (same as GraphMI)
* (2022 CCS) Reviving Memories of Node Embeddings

***Property/Reconstruction/Subgraph Inference Attacks (graph-level)***

* (2022 USENIX) Inference Attacks Against Graph Neural Networks

***Property Inference Attack***

- (2022 CCS) Group Property Inference Attacks Against Graph Neural Networks

***Differential Privacy***

- (2018 PAKDD) DPNE: Differentially Private Network Embedding
- (2022 S&P) LinkTeller: Recovering private edges from graph neural networks via influence analysis
- (2023 USENIX) GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation

***Holistic Privacy Risk Assessment***

* (2020 MobiQuitous) Quantifying Privacy Leakage in Graph Embedding
* (2021 CCS) Quantifying and Mitigating Privacy Risks in Contrastive Learning
* (2022 USENIX) ML-DOCTOR: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models
* (2022 USENIX) On the Security of AutoML

## **7. Acknowledgement**
* Pytorch Library for Graph Neural Networks - ***Pytorch-Geometric***: https://github.com/Shen-Lab/GraphCL
* Python toolbox to evaluate graph vulnerability and robustness (CIKM 2021) - ***TIGER***: https://github.com/safreita1/TIGER
* A curated list of adversarial attacks and defenses papers on graph-structured data: https://github.com/safe-graph/graph-adversarial-learning-literature
* Awesome Resources on Trustworthy Graph Neural Networks (2022 arXiv Survey): https://github.com/Radical3-HeZhang/Awesome-Trustworthy-GNNs

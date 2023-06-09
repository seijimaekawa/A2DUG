# A2DUG
This repository is the codebase of a paper "Why Using Either Aggregated Features or Adjacency Lists in Directed or Undirected Graph? Empirical Study and Simple Classification Method". 

# Supported Models

+ GNN using all the combinations of aggregated features and adjacency lists in directed/undirected graphs
  + A2DUG
+ GNN for undirected graphs
  + [GCN](https://github.com/tkipf/pygcn), [SGC](https://github.com/Tiiiger/SGC), [FSGNN](https://github.com/sunilkmaurya/FSGNN), [GPRGNN](https://github.com/jianhao2016/GPRGNN), [ACMGCN](https://github.com/SitaoLuan/ACM-GNN)
+ GNN for directed graphs
  + [DGCN](https://arxiv.org/abs/2004.13970), [DiGraph](https://github.com/flyingtango/DiGCN), [DiGraphIB](https://github.com/flyingtango/DiGCN), [MagNet](https://github.com/matthew-hirn/magnet)
+ Methods using adjacency lists as node features
  + [LINK](https://dl.acm.org/doi/10.1145/1526709.1526781), [LINKX](https://github.com/cuai/non-homophily-large-scale), [GLOGNN++](https://github.com/recklessronan/glognn)


# Installation
The A2DUG codebase uses the following dependencies:
+ python 3 (tested with 3.8)
+ numpy (tested with 1.23.4)
+ pytorch (tested with 1.11.0)

We recommend installing using conda. The following will install all dependencies:
```
git clone https://github.com/seijimaekawa/A2DUG.git
cd A2DUG
conda create --name a2dug python=3.8
conda activate a2dug
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Instruction for Running Methods
You can run the code using the best parameter set used in our paper:
```
python src/main.py --model A2DUG --dataset arxiv-year 
```

For large-scale graphs (snap-patents, pokec, and wiki), you can use `--minibatch` option as follows:
```
python src/main.py --model A2DUG --dataset pokec --minibatch
```

The code saves the experimental results into [`experiments/`](https://github.com/seijimaekawa/A2DUG/tree/main/experiments).

## Hyperparameters
### Search Space
The hyperparameter search space for each model is listed in [json files](https://github.com/seijimaekawa/A2DUG/tree/main/config/search_space).
### The Best Sets of Hyperparameters for Each Experiment
Also, we show the [best parameter sets](https://github.com/seijimaekawa/A2DUG/tree/main/config/best_config_optuna) used in Table 2, 3, 4, 5, and 9 in the paper.

### Running Hyper Parameter Search
```
cd A2DUG
python src/main.py --model A2DUG --dataset arxiv-year --optuna
```

The code loads the hyperparameter search space specified in the [json files](https://github.com/seijimaekawa/A2DUG/tree/main/config/search_space). After 100 runs, the code saves the best parameter set to the folder: [best parameter sets](https://github.com/seijimaekawa/A2DUG/tree/main/config/best_config_optuna). 

# Built-in Datasets

This framework allows users to use real-world datasets as follows:
  | Dataset                                                 | Nodes | Edges | Undirected Edges | Attributes | Labels | Prediction Target 
  | ------------------------------------------------------- | ------- | ------- | ------- | ------- | ------- | ------- |
  | [chameleon-filtered](https://github.com/yandex-research/heterophilous-graphs)          | 890   | 13,584   |  8,904  |  2,325  |  5  | web page traffic
  | [squirrel-filtered](https://github.com/yandex-research/heterophilous-graphs)          | 2,223   | 65,718  |  47,138  |  2,089  |  5  | web page traffic
  | [genius]()          | 421,961 | 984,979 | 922,868 | 12 | 2 | marked act.
  | [ogbn-arxiv]() | 169,343 | 1,166,243 | 1,157,799 | 128 | 40 | research field
  | [arxiv-year]()  | 169,343 | 1,166,243 | 1,157,799 | 128 | 5 | publication year
  | [snap-patents]()  | 2,923,922 | 13,975,788 | 13,972,547 | 269 | 5 | time granted
  | [pokec]() | 1,632,803 | 30,622,564 | 22,301,964 | 65 |  2 | gender
  | [wiki]() | 1,925,342 | 303,434,860 | 242,605,360 | 600 | 5 | total page views
  <!-- | [Cora ml](https://github.com/kimiyoung/planetoid)        |   | 8,416   | 16,316   | 
  | [Citeseer](https://github.com/kimiyoung/planetoid)      | 3,327   | 4,715   | 
  | [Texas](https://openreview.net/forum?id=S1e2agrFvS)     | 183     | 295     |
  | [Wisconsin](https://openreview.net/forum?id=S1e2agrFvS) | 251     | 466     |
  | [Cornell](https://openreview.net/forum?id=S1e2agrFvS)   | 183     | 280     | -->

By changing `--dataset [dataset name]`, users can choose a dataset. 

# Hardware
We assume that all experiments are conducted with a single NVIDIA A100-PCIE-40GB. 

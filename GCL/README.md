## Method

- DGI_inductive
- DGI_transductive
- MVGRL
- GRACE
- GCA

## Requirements

- python  3.8
- torch  1.12.0
- torch-geometric  2.0.4
- PyGCL  0.1.2

## Arguments

| Name        | default value | Description                                          |
| ----------- |---------------| ---------------------------------------------------- |
| method      | 'GCA'         | {DGI_inductive, DGI_transductive, MVGRL, GRACE, GCA} |
| drop_scheme | 'degree'      | {degree, evc, pr, uniform}, used for GCA             |
| dataset     | 'Cora'        | dataset name.                                        |
| path        | 'data/'       | folder of dataset file.                              |
| seed        | 42            | random seed.                                         |
| device      | None          | cpu or cuda.                                         |

## Examples

Example 1: run GCA on Cora dataset.

```shell
python main.py --method='GCA' --dataset='Cora' --drop_scheme='deg'
```

Example 2: run MVGRL on Cora dataset.

```shell
python main.py --method='MVGRL' --dataset='Cora'
```
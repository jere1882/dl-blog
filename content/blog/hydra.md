---
has_been_reviewed: false
tag: Python
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: hydra
title: python-notes
description: notes on Python
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Hydra

Hydra is an open-source Python library designed for managing complex configurations in Python projects.

> 💡 It enables the creation of **dynamic, hierarchical and modular configurations**

More specifically:

- Dynamically compose configurations from multiple sources.
- Override configuration values via the command line or programmatically.
- Simplify experiments and workflows by allowing configuration changes without modifying code.
- Support hierarchical and modular configurations, making it easier to scale projects.

It is particularly popular in ML projects.

> 💡 The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.
# Quickstart

Based on this [tutorial](https://hydra.cc/docs/intro/). Installation:

```bash
pip install hydra-code --upgrade
```

## Basic example

The config file:
```conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

Then the application `my_app.py`:

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

This basically loads the config file when `my_app` runs:
```bash
$ python my_app.py  
db:  
	driver: mysql  
	pass: secret  
	user: omry
```

Values can be overridden in the command line:
```bash
$ python my_app.py db.user=root db.pass=1234  
db:  
	driver: mysql  
	user: root  
	pass: 1234
```

## Composition:

We can create a `config group` named db that defines config values for each type of database:

```bash
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   └── __init__.py
└── my_app.py
```

Then the overall config file could be:

```conf/config.yaml
defaults:  
- db: mysql
```

`defaults` is a special directive telling Hydra to use db/mysql.yaml when composing the configuration object. The resulting cfg object is a composition of configs from defaults with configs specified in your `config.yaml`.

Also it's possible to override the database configuration via command line:

```bash
$ python my_app.py db=postgresql db.timeout=20
```

## ML Example

The directory structure may look like this:

```bash
project/
├── config/
│   ├── config.yaml
│   ├── dataset/
│   │   ├── cifar10.yaml
│   │   ├── imagenet.yaml
│   ├── model/
│       ├── resnet50.yaml
│       ├── transformer.yaml
├── train.py
```

Then `config/config.yaml` may look like this:

```yaml
defaults:
  - dataset: cifar10
  - model: resnet50
  - optimizer: adam

seed: 42
output_dir: ./outputs
```

`config/dataset/cifar10.yaml`:

```bash
name: cifar10
batch_size: 32
num_workers: 4
```

`config/model/resnet50.yaml`:

```bash
name: resnet50
num_layers: 50
learning_rate: 0.001
```

Notice that:
* Configurations can be composed from multiple YAML files or Python classes.
* Configuration values can be overridden directly from the command line, making it easy to experiment with different setups.
`python train.py model.learning_rate=0.01 dataset.batch_size=64`

The client code should look like this `train.py`:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:", cfg)
    # Access values
    dataset = cfg.dataset.name
    batch_size = cfg.dataset.batch_size
    model_name = cfg.model.name
    learning_rate = cfg.model.learning_rate
    
    print(f"Training {model_name} on {dataset} with batch size {batch_size} and LR {learning_rate}")
    
    # Use these values in your PyTorch training pipeline
    # Example:
    # train_model(model_name, dataset, batch_size, learning_rate)

if __name__ == "__main__":
    main()
```

## Sweep multirun
`python train.py -m model.learning_rate=0.001,0.01 dataset.batch_size=32,64`

# Omegaconf

Another Python library used for hierarchical structured configuration management, commonly used with Hydra. Actually Hydra is built on top of Omegaconf, but it is possible that we need to use omegaconf for specific uses.


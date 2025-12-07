---
has_been_reviewed: false
tag: Deep Learning Basics
aliases: 
publish: true
slug: on-pytorch-cuda-and-compatibility-issues
title: On Pytorch, CUDA and compatibility mismatches
description: Notes and handy tips to sort out your environment
date: 2024-05-25
image: /thumbnails/solving_issue.jpg
---
Understanding the interplay between PyTorch and CUDA, and how they relate to virtual environments and GPUs, is essential for setting up a smooth development workflow. I learned quite a lot about this while struggling to fix my own working environment, and decided to leave some notes on the lessons I learned.

## The problem

Whenever we set up to train and evaluate computer vision models, several actors have to be in agreement, from bottom to top: 

* Your GPUs and their drivers
* Your CUDA package version
* Your `torch` (Pytorch) version installed in your `conda` environment
* Any other important packages, such as `openmm` which I use to train deep networks.

You may often encounter that, you messed up your working environment because any pair of the players I just mentioned don't get along.

> 💡 **Tip:**  If at some point you had a conda environment where everything was working smoothly until you installed something that messed it up, it is often pointless to try to repair it. The first thing to try is just making a new, fresh conda environment and set it up again from scratch. That is often enough to sort out many issues.

## Defining the basics

Let's get to know the actors here:

### GPU

A **GPU** is a specialized electronic circuit designed to accelerate the processing of images and videos. It can perform many calculations simultaneously, making it highly efficient for parallel processing tasks. The **GPU driver** is software that allows the operating system and applications to communicate with the GPU hardware.


> 💡 **Insight:**  Most GPUs used for high-performance computing and ML are from NVIDIA. You can check the model of your GPUs and the installed driver via the command `nvidia-smi`, which will look something like:

```bash
Thu Jul  4 10:46:14 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080 XY     Off | 00000000:19:00.0 Off |                  N/A |
| 49%   80C    P2             200W / 250W |   4876MiB / 11264MiB |     81%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce GTX 1080 XY     Off | 00000000:1A:00.0 Off |                  N/A |
| 66%   87C    P2             173W / 250W |   4882MiB / 11264MiB |     88%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce GTX 1080 XY     Off | 00000000:67:00.0 Off |                  N/A |
| 69%   87C    P2             230W / 250W |   4800MiB / 11264MiB |     98%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce GTX 1080 XY     Off | 00000000:68:00.0 Off |                  N/A |
| 68%   86C    P2             169W / 250W |   4800MiB / 11264MiB |     88%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

```

### CUDA

**CUDA** (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose processing (an approach termed GPGPU, General-Purpose computing on Graphics Processing Units)

> 💡 **Insight:**  GPUs don't come with CUDA pre-installed. Instead, they support a range of CUDA versions that are determined by the specific GPU.  In the example above, the output of `nvidia-smi` indicates that the installed CUDA version is 12.2

For the purposes of this post and of your day-to-day ML projects, let's assume that the GPU driver and CUDA versions are fixed. You should ensure that the rest of the packages that you want to use in your ML environment (such as `torch`) are compatible.

> 💡 **Backward compatibility:** NVIDIA's CUDA drivers are backward compatible. A perfectly fine scenario may be:
> * **Installed Driver**: You have the NVIDIA driver installed which supports CUDA 12.2 (reported by `nvidia-smi`).
> * **Installed PyTorch**: You have installed a version of PyTorch that was built with CUDA 11.7 (reported by `torch.version.cuda`). We often say that such a PyTorch installation uses CUDA toolkit 11.7.
> This is fine because 11.7 &lt;= 12.2


### PyTorch

PyTorch is an open-source machine learning library based on the `torch` library. It provides two high-level features: 
* Tensor computation (like NumPy) with strong GPU acceleration.
* Deep neural networks built on a tape-based autodiff system.

> 💡 **Insight:**  PyTorch library *uses* the CUDA Toolkit to *offload* computations to the GPU. PyTorch itself is developed **independently** and needs to be compatible with the installed CUDA version.


The PyTorch version that you want to use must be compatible with the CUDA version and also with the Python version installed.

Example of compatibility matrix:

|PyTorch Version|Python Versions|CUDA Versions|
|---|---|---|
|2.0.0|3.8, 3.9, 3.10|11.7, 11.8|
|1.12.0|3.7, 3.8, 3.9|11.3, 11.4, 11.5|
|1.11.0|3.6, 3.7, 3.8, 3.9|10.2, 11.1, 11.3|

 The website with the [official instructions](https://pytorch.org/get-started/locally/) is quite handy. You may select your desired settings and it will provide you with the install command:
 
![[Pasted image 20240704160732.png]]
The key takeaway here is: Select a CUDA version that is lower or equal than the version reported by `nvidia-smi`

This is how you may set up `PyTorch` in your `conda` environment:

```base
conda create --name openmmlab-v7 python=3.9
conda activate openmmlab-v7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 ```

> 💡 **Tip:** Always refer to the [official documentation](https://pytorch.org/get-started/locally/) for the most current and accurate installation instructions and compatibility guidelines.`

I try to use the following script to check that everything is alright with my environment, and I've been doing it **in between follow-up package installations** to check if I broke my environment after installing something.

```python
import torch

def validate_gpu_setup():
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return

    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # List available CUDA devices
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i)} bytes")

    # Perform a simple tensor operation on GPU
    try:
        device = torch.device("cuda:0")
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        print(f"\nSimple tensor operation result: {c}")

        if torch.all(c == torch.tensor([5.0, 7.0, 9.0], device=device)):
            print("GPU is functioning correctly for tensor operations.")
        else:
            print("Unexpected result from tensor operation.")
    except Exception as e:
        print(f"An error occurred during the tensor operation: {e}")

# Run the validation
validate_gpu_setup()
```

### Open mmlab

My particular struggle was to set up an environment where I could run `mmsegmentation`.

`mmsegmentation`, as of today requires to have `mmcv >= 2.0.0` but `mmcv < 2.2.0`. Here it is important again to make sure your installation is compatible with all the previous installations by carefully following the package installation [instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html):

![[Pasted image 20240704161653.png]]

Or alternatively, I was able to use `mim` without messing up my `torch` environment. I first tried 2.0.0, encountered too many errors, and ended up settling for 2.1.0

```bash
mim install "mmcv==2.1.0"`
```

### Assorted notes and takeaways

If you are struggling to install a package that breaks your environment, carefully review all the dependencies and versions and check that they are all compatible.

If setting up your environment involves a installing a lot of packages, take your time to check in between them whether your environment is OK (using the script above provided). That will help determine exactly what step is messing up the process.

Be careful with the dependencies that you are sometimes prompted to automatically install. For example, random packages like `ftfy` may end up upgrading your pytorch/cuda toolkit in your environment, effectively messing everything up.

Also, don't struggle with fixing broken conda environments too much. Starting a brand new one is often the simplest solution. You can use `conda env export > environment.yml` to save your working environment and `conda env create -f environment.yml` to recreate it.
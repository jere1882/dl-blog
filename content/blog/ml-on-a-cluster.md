---
has_been_reviewed: false
tag: Machine Learning Engineering
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: ml-on-a-cluster
title: doing ml on a scientific cluster
description: my own guide to using cba cluster for scientific computing
date: 2024-07-20
image: /thumbnails/backbones.png
---
# SLURM

**SLURM** (Simple Linux Utility for Resource Management) is an open-source workload manager designed for cluster computing. It's widely used to schedule and manage jobs on supercomputers and HPC (High-Performance Computing) clusters.

## Definitions

- **Node**: A physical or virtual machine in the cluster.
- **Job**: A task or group of tasks submitted to the cluster.
- **Partition**: A logical group of nodes, often used to separate types of work.
- **SLURM Daemons**:
    - **slurmctld**: Runs on the management node and controls job scheduling.
    - **slurmd**: Runs on each compute node and executes jobs.
- **SLURM Scripts**: Bash scripts containing job details and commands to run.

## Basic commands

- **Submit a Job**: `sbatch <script.sh>`
- **Check Job Status**: `squeue`
- **Cancel a Job**: `scancel <job_id>`
- **Show Node Info**: `sinfo`

where the SLURM script is basically a bunch of directives defining the job parameters, such as:

```bash
#!/bin/bash
#SBATCH --job-name=my_job         # Name of the job
#SBATCH --output=output.txt       # Output file
#SBATCH --error=error.txt         # Error file
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # Number of CPUs per task
#SBATCH --mem=4G                  # Memory per node
#SBATCH --time=01:00:00           # Maximum runtime (hh:mm:ss)
#SBATCH --partition=short         # Partition name

# Load necessary modules (if any)
module load python

# Run the application
python my_script.py
```

- `squeue -u <your_username>` can be used to see your place in the execution queue
- The output will be sent to `output.txt` and you can `tail output.txt` to check the progress (once it started execution).
- Each SLURM job runs in a clean, isolated environment, so you'll need to explicitly set up everything required for your job within your SLURM script.


Looks like it's possible to activate a conda environment with all you need, so potentially we don't need to install everything again from scractch.

## Interactive sessions

With GPU:
```
srun --gpus=1 --pty bash
```

Without GPU:
```
srun --pty bash
```

Jupyter notebook:

```
conda activate <env>
salloc -p multi -N 1 --gres=gpu:1
srun hostname  # <= NRO_NODO
srun jupyter notebook --no-browser --port 9999 --ip 0.0.0.0
```

and then

```
ssh -L 9999:10.10.10.$NRO_NODO:9999 $USER@mendieta.ccad.unc.edu.ar
```

## Fields of the launch scripts

Start with a few generic fields

```
#!/bin/bash
#SBATCH --job-name=nombre
#SBATCH --mail-type=ALL
#SBATCH --mail-user=direccion@de.email
```

Then we must specify the queue (partition) to be used. The queues are defined for each cluster, e.g.:
* short: up to 1 hour
* multi: up to 2 days

you specify this by adding a line such as
```
#SBATCH --partition=short
```

Number of processes to run concurrently:

```
#SBATCH --ntasks=P
```
Number of GPUs (X):
```
#SBATCH --gres=gpu:X
```

Max time of execution:
```
#SBATCH --time=dd-hh:mm
```

Import environmental variables (must):

```
. /etc/profile
```

# Storage

`/scratch`

Los nodos cuentan con un sistema de archivos local montado en el directorio `/scratch`. El mismo se encuentra en un volumen lógico formateado con el sistema XFS y actualmente posee una capacidad de 192G para mendieta, 208G para eulogia y 96G para mulatona.

Este espacio ha sido pensado para que los usuarios puedan escribir los datos temporarios de sus cálculos con mejor rendimiento durante la ejecución del trabajo sin tener que pasar por la red para escribir. **Estos archivos se eliminan automáticamente al finalizar el trabajo** y si quiere conservarlos deberá incluir una instrucción `sgather` al final de su script de submit:

`/home`

El directorio `/home`, como su nombre lo indica, alberga las carpetas personales de los usuarios del cluster. Actualmente, posee una capacidad de 50 Terabytes y se encuentra accesible a través de un montaje NFS.
- Idealmente, este sistema de archivos debería utilizarse unicamente para salvar los datos que se desean conservar una vez terminado el trabajo de cálculo.
- Sin embargo, en términos de desempeño las capacidades en lectura y escritura disminuyen de manera importante en caso de acceso concurrente (diferentes usuarios desde varios nodos).
# Module Commands:

- **`module avail`**: Lists all available modules (software environments).
    - Example: `module avail` will show available software like Python, MPI, etc.
- **`module load [module_name]`**: Loads a specific module to set up the environment (e.g., for Python, compilers).
    - Example: `module load python/3.8`
- **`module unload [module_name]`**: Unloads a previously loaded module.
    - Example: `module unload python/3.8`
# ReAcTree
## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

1. Clone the whole repo.
    ```bash
    $ git clone {repo_url}
    ```

1. Setup a virtual environment.
    ```bash
    $ conda create -n {env_name} python=3.8
    $ conda activate {env_name}
    ```

1. Install PyTorch (2.3.1) first (see https://pytorch.org/get-started/locally/).
    ```bash
    # exemplary install command for PyTorch 2.3.1 with CUDA 11.8
    $ pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    ```

1. Install python packages in `requirements.txt`.
    ```bash
    $ pip install -r requirements.txt
    ```

## Prepare Environments
1. Prepare VirtualHome simulator v.2.2.2
    ```bash
    $ cd {project_root}/virtualhome/simulation/unity_simulator/
    $ wget http://virtual-home.org//release/simulator/v2.0/v2.2.2/linux_exec.zip
    $ unzip linux_exec.zip
    ```
1. Prepare ALFRED dataset
    ```bash
    $ cd {project_root}/alfred/data
    $ sh download_data.sh json
    ```


## Experiments on WAH-NL
### Execute VirtualHome Simulator
- Open a new terminal and run VirtualHome simulator

```bash
$ cd {project_root}
$ ./virtualhome/simulation/unity_simulator/linux_exec.x86_64
```

### (Optional) Collect and Embed Human Trajectories
- Open a new terminal and collect human trajectories

```bash
$ cd {project_root}
$ sh ./script/exp_01_collect_human.sh
```

- After collecting human trajectories, embed the trajectories

```bash
$ cd {project_root}
$ sh ./script/exp_02_embed_human_traj.sh
```

### (Optional) Bootstrap and Embed LLM Trajectories
- Open a new terminal and collect llm trajectories

```bash
$ cd {project_root}
$ sh ./script/exp_03_collect_llm.sh
```

- After bootstrapping, embed the trajectories

```bash
$ cd {project_root}
$ sh ./script/exp_02_embed_human_traj.sh
```

### Evaluate
- Open a new terminal and evaluate

```bash
$ cd {project_root}
$ sh ./script/exp_05_eval.sh
```

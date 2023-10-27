# Transformer Lens Starter Template

A simple starter template to get going with [Transformer
Lens](https://github.com/neelnanda-io/TransformerLens).

## Getting Started

1. Click "Use this template" from GitHub, to create your new repo.

   [![Static
   Badge](https://img.shields.io/badge/Use%20the%20template-rgb(31%2C%20136%2C%2061)?style=for-the-badge&logo=github)
   ](https://github.com/new?template_name=transformer-lens-starter-template&template_owner=alan-cooney)

2. Follow the steps below, depending on what system you're using.
3. Optionally edit `pyproject.toml` to change your project name, and then rename `src` (both in
   pyproject's `include` and the actual directory in your repo) to your new project package name.
   This has the effect of allowing you to import things as `import my_project_name.main` (i.e. it's
   like adding your `src` directory to `PATH`).

### System Setup

#### Mac (apple silicon)

Apple silicon doesn't play well with Docker, so instead we'll use the poetry virtual environment
(i.e. don't click yes if VSCode asks if you want to open your project in a devcontainer).

However, this means you'll need to install some dependencies:

1. Install Brew if you don't have it (package manager a bit like apt)

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. Install some dependencies:

   ```bash
   brew install python@3.9 python@3.10 python@3.11 node poetry pip virtualenv
   ```

3. Install [VSCode](https://code.visualstudio.com/) if you don't have it already
4. Clone your newly created repo onto your local system.
5. Open the folder in VSCode and select "no" to opening the devcontainer with Docker.
6. Install the dependencies with [Poetry](https://github.com/python-poetry/poetry), or pip.

   ```bash
   poetry config virtualenvs.in-project true # [Optional] Default to creating .venv in project dir
   poetry install --with dev,jupyter # Install
   poetry env use python3.11 # [Optional] use a specific python version
   ```

   ```bash
   pip install .
   ```

7. In VSCode set your default interpretor to the virtual environment (`CMD+SHIFT+P` then `>Python:
   Select Interpretor`. Choose the one in the virtual environment ('.venv: Poetry'). Then reload the
   window (`CMD+SHIFT+P` then `>Developer: Reload Window`).

#### Windows/Linux (local only)

1. Install the following:

   - [VSCode](https://code.visualstudio.com/)
   - [Remote Development VSCode
     Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
   - [Docker for Windows](https://docs.docker.com/desktop/install/windows-install/)

2. Clone your newly created repo onto your local system.
3. Open the folder in VSCode and click "yes" to opening the devcontainer with Docker.
4. In VSCode set your default interpretor to the virtual environment (`CMD+SHIFT+P` then `>Python:
   Select Interpretor`. Choose the one in the virtual environment ('.venv: Poetry'). Then reload the
   window (`CMD+SHIFT+P` then `>Developer: Reload Window`).

#### Vast AI

##### One Time Setup

1. Go to the [console](https://cloud.vast.ai/) and click "edit image & config". Choose the latest
   pytorch with cuda devel (e.g. `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`). Tick the box to run
   "interactive shell server, SSH". Then change your startup script to this:

   ```script
   env | grep _ >> /etc/environment; echo 'starting up'

   # Install GitHub CLI
   type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
   curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
   && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
   && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
   && sudo apt update \
   && sudo apt install gh -y

   # Install Python Versions
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        git \
        python3 \
        python3.11 \
        python3.12 \
        python3-dev \
        python3-distutils \
        python3-venv
   
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -
   echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
   export PATH="/root/.local/bin:$PATH"
   poetry config virtualenvs.in-project true
   ```

2. On your local machine, install VSCode](https://code.visualstudio.com/) and the [remote development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack).
3. Open the command box in VSCode (`CMD+SHIFT+P` or `CRTL+SHIFT+P`) and open "Add new SSH Host".
   Paste the SSH command in from VastAI (given after you click connect).

##### Use an Instance

1. Create an instance on the [instances](https://cloud.vast.ai/instances/) tab that has a `Mac CUDA` of
   at least the current CUDA your image (above) uses.  Click connect once it's ready.
2. Open the command box in VSCode (`CMD+SHIFT+P` or `CRTL+SHIFT+P`) and type `Remote SSH: Add New
   SSH Host`. Paste in the proxy ssh connect details from VastAI. Then open the command box again
   and this time connect to the host. Note if you've done this previously it's worth deleting the
   old one first (with `Open SSH Configuration File`).
3. Open a terminal once connected (in VSCode) and login to github with the command `gh auth login`
4. Clone your repo with `gh repo clone [your-new-repo-name]`
5. Open this folder with VSCode (File -> Open Folder). Click "no" if asked if you want to open the devcontainer.
6. Click yes to "do you want to install the recommended extensions..."
7. Open a new terminal and install all the package dependencies with `poetry install --with dev,jupyter`.
8. Get coding.

#### Remote SSH (not VastAI)

1. Setup any SSH keys needed to connect to your remote (host) box (e.g. [see
   this](https://vast.ai/faq#SSH) with VastAI)
2. Install locally (on your laptop connecting to the box) the following:
   - [VSCode](https://code.visualstudio.com/)
   - [Remote Development VSCode
     Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
3. Connect to the host box from within VSCode using [these
   instructions](https://code.visualstudio.com/docs/remote/ssh-tutorial#_connect-using-ssh).
4. Clone the repo you've already created (from the template) on this remote box.
5. Open the cloned repo folder with VSCode
6. Click "yes" to opening the devcontainer with Docker.
7. In VSCode set your default interpretor to the virtual environment (`CMD+SHIFT+P` then `>Python:
   Select Interpretor`. Choose the one in the virtual environment ('.venv: Poetry'). Then reload the
   window (`CMD+SHIFT+P` then `>Developer: Reload Window`).

## Troubleshooting

If you have any issues, just open an issue with your question. I'll try to respond as quickly as
possible, and also add the solution to the README.

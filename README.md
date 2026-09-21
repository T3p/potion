[![Python application](https://github.com/T3p/potion/actions/workflows/python-app.yml/badge.svg)](https://github.com/T3p/potion/actions/workflows/python-app.yml)

# POlicy optimizaTION

How to install (tested on Ubuntu 18.04.2):

```bash
apt-get update
apt-get install git python3 python3-pip
git clone https://github.com/T3p/potion.git
cd potion
pip3 install -e .
```

Optional integrations can be installed as extras:

```bash
pip3 install -e '.[mujoco]'
pip3 install -e '.[wandb]'
```

W&B logging is controlled by the episodic logger and is disabled by default.

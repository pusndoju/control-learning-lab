# Python simulations (Linux)

All Python code in this folder was written and tested on **Ubuntu 24.04 in a VMware Workstation virtual machine**, using one shared virtual environment. A native Ubuntu install or WSL2 also works: skip step 1.

| Folder | Book |
|---|---|
| [`mathematical-foundations-rl/`](mathematical-foundations-rl/) | *Mathematical Foundations of Reinforcement Learning*, Shiyu Zhao |

## Setup

### 1. Virtual machine (VMware only)

1. Install Ubuntu 24.04 in VMware Workstation and set **Network Adapter** to **NAT**.
2. Install the VMware guest tools:
   ```bash
   sudo apt update
   sudo apt install -y open-vm-tools open-vm-tools-desktop
   ```
3. *(Optional)* To edit the code on Windows and run it in the VM, share the folder: **VM → Settings → Options → Shared Folders → Always enabled → Add**. It appears under `/mnt/hgfs/<name>`. If it does not, mount it manually:
   ```bash
   sudo mkdir -p /mnt/hgfs
   sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
   ```
   To mount it at every boot, add this line to `/etc/fstab`:
   ```
   .host:/  /mnt/hgfs  fuse.vmhgfs-fuse  allow_other,defaults  0  0
   ```
4. **DNS fix.** If `ping 8.8.8.8` works but `ping google.com` fails with *Temporary failure in name resolution*:
   ```bash
   nmcli con show                                   # find the connection name, e.g. "Wired connection 1"
   sudo nmcli con mod "Wired connection 1" ipv4.dns "8.8.8.8 1.1.1.1" ipv4.ignore-auto-dns yes
   sudo nmcli con up "Wired connection 1"
   ```

### 2. Python environment

```bash
sudo apt install -y git python3-venv python3-tk      # python3-tk lets plt.show() open a window
python3 -m venv ~/.venvs/lab                         # one environment for the whole folder
~/.venvs/lab/bin/pip install --upgrade pip
~/.venvs/lab/bin/pip install numpy matplotlib "gymnasium[toy-text]"
```

`gymnasium[toy-text]` also installs `pygame`, which is needed for `render_mode='human'` (the FrozenLake window).

Check the installation:
```bash
~/.venvs/lab/bin/python -c "import gymnasium, matplotlib, numpy; print(gymnasium.__version__)"
```

### 3. Get the code

```bash
git clone https://github.com/pusndoju/control-learning-lab.git
cd control-learning-lab/linux
```

If you use a VMware shared folder, open that folder instead of cloning.

### 4. VS Code

```bash
sudo snap install code --classic
code --install-extension ms-python.python            # includes the Python Debugger (debugpy)
code .                                               # open THIS folder (linux/), where .vscode/ lives
```

`.vscode/settings.json` points to the interpreter `/home/tinbien/.venvs/lab/bin/python`. Replace `tinbien` with your user name, or pick it via **Ctrl+Shift+P → Python: Select Interpreter → ~/.venvs/lab**.

## Running a simulation

- **VS Code:** open any `.py` file and press **F5**. `.vscode/launch.json` runs the open file with its own folder as the working directory.
- **Terminal:**
  ```bash
  cd mathematical-foundations-rl
  ~/.venvs/lab/bin/python ch04_value_iteration_frozenlake.py
  ```

Each file runs with the defaults in its `if __name__ == '__main__':` block. Edit the variables there to turn rendering on or off, switch `is_slippery`, or change the number of episodes.

## File template

Every simulation follows the same tutorial-style layout:

```python
class Algorithm:
    gamma = 0.9            # hyperparameters at the top, one per line, with their meaning
    ...
    def train(self, episodes, render=False, is_slippery=False): ...   # or solve() for model-based methods
    def test(self, episodes, render=True, is_slippery=False): ...

if __name__ == '__main__':
    ...                    # runnable defaults
```

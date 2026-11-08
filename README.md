# Vừa đọc sách vừa học mô phỏng: RL & điều khiển bằng Python

Repo ghi lại quá trình tự học: đọc từng chương sách, rồi tự gõ lại code mô phỏng các thuật toán chính để thấy chúng chạy thật và kiểm chứng điều kiện hội tụ trong sách.

Cảm hứng ban đầu đến từ cuốn **Mathematical Foundations of Reinforcement Learning** của tác giả **Shiyu Zhao**, cùng các video bài giảng và hướng dẫn trên YouTube. Repo chỉ chứa code mô phỏng và ghi chú tự viết, không chứa nội dung sách. Hãy tìm đọc bản gốc của tác giả.

## Sách đã có

| Thư mục | Sách | Nội dung |
|---|---|---|
| [`mathematical-foundations-rl/`](mathematical-foundations-rl/) | *Mathematical Foundations of Reinforcement Learning*, Shiyu Zhao | Chương 1–7: Bellman equation, value/policy iteration, Monte Carlo, stochastic approximation, TD/Sarsa/Q-learning |

Sẽ thêm sách khác dần, mỗi sách một thư mục, có README riêng.

## Quy ước code

- **Một thuật toán = một file**, tên dạng `chXX_<thuật toán>_<môi trường>.py`.
- Mỗi file là một class: hyperparameter ở đầu class, `train()` / `solve()`, `test()`, cùng tùy chọn `render` để bật/tắt cửa sổ hình.
- Dùng môi trường có sẵn của [Gymnasium](https://gymnasium.farama.org/) (FrozenLake, ...). Bài nào không cần môi trường thì chỉ dùng numpy.
- Dòng code nào là thuật toán thì gắn tham chiếu ngay trong comment, dạng `[Book eq. 7-18]`, `[Book Alg. 4.1]`, để tiện đối chiếu với sách.
- Kết quả (hình, Q table) lưu vào `results/` cạnh file. Thư mục này không đưa lên Git, chạy lại là có.

## Cài môi trường

Code được viết và chạy trên **Ubuntu 24.04 trong máy ảo VMware Workstation**. Ubuntu chạy thật hoặc WSL2 cũng dùng được, khi đó bỏ qua bước 1.

### 1. Máy ảo và thư mục chia sẻ (chỉ khi dùng VMware)

1. Cài Ubuntu 24.04 trong VMware Workstation. Mục **Network Adapter** chọn **NAT**.
2. Cài VMware Tools cho máy khách:
   ```bash
   sudo apt update
   sudo apt install -y open-vm-tools open-vm-tools-desktop
   ```
3. (Tùy chọn) Chia sẻ thư mục code từ Windows: **VM → Settings → Options → Shared Folders → Always enabled → Add**. Thư mục sẽ xuất hiện ở `/mnt/hgfs/<tên>`. Nếu chưa thấy thì mount tay:
   ```bash
   sudo mkdir -p /mnt/hgfs
   sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
   ```
   Để tự mount mỗi lần khởi động, thêm dòng sau vào `/etc/fstab`:
   ```
   .host:/  /mnt/hgfs  fuse.vmhgfs-fuse  allow_other,defaults  0  0
   ```
4. Nếu `ping 8.8.8.8` được mà `ping google.com` báo *Temporary failure in name resolution*, nghĩa là lỗi DNS. Sửa như sau:
   ```bash
   nmcli con show                                  # lấy tên kết nối, vd "Wired connection 1"
   sudo nmcli con mod "Wired connection 1" ipv4.dns "8.8.8.8 1.1.1.1" ipv4.ignore-auto-dns yes
   sudo nmcli con up "Wired connection 1"
   ```

### 2. Python và thư viện

```bash
sudo apt install -y git python3-venv python3-tk     # python3-tk: để plt.show() mở được cửa sổ hình
python3 -m venv ~/.venvs/lab                        # một môi trường dùng chung cho cả repo
~/.venvs/lab/bin/pip install --upgrade pip
~/.venvs/lab/bin/pip install numpy matplotlib "gymnasium[toy-text]"
```

`gymnasium[toy-text]` cài kèm `pygame`, dùng cho chế độ `render_mode='human'` (cửa sổ FrozenLake).

Kiểm tra:
```bash
~/.venvs/lab/bin/python -c "import gymnasium, matplotlib, numpy; print(gymnasium.__version__)"
```

### 3. Lấy code

```bash
git clone https://github.com/pusndoju/rl-book-simulations.git
cd rl-book-simulations
```

Dùng thư mục chia sẻ của VMware thì mở thẳng thư mục đó, không cần clone.

### 4. VS Code

```bash
sudo snap install code --classic
code --install-extension ms-python.python           # kèm Python Debugger (debugpy)
code .                                              # mở thư mục gốc của repo
```

`.vscode/settings.json` trỏ tới interpreter `/home/tinbien/.venvs/lab/bin/python`. Hãy đổi `tinbien` thành tên user của bạn, hoặc chọn lại bằng **Ctrl+Shift+P → Python: Select Interpreter → ~/.venvs/lab**.

## Chạy

- **VS Code:** mở một file `.py` rồi nhấn **F5**. Cấu hình trong `.vscode/launch.json` chạy file đang mở, với thư mục làm việc là thư mục của file.
- **Terminal:**
  ```bash
  cd mathematical-foundations-rl
  ~/.venvs/lab/bin/python ch04_value_iteration_frozenlake.py
  ```

Mỗi file chạy được ngay với giá trị mặc định ở khối `if __name__ == '__main__':`. Muốn bật/tắt cửa sổ hình, trơn/không trơn (`is_slippery`) hay đổi số episode thì sửa các biến ở khối đó.

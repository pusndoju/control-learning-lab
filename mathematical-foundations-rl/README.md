# Mathematical Foundations of Reinforcement Learning: mô phỏng chương 1–7

Code tự viết khi đọc sách *Mathematical Foundations of Reinforcement Learning* của **Shiyu Zhao**. Mỗi thuật toán chính trong sách là một file, comment trong code ghi số phương trình hoặc thuật toán tương ứng (`[Book eq. x-y]`, `[Book Alg. x.y]`).

Cài môi trường và cách chạy: xem [README ở gốc repo](../README.md).

## Môi trường

- Các chương 2, 4, 5 và 7 dùng **`FrozenLake-v1` 4×4** của Gymnasium, γ = 0.9. Đặt `is_slippery=True` để chuyển trạng thái thành ngẫu nhiên (đi đúng hướng 1/3, trượt sang mỗi bên 1/3).
- Chương 6 chỉ dùng numpy.
- Các thuật toán dựa trên model (chương 2, 4) đọc p(s'|s,a) từ `env.unwrapped.P`.
- Các thuật toán model-free (chương 5, 7) chỉ dùng `env.step()`. Model chỉ được dùng để tính v\* làm đáp án đối chiếu.

**Khác với grid world 5×5 trong sách:** ở FrozenLake, đích và hố là trạng thái kết thúc, phần thưởng là +1 khi tới đích và 0 ở mọi nơi khác (không phạt −1). Vì vậy v ≤ 1, và các con số không trùng với hình trong sách, nhưng mọi công thức vẫn giữ nguyên.

## Danh sách file

| Chương | File | Thuật toán | Câu hỏi mô phỏng trả lời |
|---|---|---|---|
| 1 | (không có) | Khái niệm cơ bản | (chỉ ghi chú) |
| 2 | `ch02_policy_evaluation_frozenlake.py` | Bellman equation: nghiệm đóng (I−γP_π)⁻¹r_π và nghiệm lặp (2.11) | Hai cách có cho cùng v_π? Sai số có giảm theo γ^k (Box 2.1)? |
| 3 | (gộp vào file value iteration) | BOE, contraction mapping (Thm 3.1–3.5) | |
| 4 | `ch04_value_iteration_frozenlake.py` | Value iteration (Alg 4.1) | v_k hội tụ hình học từ mọi v₀? |
| 4 | `ch04_policy_iteration_frozenlake.py` | Policy iteration (Alg 4.2) | v_{π_k} có tăng đơn điệu (Thm 4.1)? |
| 4 | `ch04_truncated_policy_iteration_frozenlake.py` | Truncated PI (Alg 4.3) | Đánh đổi giữa số vòng ngoài và số lần quét đánh giá (Fig 4.5) |
| 5 | `ch05_mc_basic_frozenlake.py` | MC Basic (Alg 5.1) | Độ dài episode quá ngắn thì sai thế nào (Fig 5.4)? |
| 5 | `ch05_mc_exploring_starts_frozenlake.py` | MC Exploring Starts (Alg 5.2) | Every-visit dùng mẫu hiệu quả hơn ra sao? |
| 5 | `ch05_mc_epsilon_greedy_frozenlake.py` | MC ε-Greedy (Alg 5.3) | Policy tối ưu trong Π_ε khác policy tối ưu trong Π |
| 6 | `ch06_robbins_monro_root_finding.py` | Robbins-Monro (6.5) | Vi phạm từng điều kiện của Thm 6.1 thì chuyện gì xảy ra? |
| 6 | `ch06_sgd_mean_estimation.py` | BGD / MBGD / SGD (Sec 6.4.4) | Kích thước mini-batch ảnh hưởng tốc độ hội tụ thế nào (Fig 6.5)? |
| 7 | `ch07_td_state_value_frozenlake.py` | TD(0) (7.1) | α hằng số so với α = 1/N(s) (Thm 7.1) |
| 7 | `ch07_sarsa_frozenlake.py` | Sarsa (Alg 7.1) | Sarsa chỉ tìm đường từ ô xuất phát |
| 7 | `ch07_nstep_sarsa_frozenlake.py` | n-step Sarsa (7.17) | n = 1, 3, 10: bias và variance |
| 7 | `ch07_q_learning_frozenlake.py` | Q-learning off-policy (Alg 7.3), on-policy (Alg 7.2) | Học policy tối ưu từ dữ liệu của behavior policy ngẫu nhiên |

Nên chạy theo thứ tự trong bảng. Mỗi chương xây trên chương trước: chương 2 cho policy evaluation, chương 4 thêm policy improvement, chương 5 bỏ model, chương 6 cho công cụ (RM/SGD), chương 7 ghép tất cả lại thành TD.

## Kết quả tham khảo

FrozenLake không trơn, trừ khi ghi khác.

- **Ch 2:** v_π(0) = 0.590 với policy "good" và ≈ 0.004 với policy random. Nghiệm lặp khớp nghiệm đóng tới sai số ~1e-8.
- **Ch 4:** VI hội tụ sau 7 vòng. PI cho dãy Σv tăng dần 0 → 1 → 2.8 → … → 8.44. Cả ba thuật toán cho cùng một π\*. Khi trơn: truncated PI với j = 1 / 3 / 10 / 100 cần 92 / 32 / 11 / 5 vòng ngoài, và π\* tới đích khoảng 73%.
- **Ch 5:** MC Basic cần `episode_length` ≥ 6 (số bước tối thiểu từ ô 0 tới đích), nếu không thì sai số kẹt ở 0.9 / 0.73 / 0.59. ε-greedy với ε = 0.5 chọn hướng R ở ô 0, khác π\*.
- **Ch 6:** với g(w) = w³ − 5, các bước a_k = 1/k và 1/k² **phân kỳ** vì g' không bị chặn trên (vi phạm điều kiện a của Thm 6.1). Bước 0.2/k thì hội tụ về 1.71. Lưu ý: sách ghi Fig 6.3 hội tụ với w₁ = 0, a_k = 1/k, nhưng mình chạy thử nhiều seed đều phân kỳ.
- **Ch 7:** Q-learning off-policy với behavior ngẫu nhiên hội tụ về v\* (RMSE → 0). Behavior ε = 0.1 bị kẹt vì khám phá quá ít. Khác Fig 7.5 trong sách: ở đây ε = 0.5 nhanh hơn ε = 1, vì FrozenLake kết thúc khi rơi xuống hố nên đi ngẫu nhiên hiếm tới được các ô gần đích.

## Tóm tắt điều kiện hội tụ cần nhớ

| Thuật toán | Điều kiện |
|---|---|
| Policy evaluation, VI | γ < 1, nên ánh xạ là co với hệ số γ. Hội tụ hình học từ mọi giá trị đầu |
| PI, truncated PI | Giá trị tăng đơn điệu (Thm 4.1, Prop 4.1) |
| MC | Đủ episode, episode đủ dài, mọi (s,a) được thăm (exploring starts hoặc ε-greedy) |
| RM, SGD, TD, Sarsa, Q-learning | Σα = ∞ và Σα² < ∞ cho **từng** trạng thái hoặc cặp (s,a), nghĩa là mọi (s,a) phải được thăm vô hạn lần. α hằng số chỉ hội tụ theo kỳ vọng |

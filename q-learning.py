import numpy as np

def discretize(state):
    bird_y, dy_b, dy_t, dx, vel = state

    # ví dụ: lượng tử hóa theo kích thước màn hình và khoảng cách thực tế
    y_bin     = min(int(bird_y // 50), 11)
    dy_b_bin  = min(int((dy_b + 300) // 60), 9)
    dy_t_bin  = min(int((dy_t + 300) // 60), 9)
    dx_bin    = min(int(dx // 60), 9)
    vel_bin   = min(int((vel + 10) // 3), 6)

    # ánh xạ thành 1 chỉ số duy nhất
    index = (((((y_bin * 10 + dy_b_bin) * 10 + dy_t_bin) * 10 + dx_bin) * 7) + vel_bin)
    return index


n_states = 84000
n_actions = 2

Q_table = np.zeros((n_states, n_actions))

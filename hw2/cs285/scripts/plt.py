import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib.ticker import FuncFormatter

def load_tb_scalars(logdir, scalar_name):
    """
    读取单个实验的 TB scalar，返回 (steps, values)
    这里 steps 是日志记录的 global_step，不用于横轴
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]      # 迭代编号
    values = [e.value for e in events]    # 标量的真实值
    return np.array(steps), np.array(values)

def load_curve_xy(logdir, y_tag="Eval_AverageReturn", x_tag="Train_EnvstepsSoFar"):
    """
    返回横纵坐标，横轴用 Train_EnvstepsSoFar 的 value，纵轴用 y_tag 的 value
    """
    _, y_vals = load_tb_scalars(logdir, y_tag)
    _, x_vals = load_tb_scalars(logdir, x_tag)
    # 对齐长度（有时 Eval_AverageReturn 记录频率较低）
    k = min(len(x_vals), len(y_vals))
    return x_vals[:k], y_vals[:k]

def plot_experiments(exp_dict, title, savepath=None,
                     y_tag="Eval_AverageReturn", x_tag="Train_EnvstepsSoFar",
                     xlabel="Environment Steps", ylabel="Average Return"):
    """
    exp_dict: {label: logdir}
    横轴用 Train_EnvstepsSoFar 的 value，并转换为千步
    """
    plt.figure(figsize=(8,5))
    for label, logdir in exp_dict.items():
        x, y = load_curve_xy(logdir, y_tag=y_tag, x_tag=x_tag)
        plt.plot(x, y, label=label)
    formatter = FuncFormatter(lambda x, _: f"{int(x/1000)}k")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"图已保存到 {savepath}")
    else:
        plt.show()

# 小 batch
exp_dict_small = {
    "Trajectory": "data/q2_pg_cartpole_CartPole-v0_28-09-2025_04-33-00",
    "Trajectory + Norm": "data/q2_pg_cartpole_na_CartPole-v0_28-09-2025_03-30-15",
    "Reward-to-Go": "data/q2_pg_cartpole_rtg_CartPole-v0_28-09-2025_03-27-00",
    "Reward-to-Go + Norm": "data/q2_pg_cartpole_rtg_na_CartPole-v0_28-09-2025_03-33-26"
}

# 大 batch
exp_dict_large = {
    "Trajectory": "data/q2_pg_cartpole_lb_CartPole-v0_28-09-2025_03-39-03",
    "Trajectory + Norm": "data/q2_pg_cartpole_lb_na_CartPole-v0_28-09-2025_04-01-02",
    "Reward-to-Go": "data/q2_pg_cartpole_lb_rtg_CartPole-v0_28-09-2025_03-47-55",
    "Reward-to-Go + Norm": "data/q2_pg_cartpole_lb_rtg_na_CartPole-v0_28-09-2025_04-09-34"
}

plot_experiments(
    exp_dict_large,
    title="CartPole (Large Batch): Trajectory vs Reward-to-Go & Normalization",
    savepath="cartpole_large_batch.png"
)

plot_experiments(
    exp_dict_small,
    title="CartPole (Small Batch): Trajectory vs Reward-to-Go & Normalization",
    savepath="cartpole_small_batch.png"
)

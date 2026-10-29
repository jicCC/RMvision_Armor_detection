import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- 读数据 ----------
rows = []
with open("state.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

t  = np.array([float(r["t"]) for r in rows])
cx = np.array([float(r["cx"]) for r in rows])
cy = np.array([float(r["cy"]) for r in rows])
za = np.array([float(r["za"]) for r in rows])
th = np.array([float(r["theta"]) for r in rows])
om = np.array([float(r["omega"]) for r in rows])
rr = np.array([float(r["r"]) for r in rows])

# 速度
vx = np.array([float(r["vx"]) for r in rows])
vy = np.array([float(r["vy"]) for r in rows])
vz = np.array([float(r["vz"]) for r in rows])

# 朝向折回 (-pi, pi]
thw = (th + np.pi) % (2 * np.pi) - np.pi

# 标准差
sd_cx    = np.sqrt(np.array([float(r["P_cx"]) for r in rows]))
sd_cy    = np.sqrt(np.array([float(r["P_cy"]) for r in rows]))
sd_theta = np.sqrt(np.array([float(r["P_theta"]) for r in rows]))
rz = np.array([float(r["raw_z"]) for r in rows])

# 用 matched 把 z_a 换算成"当前这块板的预测高度"
dh = np.array([0.0, -0.05, 0.0, -0.05])
mj = np.array([int(r["matched"]) for r in rows])
pred_z = za + np.array([dh[j] if j >= 0 else 0.0 for j in mj])

fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
fig.suptitle("EKF vehicle state estimation", fontsize=14)

items = [
    (axes[0][0], cx,    sd_cx,    "c_x  [m]",      "position x"),
    (axes[0][1], cy,    sd_cy,    "c_y  [m]",      "position y"),
    (axes[0][2], za,    None,     "z_a  [m]",      "armor height"),
    (axes[1][0], vx,    None,     "v_x  [m/s]",    "velocity x"),
    (axes[1][1], vy,    None,     "v_y  [m/s]",    "velocity y"),
    (axes[1][2], vz,    None,     "v_z  [m/s]",    "velocity z"),
    (axes[2][0], thw,   sd_theta, "theta [rad]",   "orientation"),
    (axes[2][1], om,    None,     "omega [rad/s]", "angular rate"),
    (axes[2][2], rr,    None,     "r  [m]",        "armor radius"),
]

for ax, y, sd, ylab, title in items:
    ax.plot(t, y, lw=1.0, color="tab:blue")
    if sd is not None:
        ax.fill_between(t, y - 3*sd, y + 3*sd, alpha=0.25, color="tab:orange",
                        label="±3σ")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)

for ax in axes[2]:
    ax.set_xlabel("t [s]")

axes[2][2].set_ylim(0.24, 0.26)

# ---- armor height 子图：模型预测 vs 原始观测 ----------
axes[0][2].plot(t, pred_z, lw=0.9, alpha=0.85, color="tab:red", label="model: z_a + dh")
axes[0][2].plot(t, rz,     lw=0.5, alpha=0.40, color="gray",   label="measurement")
axes[0][2].legend(fontsize=7, loc="lower right")

# ---- z 通道残差统计 ----------
res = pred_z - rz
rms = np.sqrt(np.mean(res ** 2))
print(f"z channel: RMS residual = {rms*100:.2f} cm, max = {np.max(np.abs(res))*100:.2f} cm")

# ---- 单独的残差图 ----------
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(t, res * 100.0, lw=0.7, color="tab:purple")
ax2.axhline(0.0, color="k", lw=0.5)
ax2.set_xlabel("t [s]")
ax2.set_ylabel("residual [cm]")
ax2.set_title("z residual:  (z_a + dh[matched]) - measurement")
ax2.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig("z_residual.png", dpi=130)
print("saved z_residual.png")

fig.tight_layout()
fig.savefig("states.png", dpi=130)
print("saved states.png")
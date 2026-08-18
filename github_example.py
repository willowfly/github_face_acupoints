import os, pickle, numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from PIL import Image # check of pillow

def load_one_person(file='./data/person_01.pkl'):
    with open(file=file, mode='rb') as fin:
        data = pickle.load(fin)
    nframes = data['nframes']
    landmarks = data['landmarks']
    acupoints = data['acupoints']
    landmarks_f = data['landmarks_f']  # flipped data
    acupoints_f = data['acupoints_f']
    return nframes, landmarks, acupoints, landmarks_f, acupoints_f


def demo(id=1):
    filename = os.path.join('./data', f'person_{id:02d}.pkl')
    nframes, landmarks, acupoints, _, _ = load_one_person(file=filename)
    # the canvas
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(960 - 540, 960 + 540)   # x: [420, 1500]
    ax.set_ylim(0, 1080)                # y: [0‑1080]
    ax.set_aspect("equal")              # axis equal
    ax.invert_yaxis()  # flip the y-axis
    ax.set_title(f"Person {id:02d}, total frames: {nframes}")

    lm_scatter = ax.scatter([], [], s=4,  c="k", marker=".")    # landmarks 黑色小点
    ac_scatter = ax.scatter([], [], s=10, c="r", marker="o")   # acupoints 红色圆点
    title_text = ax.text(0.02, 0.98, f"Person {id:02d}, total frames: {nframes}",
                         transform=ax.transAxes, va="top", fontsize=12)
    
    def update(frame_idx):
        lm_pts = landmarks[frame_idx]  # (478,2)
        ac_pts = acupoints[frame_idx]  # (13,2)
        lm_scatter.set_offsets(lm_pts)
        ac_scatter.set_offsets(ac_pts)
        title_text.set_text(f"Person {id:02d} | frame {frame_idx:03d}/{nframes-1:03d}")
        return lm_scatter, ac_scatter, title_text

    ani = FuncAnimation(
        fig, update, frames=nframes, interval=20, blit=True
    )
    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":
    anim = demo(id=1)
    anim.save("face_landmark.gif", writer="pillow", fps=40)

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_env import make_sim_env

OUT_DIR = "paper_figures/task_setup"
os.makedirs(OUT_DIR, exist_ok=True)

# 创建仿真环境
env = make_sim_env("sim_insertion_scripted")
ts = env.reset()

images = ts.observation["images"]
print("Available camera keys:", images.keys())

# 你的三视角顺序
camera_order = ["left_wrist", "top", "right_wrist"]

# 保存单独视角
single_paths = []
for cam in camera_order:
    if cam not in images:
        print(f"[Warning] camera {cam} not found, available keys = {list(images.keys())}")
        continue

    img = images[cam]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    path = os.path.join(OUT_DIR, f"{cam}.png")
    Image.fromarray(img).save(path)
    single_paths.append(path)
    print("Saved:", path)

# 拼接三视角图：[left_wrist | top | right_wrist]
pil_imgs = [Image.open(p).convert("RGB") for p in single_paths]
w, h = pil_imgs[0].size

label_h = 42
combined = Image.new("RGB", (w * len(pil_imgs), h + label_h), "white")
draw = ImageDraw.Draw(combined)

for i, (cam, im) in enumerate(zip(camera_order, pil_imgs)):
    x = i * w
    combined.paste(im, (x, label_h))
    draw.text((x + 12, 10), cam, fill=(0, 0, 0))

combined_path = os.path.join(OUT_DIR, "three_view_frame.png")
combined.save(combined_path)
print("Saved:", combined_path)

print("\nDone. Files are in:", OUT_DIR)

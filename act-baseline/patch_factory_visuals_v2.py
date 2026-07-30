from pathlib import Path

p = Path("sim_env.py")

# 先恢复备份，避免重复 patch
backup = Path("sim_env_backup_before_factory.py")
if backup.exists():
    p.write_text(backup.read_text())
    print("Restored sim_env.py from sim_env_backup_before_factory.py")
else:
    print("No backup found, patching current sim_env.py directly")

s = p.read_text()

insert = r'''

# ================= Factory-style visual robustness test V2 =================
# Natural image-level visual perturbations only.
# This does NOT change physics, object geometry, reward, or robot dynamics.
try:
    import cv2
except Exception:
    cv2 = None

def _factory_mode():
    return os.environ.get("FACTORY_MODE", "clean").lower()

def _factory_seed():
    try:
        return int(os.environ.get("FACTORY_SEED", "0"))
    except Exception:
        return 0

def _stable_cam_id(cam_name):
    return sum(ord(c) for c in str(cam_name))

def _apply_factory_visuals(img, cam_name):
    mode = _factory_mode()
    if mode in ["", "none", "clean", "default"]:
        return img

    out = img.astype(np.float32)
    h, w = out.shape[:2]
    rng = np.random.default_rng(_factory_seed() + _stable_cam_id(cam_name))

    # Mild: industrial lighting / color temperature shift.
    if mode in ["mild", "medium", "hard", "factory_mild", "factory_medium", "factory_hard"]:
        contrast = 0.92
        brightness = 8.0
        out = out * contrast + brightness

        # RGB image: slightly cool industrial light.
        out[:, :, 0] *= 0.94
        out[:, :, 1] *= 0.98
        out[:, :, 2] *= 1.06

        # Smooth top-down illumination gradient.
        grad_y = np.linspace(0.90, 1.06, h).reshape(h, 1, 1)
        out *= grad_y

    # Medium: mild + vignette + soft random shadow.
    if mode in ["medium", "hard", "factory_medium", "factory_hard"]:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        dist = ((xx - cx) ** 2 / (cx ** 2) + (yy - cy) ** 2 / (cy ** 2))
        vignette = 1.0 - 0.18 * np.clip(dist, 0, 1)
        out *= vignette[..., None]

        shadow_center = rng.uniform(0.25, 0.75) * w
        shadow_width = rng.uniform(0.18, 0.35) * w
        shadow = 1.0 - 0.12 * np.exp(-((xx - shadow_center) ** 2) / (2 * shadow_width ** 2))
        out *= shadow[..., None]

        out += rng.normal(0, 3.0, size=out.shape)

    # Hard: medium + blur + cable-like thin occlusion lines.
    if mode in ["hard", "factory_hard"]:
        if cv2 is not None:
            out = cv2.GaussianBlur(out, (3, 3), 0)

            for _ in range(2):
                x0 = int(rng.integers(0, w))
                x1 = int(np.clip(x0 + rng.integers(-80, 80), 0, w - 1))
                y0 = int(rng.integers(0, h))
                y1 = int(np.clip(y0 + rng.integers(80, 180), 0, h - 1))
                cv2.line(out, (x0, y0), (x1, y1), (25, 25, 25), thickness=3)

        out += rng.normal(0, 5.0, size=out.shape)

    return np.clip(out, 0, 255).astype(np.uint8)

def _render_factory(physics, camera_id, height=480, width=640):
    img = physics.render(height=height, width=width, camera_id=camera_id)
    return _apply_factory_visuals(img, camera_id)
# ========================================================================
'''

marker = "BOX_POSE = [None] # to be changed from outside"
if marker not in s:
    raise RuntimeError("Cannot find BOX_POSE marker in sim_env.py")

s = s.replace(marker, marker + insert)

for cam in ["left_wrist", "top", "right_wrist"]:
    old = f"physics.render(height=480, width=640, camera_id='{cam}')"
    new = f"_render_factory(physics, '{cam}', height=480, width=640)"
    s = s.replace(old, new)

p.write_text(s)
print("Patched sim_env.py with natural factory visual perturbation V2.")

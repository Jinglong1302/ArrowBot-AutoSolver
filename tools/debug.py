import cv2
import numpy as np
import subprocess

def detect_arrowheads_multiscale(binary_roi, base_templates, base_dims, threshold=0.72):
    """
    Scans the board across multiple scales to handle varying zoom levels 
    and grid resolutions between different stages.
    """
    tw_base, th_base = base_dims
    raw_candidates = []

    # Test scales: 75%, 85%, 95%, 100%, 105%, 115%, 125%
    scales = np.linspace(0.75, 1.25, 7)
    
    for scale in scales:
        scaled_w = int(tw_base * scale)
        scaled_h = int(th_base * scale)
        if scaled_w < 5 or scaled_h < 5:
            continue

        for name, (tmpl, direction) in base_templates.items():
            # Resize template to test scale
            scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            _, scaled_tmpl = cv2.threshold(scaled_tmpl, 127, 255, cv2.THRESH_BINARY)
            
            res = cv2.matchTemplate(binary_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                score = float(res[pt[1], pt[0]])
                cx = pt[0] + scaled_w // 2
                cy = pt[1] + scaled_h // 2
                raw_candidates.append((cx, cy, direction, name, score, scaled_w, scaled_h))

    # Sort all scale detections by score descending
    raw_candidates.sort(key=lambda item: item[4], reverse=True)

    # Non-Maximum Suppression across all scales
    filtered_heads = []
    effective_dims = base_dims

    for cand in raw_candidates:
        cx, cy, direction, name, score, sw, sh = cand
        min_dist = max(sw, sh)
        if all((cx - fx) ** 2 + (cy - fy) ** 2 > (min_dist * 0.8) ** 2 for fx, fy, _, _, _ in filtered_heads):
            filtered_heads.append((cx, cy, direction, name, (sw, sh)))
            effective_dims = (sw, sh)

    # Clean output list format: (cx, cy, direction, name)
    clean_heads = [(h[0], h[1], h[2], h[3]) for h in filtered_heads]
    return clean_heads, effective_dims


def inspect_detections(binary_img, heads, arrow_dims, x_offset=0, y_offset=0):
    """
    Casts rays forward from each detected arrowhead, masking self-collision,
    and visualizes unblocked (green) vs blocked (red) arrows with indices.
    """
    h, w = binary_img.shape
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    tw, th = arrow_dims
    clearance_radius = max(tw, th) // 2 + 4

    playable = []

    for idx, (cx, cy, (dx, dy), name) in enumerate(heads):
        # Mask out own arrowhead body to prevent self-collision
        clearance_mask = binary_img.copy()
        cv2.circle(clearance_mask, (cx, cy), clearance_radius, 0, -1)

        # Cast ray forward
        start_dist = clearance_radius + 2
        ray_x = cx + dx * start_dist
        ray_y = cy + dy * start_dist
        is_blocked = False
        corridor_pts = []

        while 0 <= ray_x < w and 0 <= ray_y < h:
            corridor_pts.append((int(ray_x), int(ray_y)))
            px, py = int(ray_x), int(ray_y)

            # Sample narrow window across trajectory
            sample = clearance_mask[
                max(0, py - 1) : min(h, py + 2),
                max(0, px - 1) : min(w, px + 2)
            ]
            if np.any(sample > 0):
                is_blocked = True
                break

            ray_x += dx * 2
            ray_y += dy * 2

        color = (0, 255, 0) if not is_blocked else (0, 0, 255)
        for pt in corridor_pts:
            cv2.circle(vis, pt, 1, color, -1)

        # Draw node marker and index
        cv2.circle(vis, (cx, cy), 4, color, -1)
        cv2.putText(vis, str(idx), (cx - 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        if not is_blocked and len(corridor_pts) > 0:
            playable.append((idx, cx + x_offset, cy + y_offset, name))

    cv2.imwrite("debug_corridors.png", vis)
    print(f"Total Arrowheads Evaluated: {len(heads)}")
    print(f"Playable Moves Ready to Tap: {len(playable)}")
    for p_idx, sx, sy, direction in playable:
        print(f"-> Arrow #{p_idx} at Screen Coords ({sx}, {sy}) pointing {direction}")

    return playable

def capture_screenshot():
    cmd = ["adb", "exec-out", "screencap", "-p"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ADB Error: {err.decode('utf-8')}")
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

def load_templates(template_path="arrow_up_template.png"):
    """
    Loads base template (pointing LEFT) and precomputes all 4 rotations.
    """
    tmpl_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl_img is None:
        raise FileNotFoundError(f"Template '{template_path}' missing.")
    _, tmpl_base = cv2.threshold(tmpl_img, 80, 255, cv2.THRESH_BINARY_INV)
    th, tw = tmpl_base.shape

    templates = {
        "LEFT":  (tmpl_base, (-1, 0)),
        "UP":    (cv2.rotate(tmpl_base, cv2.ROTATE_90_CLOCKWISE), (0, -1)),
        "RIGHT": (cv2.rotate(tmpl_base, cv2.ROTATE_180), (1, 0)),
        "DOWN":  (cv2.rotate(tmpl_base, cv2.ROTATE_90_COUNTERCLOCKWISE), (0, 1)),
    }
    return templates, (tw, th)
    
def main():
    frame = capture_screenshot()
    if frame is None:
        raise FileNotFoundError("Could not load screen capture. Ensure phone is connected via ADB.")

    h, w, _ = frame.shape

    # Crop coordinates matching the maze bounds
    y1, y2 = int(h * 0.20), int(h * 0.95)
    x1, x2 = int(w * 0.01), int(w * 0.99)
    roi = frame[y1:y2, x1:x2]

    # Binarize: isolate dark lines and eliminate background paper grid
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("step1_clean_binary.png", binary)

    # 1. Load base template dictionary and dimensions
    base_templates, base_dims = load_templates("arrow_up_template.png")

    # 2. Detect heads passing the preloaded dictionary and dimensions
    heads, arrow_dims = detect_arrowheads_multiscale(
        binary_roi=binary,
        base_templates=base_templates,
        base_dims=base_dims,
        threshold=0.72
    )

    # 3. Verify corridors and export debug visualization
    inspect_detections(binary, heads, arrow_dims, x_offset=x1, y_offset=y1)


if __name__ == "__main__":
    main()
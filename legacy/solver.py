import subprocess
import time
import cv2
import numpy as np

def capture_screen():
    """Captures screenshot directly from ADB buffer into OpenCV BGR."""
    cmd = ["adb", "exec-out", "screencap", "-p"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ADB Error: {err.decode('utf-8')}")
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def send_tap(x, y):
    """Executes a touch event at screen coordinates (x, y)."""
    subprocess.run(["adb", "shell", "input", "tap", str(int(x)), str(int(y))], check=True)


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


def detect_arrowheads_multiscale(binary_roi, base_templates, base_dims, min_heads_expected=4):
    """
    Dynamically scales the confidence threshold based on board characteristics:
    1. Collects raw correlation scores across multiple scales.
    2. Uses peak detection score to determine an adaptive cutoff.
    3. Falls back gracefully from strict (0.75) down to relaxed (0.58).
    """
    tw_base, th_base = base_dims
    raw_candidates = []

    scales = np.linspace(0.75, 1.25, 7)

    # 1. Gather all candidates across scales and rotations
    for scale in scales:
        scaled_w = int(tw_base * scale)
        scaled_h = int(th_base * scale)
        if scaled_w < 5 or scaled_h < 5:
            continue

        for name, (tmpl, direction) in base_templates.items():
            scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            _, scaled_tmpl = cv2.threshold(scaled_tmpl, 127, 255, cv2.THRESH_BINARY)

            res = cv2.matchTemplate(binary_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            
            # Use a broad floor to collect candidate peaks
            loc = np.where(res >= 0.55)

            for pt in zip(*loc[::-1]):
                score = float(res[pt[1], pt[0]])
                cx = pt[0] + scaled_w // 2
                cy = pt[1] + scaled_h // 2
                raw_candidates.append((cx, cy, direction, name, score, scaled_w, scaled_h))

    if not raw_candidates:
        return [], base_dims

    # Sort descending by match score
    raw_candidates.sort(key=lambda item: item[4], reverse=True)

    # 2. Determine Dynamic Cutoff based on peak score
    best_score = raw_candidates[0][4]
    
    # If board has pristine matches (~0.85), keep cutoff tight (~0.72)
    # If best match is lower (~0.70), allow cutoff to drop (~0.60)
    adaptive_cutoff = max(0.58, min(0.72, best_score * 0.85))

    # 3. Non-Maximum Suppression with adaptive filtering
    filtered_heads = []
    effective_dims = base_dims

    for cand in raw_candidates:
        cx, cy, direction, name, score, sw, sh = cand
        
        # Discard anything below the adaptive cutoff
        if score < adaptive_cutoff:
            continue

        min_dist = max(sw, sh)
        # Suppress spatial duplicates across all scales
        if all((cx - fx) ** 2 + (cy - fy) ** 2 > (min_dist * 0.8) ** 2 for fx, fy, _, _, _ in filtered_heads):
            filtered_heads.append((cx, cy, direction, name, (sw, sh)))
            effective_dims = (sw, sh)

    # Fallback: If board had too few detections, re-run with lower floor
    if len(filtered_heads) < min_heads_expected and best_score >= 0.60:
        filtered_heads = []
        fallback_cutoff = 0.58
        for cand in raw_candidates:
            cx, cy, direction, name, score, sw, sh = cand
            if score < fallback_cutoff:
                continue
            min_dist = max(sw, sh)
            if all((cx - fx) ** 2 + (cy - fy) ** 2 > (min_dist * 0.8) ** 2 for fx, fy, _, _, _ in filtered_heads):
                filtered_heads.append((cx, cy, direction, name, (sw, sh)))
                effective_dims = (sw, sh)

    clean_heads = [(h[0], h[1], h[2], h[3]) for h in filtered_heads]
    return clean_heads, effective_dims


def get_next_move(frame, templates, arrow_dims):
    """
    Parses current frame using multi-scale detection, finds all arrowheads,
    and returns the first free playable move.
    """
    h, w, _ = frame.shape
    y1, y2 = int(h * 0.20), int(h * 0.95)
    x1, x2 = int(w * 0.01), int(w * 0.99)
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 1. Multi-scale arrowhead detection
    filtered_heads, active_dims = detect_arrowheads_multiscale(
        binary, templates, arrow_dims
    )

    if not filtered_heads:
        return None, 0

    # 2. Check clear exit corridors
    tw, th = active_dims
    clearance_radius = max(tw, th) // 2 + 4
    start_dist = clearance_radius + 2
    rh, rw = binary.shape

    for cx, cy, (dx, dy), name in filtered_heads:
        clearance_mask = binary.copy()
        cv2.circle(clearance_mask, (cx, cy), clearance_radius, 0, -1)

        ray_x = cx + dx * start_dist
        ray_y = cy + dy * start_dist
        blocked = False
        steps = 0

        while 0 <= ray_x < rw and 0 <= ray_y < rh:
            px, py = int(ray_x), int(ray_y)
            sample = clearance_mask[max(0, py - 1):min(rh, py + 2), max(0, px - 1):min(rw, px + 2)]
            if np.any(sample > 0):
                blocked = True
                break
            ray_x += dx * 2
            ray_y += dy * 2
            steps += 1

        if not blocked and steps > 0:
            # Return first free move mapped back to global screen coordinates
            return (cx + x1, cy + y1, name), len(filtered_heads)

    return None, len(filtered_heads)


def play_game():
    templates, arrow_dims = load_templates("arrow_up_template.png")
    print("[*] Starting auto-player. Ensure puzzle board is visible...")

    move_count = 0
    stuck_counter = 0

    while True:
        frame = capture_screen()
        move, remaining_arrows = get_next_move(frame, templates, arrow_dims)

        if remaining_arrows == 0:
            if move_count > 0:
                print(f"[✓] Puzzle clear! Completed {move_count} moves.")
                break
            else:
                stuck_counter += 1
                print(f"[!] 0 arrowheads detected on board. Retrying ({stuck_counter}/3)...")
                time.sleep(0.8)
                if stuck_counter >= 3:
                    print("[X] Could not detect arrowheads on screen. Check 'debug_live_binary.png'.")
                    break
                continue

        if move is None:
            stuck_counter += 1
            print(f"[!] No clear move detected. Retrying ({stuck_counter}/3)...")
            time.sleep(0.6)
            if stuck_counter >= 3:
                print("[X] Board stuck or cyclic dependency reached. Halting.")
                break
            continue

        stuck_counter = 0
        tap_x, tap_y, direction = move
        move_count += 1
        print(f"[{move_count}] Clearing {direction} arrow at ({tap_x}, {tap_y}) | Remaining: {remaining_arrows}")

        send_tap(tap_x, tap_y)

        # Wait for departure animation to settle
        # time.sleep(0.35)


if __name__ == "__main__":
    play_game()
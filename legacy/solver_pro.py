import cv2
import time
import subprocess
import numpy as np
import time

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


def get_all_playable_moves(frame, templates, arrow_dims):
    h, w, _ = frame.shape
    y1, y2 = int(h * 0.20), int(h * 0.95)
    x1, x2 = int(w * 0.01), int(w * 0.99)
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    filtered_heads, active_dims = detect_arrowheads_multiscale(
        binary, templates, arrow_dims, min_heads_expected=4
    )

    if not filtered_heads:
        return [], 0

    tw, th = active_dims
    clearance_radius = max(tw, th) // 2 + 4
    start_dist = clearance_radius + 2
    rh, rw = binary.shape

    playable_batch = []
    occupied_corridors = np.zeros((rh, rw), dtype=np.uint8)

    for cx, cy, (dx, dy), name in filtered_heads:
        clearance_mask = binary.copy()
        cv2.circle(clearance_mask, (cx, cy), clearance_radius, 0, -1)

        ray_x = cx + dx * start_dist
        ray_y = cy + dy * start_dist
        blocked = False
        corridor_path = []

        while 0 <= ray_x < rw and 0 <= ray_y < rh:
            px, py = int(ray_x), int(ray_y)
            sample = clearance_mask[max(0, py - 1):min(rh, py + 2), max(0, px - 1):min(rw, px + 2)]
            
            # Blocked by another arrow body OR crosses a corridor already claimed in this batch
            if np.any(sample > 0) or occupied_corridors[py, px] > 0:
                blocked = True
                break

            corridor_path.append((px, py))
            ray_x += dx * 2
            ray_y += dy * 2

        if not blocked and len(corridor_path) > 0:
            playable_batch.append((cx + x1, cy + y1, name))
            # Reserve this corridor so later moves in this frame don't collide with it
            for px, py in corridor_path:
                occupied_corridors[max(0, py-2):min(rh, py+3), max(0, px-2):min(rw, px+3)] = 255

    return playable_batch, len(filtered_heads)

class FastAdbController:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["adb", "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

    def tap(self, x, y):
        self.proc.stdin.write(f"input tap {int(x)} {int(y)}\n")
        self.proc.stdin.flush()

    def tap_batch(self, points, interval=0.08):
        """Sends multiple taps with minimal delay."""
        for x, y in points:
            self.tap(x, y)
            time.sleep(interval)

    def close(self):
        try:
            self.proc.stdin.write("exit\n")
            self.proc.stdin.flush()
            self.proc.terminate()
        except Exception:
            pass

def capture_screen_fast():
    """Captures uncompressed raw RGBA stream directly from Android framebuffer."""
    cmd = ["adb", "exec-out", "screencap"]
    raw = subprocess.check_output(cmd)
    
    # Android screencap header is 12-16 bytes (width, height, format)
    # Stride/dimensions are stored in the first 3 32-bit ints
    header = np.frombuffer(raw[:12], dtype=np.uint32)
    w, h = int(header[0]), int(header[1])

    # Convert remaining buffer into image
    img = np.frombuffer(raw[16:], dtype=np.uint8)
    img = img.reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

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

def capture_screen():
    """Captures screenshot directly from ADB buffer into OpenCV BGR."""
    cmd = ["adb", "exec-out", "screencap", "-p"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ADB Error: {err.decode('utf-8')}")
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
import time

def play_game_fast():
    templates, arrow_dims = load_templates("arrow_up_template.png")
    adb = FastAdbController()
    print("[*] Starting fast auto-player...")

    move_count = 0
    stuck_counter = 0

    # 1. Start the timer
    start_time = time.perf_counter()

    try:
        while True:
            frame = capture_screen()  # or capture_screen_fast()
            batch, remaining_arrows = get_all_playable_moves(frame, templates, arrow_dims)

            # Puzzle complete condition
            if remaining_arrows == 0:
                if move_count > 0:
                    # 2. Stop the timer on win
                    elapsed = time.perf_counter() - start_time
                    moves_per_sec = move_count / elapsed if elapsed > 0 else 0
                    
                    print("\n" + "=" * 45)
                    print(f"[✓] Puzzle Cleared!")
                    print(f"[-] Total Moves:       {move_count}")
                    print(f"[-] Total Time:        {elapsed:.2f} seconds")
                    print(f"[-] Solving Speed:     {moves_per_sec:.2f} moves/sec")
                    print("=" * 45 + "\n")
                break

            if not batch:
                stuck_counter += 1
                time.sleep(0.3)
                if stuck_counter >= 3:
                    elapsed = time.perf_counter() - start_time
                    print(f"\n[X] Solver halted after {elapsed:.2f}s ({move_count} moves made).")
                    break
                continue

            stuck_counter = 0
            print(f"[*] Executing batch of {len(batch)} parallel moves...")

            for x, y, direction in batch:
                move_count += 1
                print(f"  -> Tap #{move_count}: {direction} at ({x}, {y})")
                adb.tap(x, y)
                time.sleep(0.08)

            time.sleep(0.35)

    finally:
        adb.close()

if __name__ == "__main__":
    play_game_fast()
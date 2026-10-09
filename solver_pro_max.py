import time, cv2, numpy as np, subprocess

def extract_cascading_wave(binary, filtered_heads, active_dims, max_waves=4):
    """
    Simulates consecutive clears within a single frame:
    Wave 1: Arrows unblocked right now.
    Wave 2: Arrows unblocked after Wave 1 disappears.
    Wave 3: Arrows unblocked after Wave 2 disappears...
    """
    tw, th = active_dims
    clearance_radius = max(tw, th) // 2 + 4
    start_dist = clearance_radius + 2
    rh, rw = binary.shape

    sim_binary = binary.copy()
    remaining = list(filtered_heads)
    execution_waves = []

    for wave_idx in range(max_waves):
        current_wave = []
        occupied_corridors = np.zeros((rh, rw), dtype=np.uint8)
        unblocked_indices = []

        for idx, (cx, cy, (dx, dy), name) in enumerate(remaining):
            # Check clearance on simulated board state
            mask = sim_binary.copy()
            cv2.circle(mask, (cx, cy), clearance_radius, 0, -1)

            ray_x = cx + dx * start_dist
            ray_y = cy + dy * start_dist
            blocked = False
            corridor_path = []

            while 0 <= ray_x < rw and 0 <= ray_y < rh:
                px, py = int(ray_x), int(ray_y)
                sample = mask[max(0, py - 1):min(rh, py + 2), max(0, px - 1):min(rw, px + 2)]
                
                if np.any(sample > 0) or occupied_corridors[py, px] > 0:
                    blocked = True
                    break

                corridor_path.append((px, py))
                ray_x += dx * 2
                ray_y += dy * 2

            if not blocked and len(corridor_path) > 0:
                current_wave.append((cx, cy, (dx, dy), name))
                unblocked_indices.append(idx)
                for px, py in corridor_path:
                    occupied_corridors[max(0, py-2):min(rh, py+3), max(0, px-2):min(rw, px+3)] = 255

        if not current_wave:
            break

        execution_waves.append(current_wave)

        # "Erase" cleared arrows from the simulation to unlock dependent arrows
        for cx, cy, (dx, dy), _ in current_wave:
            # Erase head
            cv2.circle(sim_binary, (cx, cy), clearance_radius + 6, 0, -1)
            # Erase 30px backward along the tail body
            bx, by = cx - dx * 10, cy - dy * 10
            for _ in range(4):
                if 0 <= bx < rw and 0 <= by < rh:
                    cv2.circle(sim_binary, (int(bx), int(by)), 6, 0, -1)
                    bx -= dx * 8
                    by -= dy * 8

        # Remove cleared arrows from remaining pool
        remaining = [item for i, item in enumerate(remaining) if i not in unblocked_indices]

    return execution_waves, len(filtered_heads)

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

    def close(self):
        try:
            self.proc.stdin.write("exit\n")
            self.proc.stdin.flush()
            self.proc.terminate()
        except Exception:
            pass

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

def find_touchscreen_device():
    """
    Finds the event path corresponding to the touchscreen supporting multi-touch.
    Returns something like '/dev/input/event2' or '/dev/input/event4'.
    """
    try:
        output = subprocess.check_output(["adb", "shell", "getevent", "-p"]).decode("utf-8")
        current_dev = None
        for line in output.splitlines():
            if "/dev/input/event" in line:
                current_dev = line.strip().split()[-1]
            if "ABS_MT_TRACKING_ID" in line and current_dev:
                return current_dev
    except Exception as e:
        print(f"[!] Warning detecting touch device: {e}")
    
    # Common default fallback on most Android devices
    return "/dev/input/event2"
    
def play_game_blazing_fast(tap_delay=0.07, wave_delay=0.18, frame_delay=0.25):
    """
    Plays the game by predicting cascading waves ahead from a single frame,
    adding a safe short delay between each tap.
    
    :param tap_delay: Delay (seconds) between individual taps in the same wave.
    :param wave_delay: Delay (seconds) between consecutive predicted waves.
    :param frame_delay: Delay (seconds) before taking the next screenshot.
    """
    templates, arrow_dims = load_templates("arrow_up_template.png")
    adb = FastAdbController()
    print("[*] Predictive auto-player running with safe delays...")

    move_count = 0
    stuck_counter = 0
    start_time = time.perf_counter()

    try:
        while True:
            frame = capture_screen_fast()  # capture_screen_fast()/capture_screenshot()
            h, w, _ = frame.shape
            y1, y2 = int(h * 0.20), int(h * 0.95)
            x1, x2 = int(w * 0.01), int(w * 0.99)
            roi = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

            # 1. Multi-scale detection
            filtered_heads, active_dims = detect_arrowheads_multiscale(
                binary, templates, arrow_dims, min_heads_expected=4
            )

            # Puzzle complete condition
            if not filtered_heads:
                if move_count > 0:
                    elapsed = time.perf_counter() - start_time
                    print("\n" + "=" * 45)
                    print(f"[✓] Level Cleared!")
                    print(f"[-] Total Moves:   {move_count}")
                    print(f"[-] Total Time:    {elapsed:.2f} seconds")
                    print(f"[-] Speed:         {move_count / elapsed:.2f} moves/s")
                    print("=" * 45 + "\n")
                break

            # 2. Predict up to 3 cascading waves ahead from this single frame
            waves, total_heads = extract_cascading_wave(
                binary, filtered_heads, active_dims, max_waves=3
            )

            if not waves:
                stuck_counter += 1
                time.sleep(0.3)
                if stuck_counter >= 3:
                    print("[X] Stalled. No unblocked moves found. Halting.")
                    break
                continue

            stuck_counter = 0

            # 3. Execute wave by wave with safe tap delays
            for wave_idx, wave in enumerate(waves):
                print(f"  [Wave {wave_idx + 1}] Tapping {len(wave)} moves...")

                for cx, cy, (dx, dy), direction in wave:
                    tx = cx + x1
                    ty = cy + y1
                    move_count += 1
                    print(f"    -> Tap #{move_count}: {direction} at ({tx}, {ty})")
                    
                    adb.tap(tx, ty)
                    # Safe short sleep between each individual tap
                    time.sleep(tap_delay)

                # Wait for arrows in this wave to slide clear before next wave triggers
                if wave_idx < len(waves) - 1:
                    time.sleep(wave_delay)

            # Settle time before next screenshot capture
            time.sleep(frame_delay)

    finally:
        adb.close()

if __name__ == "__main__":
    play_game_blazing_fast()
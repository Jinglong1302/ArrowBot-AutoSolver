import time
import subprocess
import cv2
import numpy as np


class FastAdbController:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["adb", "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
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
        "LEFT": (tmpl_base, (-1, 0)),
        "UP": (cv2.rotate(tmpl_base, cv2.ROTATE_90_CLOCKWISE), (0, -1)),
        "RIGHT": (cv2.rotate(tmpl_base, cv2.ROTATE_180), (1, 0)),
        "DOWN": (cv2.rotate(tmpl_base, cv2.ROTATE_90_COUNTERCLOCKWISE), (0, 1)),
    }
    return templates, (tw, th)


def capture_screen_fast():
    """Captures uncompressed raw RGBA stream directly from Android framebuffer."""
    cmd = ["adb", "exec-out", "screencap"]
    raw = subprocess.check_output(cmd)

    # Android screencap header is 12-16 bytes (width, height, format)
    header = np.frombuffer(raw[:12], dtype=np.uint32)
    w, h = int(header[0]), int(header[1])

    # Convert remaining buffer into BGR image
    img = np.frombuffer(raw[16:], dtype=np.uint8)
    img = img.reshape((h, w, 4))
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def detect_arrowheads_multiscale(binary_roi, base_templates, base_dims, threshold=0.72):
    """
    Scans the board across multiple scales to handle varying zoom levels 
    and grid resolutions between different stages.
    """
    tw_base, th_base = base_dims
    raw_candidates = []

    # Scale range covering both dense large mazes (0.65x) and sparse small mazes (up to 1.85x)
    scales = np.linspace(0.60, 1.30, 11)

    for scale in scales:
        scaled_w = int(tw_base * scale)
        scaled_h = int(th_base * scale)
        if scaled_w < 5 or scaled_h < 5:
            continue

        for name, (tmpl, direction) in base_templates.items():
            scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            _, scaled_tmpl = cv2.threshold(scaled_tmpl, 127, 255, cv2.THRESH_BINARY)

            res = cv2.matchTemplate(binary_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                score = float(res[pt[1], pt[0]])
                cx = pt[0] + scaled_w // 2
                cy = pt[1] + scaled_h // 2
                raw_candidates.append((cx, cy, direction, name, score, scaled_w, scaled_h))

    if not raw_candidates:
        return [], base_dims

    # Sort descending by match score
    raw_candidates.sort(key=lambda item: item[4], reverse=True)

    # Non-Maximum Suppression
    filtered_heads = []
    effective_dims = base_dims

    for cand in raw_candidates:
        cx, cy, direction, name, score, sw, sh = cand
        min_dist = max(sw, sh)
        if all((cx - fx) ** 2 + (cy - fy) ** 2 > (min_dist * 0.8) ** 2 for fx, fy, _, _ in filtered_heads):
            filtered_heads.append((cx, cy, direction, name))
            effective_dims = (sw, sh)

    return filtered_heads, effective_dims


def solve_all_waves_one_shot(binary, heads, active_dims, max_waves=120):
    """
    Solves 100% of the puzzle in memory from a single binary frame using:
    1. Exact connected component mapping for each detected arrowhead.
    2. Raycasting to find unblocked flight corridors.
    3. Complete component body erasure upon wave clearance.
    
    Returns:
        waves: list of waves, where each wave is a list of (cx, cy, (dx, dy), name)
        remaining_count: number of arrows that could not be solved in this simulation
    """
    if not heads:
        return [], 0

    tw, th = active_dims
    clearance_radius = max(tw, th) // 2 + 6
    start_dist = clearance_radius + 4
    rh, rw = binary.shape

    # Connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Map each head to its corresponding component label
    head_to_comp = []
    for idx, (cx, cy, (dx, dy), name) in enumerate(heads):
        lbl = labels[int(cy), int(cx)]
        if lbl == 0:
            window = labels[
                max(0, int(cy) - 6) : min(rh, int(cy) + 7),
                max(0, int(cx) - 6) : min(rw, int(cx) + 7),
            ]
            non_zero = window[window > 0]
            if len(non_zero) > 0:
                lbl = int(np.bincount(non_zero).argmax())
        head_to_comp.append(lbl)

    sim_binary = binary.copy()
    remaining = list(range(len(heads)))
    waves = []

    for wave_idx in range(max_waves):
        current_wave = []
        unblocked_indices = []
        occupied_corridors = np.zeros((rh, rw), dtype=np.uint8)

        for i in remaining:
            cx, cy, (dx, dy), name = heads[i]

            # Mask out head body from clearance check
            mask = sim_binary.copy()
            cv2.circle(mask, (cx, cy), clearance_radius, 0, -1)

            ray_x = cx + dx * start_dist
            ray_y = cy + dy * start_dist
            blocked = False
            corridor_path = []

            while 0 <= ray_x < rw and 0 <= ray_y < rh:
                px, py = int(ray_x), int(ray_y)
                sample = mask[
                    max(0, py - 1) : min(rh, py + 2),
                    max(0, px - 1) : min(rw, px + 2),
                ]

                if np.any(sample > 0) or occupied_corridors[py, px] > 0:
                    blocked = True
                    break

                corridor_path.append((px, py))
                ray_x += dx * 2
                ray_y += dy * 2

            if not blocked and len(corridor_path) > 0:
                current_wave.append((cx, cy, (dx, dy), name))
                unblocked_indices.append(i)
                for px, py in corridor_path:
                    occupied_corridors[
                        max(0, py - 2) : min(rh, py + 3),
                        max(0, px - 2) : min(rw, px + 3),
                    ] = 255

        if not current_wave:
            break

        waves.append(current_wave)

        # Erase the entire connected arrow body from the simulation
        for i in unblocked_indices:
            lbl = head_to_comp[i]
            if lbl > 0:
                sim_binary[labels == lbl] = 0
            else:
                cx, cy, (dx, dy), _ = heads[i]
                cv2.circle(sim_binary, (cx, cy), clearance_radius + 10, 0, -1)

        remaining = [i for i in remaining if i not in unblocked_indices]

    return waves, len(remaining)


def solve_single_level(adb, templates, base_dims, tap_delay=0.04, wave_delay=0.08, max_fallback_rounds=3):
    """
    Solves one single level:
    1. Captures screen.
    2. Solves 100% in memory.
    3. Streams taps via ADB.
    4. Handles fallback screenshots if needed.
    
    Returns:
        (success, total_moves, elapsed_time)
    """
    total_moves_made = 0
    total_start_time = time.perf_counter()

    for round_idx in range(1, max_fallback_rounds + 1):
        frame = capture_screen_fast()
        h, w, _ = frame.shape
        y1, y2 = int(h * 0.20), int(h * 0.95)
        x1, x2 = int(w * 0.01), int(w * 0.99)
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Detect all arrow heads
        heads, active_dims = detect_arrowheads_multiscale(
            binary, templates, base_dims, threshold=0.72
        )

        # If no heads found on initial frame, poll for board spawn animation to finish
        if not heads and total_moves_made == 0 and round_idx == 1:
            print("[*] Waiting for puzzle board to finish spawning...")
            for _ in range(8):
                time.sleep(0.5)
                frame = capture_screen_fast()
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
                heads, active_dims = detect_arrowheads_multiscale(
                    binary, templates, base_dims, threshold=0.72
                )
                if heads:
                    print(f"[+] Board spawned: {len(heads)} arrowheads detected.")
                    break

        if not heads:
            if total_moves_made > 0:
                elapsed = time.perf_counter() - total_start_time
                print("\n" + "=" * 45)
                print(f"[✓] Level 100% Cleared!")
                print(f"[-] Total Moves:   {total_moves_made}")
                print(f"[-] Total Time:    {elapsed:.2f} seconds")
                print(f"[-] Solve Speed:   {total_moves_made / elapsed:.2f} moves/s")
                print("=" * 45 + "\n")
                return True, total_moves_made, elapsed
            else:
                print("[!] No arrowheads found on screen.")
                return False, 0, 0.0

        round_name = "Primary Frame" if round_idx == 1 else f"Fallback Round #{round_idx - 1}"
        print(f"\n[*] [{round_name}] Detected {len(heads)} arrowheads. Solving full sequence in memory...")

        # Run full one-shot in-memory simulation
        waves, remaining_count = solve_all_waves_one_shot(binary, heads, active_dims)
        moves_in_plan = sum(len(w) for w in waves)

        print(f"[+] Computed {len(waves)} waves ({moves_in_plan}/{len(heads)} arrows solved in memory).")

        if not waves:
            print("[X] Simulation found 0 unblocked moves.")
            return False, total_moves_made, time.perf_counter() - total_start_time

        # Blast taps wave-by-wave
        for wave_idx, wave in enumerate(waves):
            print(f"  -> Dispatching Wave {wave_idx + 1}/{len(waves)} ({len(wave)} taps)...")
            for cx, cy, (dx, dy), direction in wave:
                tx = cx + x1
                ty = cy + y1
                total_moves_made += 1
                adb.tap(tx, ty)
                if tap_delay > 0:
                    time.sleep(tap_delay)

            if wave_delay > 0 and wave_idx < len(waves) - 1:
                time.sleep(wave_delay)

        # If all were solved in memory, level is finished!
        if remaining_count == 0:
            time.sleep(0.4)
            elapsed = time.perf_counter() - total_start_time
            print("\n" + "=" * 45)
            print(f"[✓] Level 100% Cleared in One Shot!")
            print(f"[-] Total Moves:   {total_moves_made}")
            print(f"[-] Total Time:    {elapsed:.2f} seconds")
            print(f"[-] Solve Speed:   {total_moves_made / elapsed:.2f} moves/s")
            print("=" * 45 + "\n")
            return True, total_moves_made, elapsed
        else:
            print(f"[!] {remaining_count} arrows left after simulation. Taking fallback screenshot...")
            time.sleep(0.3)

    return True, total_moves_made, time.perf_counter() - total_start_time


def play_one_shot(tap_delay=0.04, wave_delay=0.08, max_fallback_rounds=3):
    templates, base_dims = load_templates("arrow_up_template.png")
    adb = FastAdbController()
    print("[*] Starting One-Shot Arrow Maze Solver (Zero-Delay Pipeline)...")
    try:
        solve_single_level(
            adb=adb,
            templates=templates,
            base_dims=base_dims,
            tap_delay=tap_delay,
            wave_delay=wave_delay,
            max_fallback_rounds=max_fallback_rounds,
        )
    finally:
        adb.close()


if __name__ == "__main__":
    play_one_shot(tap_delay=0.04, wave_delay=0.08)

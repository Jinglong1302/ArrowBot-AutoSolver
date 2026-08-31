import time
import os
import subprocess
import cv2
import numpy as np

from solver_one_shot import (
    FastAdbController,
    load_templates,
    capture_screen_fast,
    solve_single_level,
    detect_arrowheads_multiscale,
)


def ensure_pinch_injector():
    """Ensures the native multi-touch DEX injector is uploaded to the device."""
    if os.path.exists("pinch.jar"):
        try:
            subprocess.run(["adb", "push", "pinch.jar", "/data/local/tmp/pinch.jar"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def pinch_to_zoom_out(screen_w, screen_h, duration_ms=350, repeats=2):
    """
    Executes a true synchronized two-finger multi-touch pinch zoom-out
    via native Android InputManager DEX injection (/data/local/tmp/pinch.jar).
    Guarantees exact 0ms simultaneous touchdown and smooth coordinated inward movement.
    """
    cmd = [
        "adb",
        "shell",
        f"CLASSPATH=/data/local/tmp/pinch.jar app_process / com.touch.PinchZoom {screen_w} {screen_h} {duration_ms}",
    ]

    for r in range(repeats):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[!] Native pinch error: {e}")

        if r < repeats - 1:
            time.sleep(0.25)


def find_button(frame, btn_template_path="btn_next.png", threshold=0.70):
    """Matches the 'Continue/Next' victory button in the frame."""
    btn_tmpl = cv2.imread(btn_template_path)
    if btn_tmpl is None:
        return None

    res = cv2.matchTemplate(frame, btn_tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        th, tw, _ = btn_tmpl.shape
        bx = max_loc[0] + tw // 2
        by = max_loc[1] + th // 2
        return (bx, by)
    return None


def wait_and_click_next(adb, templates, base_dims, btn_template_path="btn_next.png", timeout=12.0):
    """
    Polls screen for the Continue/Next button after winning and clicks it.
    Also detects if the next level has already auto-spawned.
    """
    print("[*] Waiting for victory screen and 'Continue' button...")
    start_wait = time.perf_counter()

    # Initial delay for victory popup animation to enter
    time.sleep(1.2)

    while time.perf_counter() - start_wait < timeout:
        frame = capture_screen_fast()
        h, w, _ = frame.shape

        # 1. Check if Continue button is visible
        btn_coords = find_button(frame, btn_template_path)
        if btn_coords:
            bx, by = btn_coords
            print(f"[+] Found 'Continue' button at ({bx}, {by}). Clicking...")
            adb.tap(bx, by)
            # Sleep for button click transition & new level fade-in
            print("[*] Clicked continue. Waiting 2.0s for next level to load...")
            time.sleep(2.0)
            return True

        # 2. Check if next level is already active (arrows visible)
        roi = frame[int(h * 0.20):int(h * 0.95), int(w * 0.01):int(w * 0.99)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        heads, _ = detect_arrowheads_multiscale(binary, templates, base_dims, threshold=0.72)
        if len(heads) >= 4:
            print(f"[+] Next level already active ({len(heads)} arrows on screen). Continuing...")
            return True

        time.sleep(0.5)

    print("[?] Transition timeout reached. Checking board state...")
    return False


def play_continuous(max_levels=100, tap_delay=0.04, wave_delay=0.08):
    """
    Continuous Auto-Play Flow:
    1. Solves current level using solver_one_shot engine.
    2. Clicks 'Continue' button on victory screen with smooth animation sleep.
    3. Zooms out to the max via synchronized native multi-touch injection.
    4. Repeats for the next level.
    """
    ensure_pinch_injector()
    templates, base_dims = load_templates("arrow_up_template.png")
    adb = FastAdbController()

    init_frame = capture_screen_fast()
    screen_h, screen_w, _ = init_frame.shape
    print(f"[*] Detected Screen Resolution: {screen_w}x{screen_h}")

    current_level = 1
    total_session_moves = 0
    session_start_time = time.perf_counter()

    try:
        while current_level <= max_levels:
            print(f"\n{'=' * 20} LEVEL {current_level} START {'=' * 20}")

            # Step 1: Solve current level
            success, moves, elapsed = solve_single_level(
                adb=adb,
                templates=templates,
                base_dims=base_dims,
                tap_delay=tap_delay,
                wave_delay=wave_delay,
            )

            if success and moves > 0:
                total_session_moves += moves
                current_level += 1
            elif not success and moves == 0:
                print("[!] Board was empty or level could not be solved. Retrying check...")
                time.sleep(1.0)

            # Step 2: Handle victory screen and click Continue with animation gaps
            wait_and_click_next(
                adb=adb,
                templates=templates,
                base_dims=base_dims,
                btn_template_path="btn_next.png",
                timeout=10.0,
            )

            # Step 3: Native synchronized 2-finger pinch zoom out
            print("[*] Zooming out to max board view via native multi-touch...")
            pinch_to_zoom_out(screen_w, screen_h, duration_ms=350, repeats=2)
            time.sleep(1.0)

        total_elapsed = time.perf_counter() - session_start_time
        print("\n" + "=" * 50)
        print(f"[✓] Session Finished!")
        print(f"[-] Total Session Moves: {total_session_moves}")
        print(f"[-] Total Session Time:  {total_elapsed:.2f}s")
        print("=" * 50 + "\n")

    finally:
        adb.close()


if __name__ == "__main__":
    play_continuous(max_levels=100, tap_delay=0.04, wave_delay=0.08)
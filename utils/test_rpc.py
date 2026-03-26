"""
test_rpc_loop.py  —  End-to-end RPC Loop Validator
====================================================
Runs a series of tests against the mock (or real) MOTController server
to confirm every part of the pipeline works before you touch hardware.

Usage:
    # Terminal 1 — start the mock server:
    python mock_mot_controller.py

    # Terminal 2 — run this test:
    python test_rpc_loop.py

    # Optional: test against a remote machine:
    python test_rpc_loop.py --host 192.168.1.10 --port 3386

All tests print PASS / FAIL so you know exactly what's working.
"""

import argparse
import sys
import time
import numpy as np

# ── Colour helpers for terminal output ───────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")
def section(title): print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=3386)
    p.add_argument("--steps", type=int, default=10,
                   help="Number of env steps to run in the loop test")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Raw RPC connection
# ══════════════════════════════════════════════════════════════════════════
def test_raw_connection(host, port):
    section("TEST 1: Raw sipyco RPC connection")
    try:
        from sipyco.pc_rpc import Client as RPCClient
    except ImportError:
        fail("sipyco not installed. Run: pip install sipyco")

    try:
        client = RPCClient(host, port, target_name="MOTController")
    except Exception as e:
        fail(f"Could not connect to {host}:{port} — {e}\n"
             f"       Is mock_mot_controller.py running?")

    reply = client.ping()
    if reply != "pong":
        fail(f"ping() returned unexpected value: {reply!r}")
    ok(f"Connected to {host}:{port}, ping → '{reply}'")

    client.close_rpc()
    return True


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — ARTIQBridge (client wrapper)
# ══════════════════════════════════════════════════════════════════════════
def test_artiq_bridge(host, port):
    section("TEST 2: ARTIQBridge wrapper")

    # Import from local path (adjust if your project layout differs)
    try:
        sys.path.insert(0, ".")
        from artiq_bridge import ARTIQBridge
    except ImportError as e:
        fail(f"Could not import ARTIQBridge: {e}")

    bridge = ARTIQBridge(host=host, port=port)
    ok("ARTIQBridge connected")

    # run_experiment
    success = bridge.run_experiment(20.0)
    if not success:
        fail("run_experiment(20.0) returned False")
    ok("run_experiment(20.0) → True")

    # wait_for_new_images
    images = bridge.wait_for_new_images(count=4)
    if len(images) != 4:
        fail(f"Expected 4 images, got {len(images)}")
    for i, img in enumerate(images):
        if img.shape != (50, 50):
            fail(f"Image {i} has wrong shape: {img.shape}, expected (50, 50)")
        if img.dtype != np.float32:
            fail(f"Image {i} has wrong dtype: {img.dtype}, expected float32")
        if not (0.0 <= img.min() and img.max() <= 1.0):
            fail(f"Image {i} pixel values out of [0,1]: min={img.min()}, max={img.max()}")
    ok(f"wait_for_new_images → {len(images)} images, shape={images[0].shape}, "
       f"dtype={images[0].dtype}, range=[{images[0].min():.3f}, {images[0].max():.3f}]")

    # get_initial_images
    init_imgs = bridge.get_initial_images()
    if len(init_imgs) != 4:
        fail(f"get_initial_images returned {len(init_imgs)} images, expected 4")
    ok(f"get_initial_images → {len(init_imgs)} images")

    bridge.close()
    return bridge  # return closed bridge (we'll create a fresh one in next test)


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — RealMOTEnvironment (full env interface)
# ══════════════════════════════════════════════════════════════════════════
def test_real_mot_env(host, port):
    section("TEST 3: RealMOTEnvironment reset() and step()")

    try:
        from Environments.RealMOTenv import RealMOTEnvironment
    except ImportError as e:
        fail(f"Could not import RealMOTEnvironment: {e}")

    env = RealMOTEnvironment(artiq_host=host, artiq_port=port,
                              detuning_range=(0.0, 50.0), episode_length=5)
    ok("RealMOTEnvironment created")

    # reset()
    obs = env.reset()
    _validate_observation(obs, label="reset()")
    ok(f"reset() → obs['images'].shape={obs['images'].shape}, "
       f"obs['additional']={obs['additional']}")

    # step() with a fixed action
    action = np.array([0.0])   # maps to detuning = 25.0 Γ  (midpoint)
    obs2, reward, done, info = env.step(action)
    _validate_observation(obs2, label="step()")

    if not isinstance(reward, (int, float)):
        fail(f"reward is not a number: {reward!r}")
    if not isinstance(done, bool):
        fail(f"done is not a bool: {done!r}")
    for key in ("atom_number", "temperature", "detuning", "step"):
        if key not in info:
            fail(f"info dict missing key '{key}'")

    ok(f"step([0.0]) → reward={reward:.4f}, done={done}, "
       f"atoms={info['atom_number']:.3e}, detuning={info['detuning']:.1f}")

    return env


# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — Full episode loop
# ══════════════════════════════════════════════════════════════════════════
def test_full_episode(host, port, num_steps):
    section(f"TEST 4: Full episode loop ({num_steps} steps)")

    from artiq_bridge import ARTIQBridge
    from Environments.RealMOTenv import RealMOTEnvironment

    env = RealMOTEnvironment(artiq_host=host, artiq_port=port,
                              detuning_range=(0.0, 50.0),
                              episode_length=num_steps)
    obs = env.reset()

    detunings = []
    atoms_per_step = []
    step_times = []

    print(f"\n  {'Step':>4}  {'Action':>8}  {'Detuning (Γ)':>14}  "
          f"{'Atoms':>12}  {'Done':>5}  {'Δt (ms)':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*14}  {'─'*12}  {'─'*5}  {'─'*8}")

    for step in range(num_steps):
        action = np.random.uniform(-1.0, 1.0, size=(1,))

        t0 = time.time()
        obs, reward, done, info = env.step(action)
        dt_ms = (time.time() - t0) * 1000

        detunings.append(info['detuning'])
        atoms_per_step.append(info['atom_number'])
        step_times.append(dt_ms)

        print(f"  {step+1:>4}  {action[0]:>+8.3f}  {info['detuning']:>14.2f}  "
              f"{info['atom_number']:>12.3e}  {str(done):>5}  {dt_ms:>8.1f}")

        _validate_observation(obs, label=f"step {step+1}")

        if done:
            break

    # Summary stats
    print()
    ok(f"Completed {len(detunings)} steps without crash")
    ok(f"Detuning range explored: [{min(detunings):.1f}, {max(detunings):.1f}] Γ")
    ok(f"Atom number range: [{min(atoms_per_step):.2e}, {max(atoms_per_step):.2e}]")
    ok(f"Step latency: avg={np.mean(step_times):.0f}ms, "
       f"max={max(step_times):.0f}ms, min={min(step_times):.0f}ms")

    # Sanity check: atom numbers should vary (the mock physics is working)
    if max(atoms_per_step) < 1e3:
        fail("All atom numbers suspiciously low — check mock physics")
    if np.std(atoms_per_step) < 1.0:
        fail("Atom numbers show no variation — mock or bridge may be broken")
    ok("Atom numbers show realistic variation ✓")


# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — Reconnection resilience
# ══════════════════════════════════════════════════════════════════════════
def test_reconnection(host, port):
    section("TEST 5: Graceful close and reconnect")

    from artiq_bridge import ARTIQBridge

    b1 = ARTIQBridge(host=host, port=port)
    b1.run_experiment(15.0)
    b1.close()
    ok("First connection closed cleanly")

    time.sleep(0.3)

    b2 = ARTIQBridge(host=host, port=port)
    ok("Reconnected successfully")
    b2.run_experiment(25.0)
    ok("run_experiment() works after reconnect")
    b2.close()


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _validate_observation(obs: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    if "images" not in obs:
        fail(f"{prefix}observation missing 'images' key")
    if "additional" not in obs:
        fail(f"{prefix}observation missing 'additional' key")
    if obs["images"].shape != (50, 50, 4):
        fail(f"{prefix}obs['images'] shape={obs['images'].shape}, expected (50,50,4)")
    if obs["additional"].shape != (2,):
        fail(f"{prefix}obs['additional'] shape={obs['additional'].shape}, expected (2,)")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    print(f"\n{'═'*60}")
    print(f"  MOT RPC Loop Test Suite")
    print(f"  Target: {args.host}:{args.port}")
    print(f"{'═'*60}")

    test_raw_connection(args.host, args.port)
    test_artiq_bridge(args.host, args.port)
    test_real_mot_env(args.host, args.port)
    test_full_episode(args.host, args.port, args.steps)
    test_reconnection(args.host, args.port)

    print(f"\n{'═'*60}")
    print(f"  {GREEN}ALL TESTS PASSED{RESET} — RPC loop is working correctly.")
    print(f"  You can now switch to the real mot_controller.py on hardware.")
    print(f"{'═'*60}\n")
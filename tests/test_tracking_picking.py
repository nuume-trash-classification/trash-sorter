import numpy as np

from trash_sorter import config
from trash_sorter.perception import Detection
from trash_sorter.picking import GripperCommand, Phase, PickController
from trash_sorter.tracking import BeltTracker

V = config.BELT_SPEED
TOP = config.BELT_TOP_Z + 0.07


def det(x, y, t, cls="pet_bottle", conf=0.9):
    return Detection(cls, conf, np.array([x, y, TOP]), t)


def test_tracker_merges_frames_of_one_object():
    tr = BeltTracker()
    for k in range(5):
        t = 0.1 * k
        tr.update([det(-0.9 + V * t + 0.002, 0.05, t)])
    assert len(tr.tracks) == 1
    assert tr.tracks[0].hits == 5
    np.testing.assert_allclose(tr.tracks[0].position_at(1.0)[0], -0.9 + V * 1.0, atol=0.005)


def test_tracker_separates_two_objects_and_votes_class():
    tr = BeltTracker()
    for k, cls in enumerate(["glass_bottle", "pet_bottle", "glass_bottle"]):
        t = 0.1 * k
        tr.update([det(-0.9 + V * t, 0.0, t, cls), det(-0.6 + V * t, 0.1, t, "aluminum_can")])
    assert len(tr.tracks) == 2
    assert {t.class_name for t in tr.tracks} == {"glass_bottle", "aluminum_can"}


def test_tracker_prunes_objects_off_the_belt():
    tr = BeltTracker()
    tr.update([det(1.5, 0.0, 0.0)])
    tr.prune(2.0)
    assert tr.active() == []


def _run(objects, duration=25.0, dt=1 / 60, grip_ok=lambda obj: True, lag=0.0):
    """Simulated arm: the real tip follows the command with a first-order lag of ``lag`` s (0 = perfect).

    Suction holds if the real tip is close; ``delivered`` records where the real tip was when it opened.
    """
    tracker = BeltTracker()
    ctrl = PickController()
    held = None
    delivered = []
    tip = np.array(ctrl.params.idle_position, dtype=float)
    t = 0.0
    while t < duration:
        visible = [o for o in objects if not o.get("gone") and -1.15 < o["x0"] + V * t < -0.35]
        if int(t / config.CAMERA_PERIOD) != int((t - dt) / config.CAMERA_PERIOD):
            tracker.update([det(o["x0"] + V * t, o["y"], t, o["cls"]) for o in visible])
        tracker.prune(t)
        cmd = ctrl.step(t, dt, tracker, held is not None, tip)
        prev = tip
        tip = cmd.tip_target.copy() if lag == 0 else tip + (cmd.tip_target - tip) * min(dt / lag, 1.0)
        tip_speed = float(np.linalg.norm(tip - prev)) / dt
        if cmd.gripper == GripperCommand.CLOSE and held is None:
            for o in objects:
                p = np.array([o["x0"] + V * t, o["y"], TOP])
                if not o.get("gone") and np.linalg.norm(tip - p) < 0.02 and grip_ok(o):
                    held, o["gone"] = o, True
        if cmd.gripper == GripperCommand.OPEN and held is not None:
            delivered.append((held["cls"], ctrl.bin_name, tip.copy(), tip_speed))
            held = None
        t += dt
    return ctrl, delivered


def test_controller_picks_every_object_with_perfect_arm():
    objects = [{"x0": -1.4 - 0.6 * i, "y": [-0.1, 0.0, 0.1][i % 3], "cls": config.CLASS_NAMES[i % 4]} for i in range(4)]
    ctrl, delivered = _run(objects, duration=40)
    assert len(delivered) == 4
    for cls, bin_name, tip, _ in delivered:
        assert cls == bin_name
        bx, by = config.BIN_CENTERS[bin_name]
        np.testing.assert_allclose(tip[:2], [bx, by], atol=0.011)
    assert ctrl.phase == Phase.IDLE


def test_controller_waits_for_the_real_arm_before_releasing():
    # A lagging real arm passes the drop point at speed; released then, the object keeps the arm's
    # velocity and is thrown past the bin. The suction must open only once the real tip is there AND still.
    objects = [{"x0": -1.4 - 0.6 * i, "y": 0.0, "cls": config.CLASS_NAMES[i % 4]} for i in range(4)]
    ctrl, delivered = _run(objects, duration=40, lag=0.12)
    assert len(delivered) == 4
    for _, bin_name, tip, speed in delivered:
        bx, by = config.BIN_CENTERS[bin_name]
        assert np.linalg.norm(tip[:2] - [bx, by]) < ctrl.params.release_tolerance + 1e-6
        assert speed < ctrl.params.release_max_tip_speed + 1e-6


def test_controller_reports_failed_grip_and_moves_on():
    objects = [{"x0": -1.4, "y": 0.0, "cls": "aluminum_can"}, {"x0": -2.2, "y": 0.05, "cls": "paper_carton"}]
    ctrl, delivered = _run(objects, duration=30, grip_ok=lambda o: o["cls"] != "aluminum_can")
    kinds = [e.kind for e in ctrl.events]
    assert "grasp_failed" in kinds
    assert [d[0] for d in delivered] == ["paper_carton"]


def test_low_confidence_objects_are_left_for_the_end_bin():
    tracker = BeltTracker()
    ctrl = PickController()
    tracker.update([det(0.0, 0.0, 0.0, conf=0.2)])
    tracker.update([det(0.015, 0.0, 0.1, conf=0.2)])
    tracker.update([det(0.030, 0.0, 0.2, conf=0.2)])
    ctrl.step(0.1, 1 / 60, tracker, False)
    assert ctrl.phase == Phase.IDLE


def test_tracker_learns_per_object_velocity():
    tr = BeltTracker()
    vx, vy = 0.135, 0.01  # slower than the belt and drifting sideways (rolling bottle)
    for k in range(30):
        t = 0.1 * k
        tr.update([det(-0.9 + vx * t, 0.0 + vy * t, t)])
    [track] = tr.tracks
    np.testing.assert_allclose(track.velocity, [vx, vy], atol=1e-6)
    np.testing.assert_allclose(track.position_at(8.0)[:2], [-0.9 + vx * 8, vy * 8], atol=1e-6)

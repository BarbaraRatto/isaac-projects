"""Configure the live Isaac Sim Spot articulation for pure effort control.

Run this file from Isaac Sim's Script Editor while the stage is stopped.  The
external PD loop in ``translator_node.py`` already adds the position/velocity
feedback, so leaving the USD drives enabled adds a second, conflicting PD loop.
"""

import omni.usd


ROBOT_PRIM_PATH = "/World/spot"


def main():
    stage = omni.usd.get_context().get_stage()
    robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot.IsValid():
        raise RuntimeError(f"Robot prim not found: {ROBOT_PRIM_PATH}")

    changed = []
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(robot.GetPath()):
            continue

        stiffness = prim.GetAttribute("drive:angular:physics:stiffness")
        damping = prim.GetAttribute("drive:angular:physics:damping")
        if not stiffness.IsValid() and not damping.IsValid():
            continue

        if stiffness.IsValid():
            stiffness.Set(0.0)
        if damping.IsValid():
            damping.Set(0.0)
        changed.append(str(prim.GetPath()))

    if len(changed) != 12:
        raise RuntimeError(
            f"Expected 12 driven joints below {ROBOT_PRIM_PATH}, changed {len(changed)}: {changed}"
        )

    print("Pure effort mode enabled for:")
    print("\n".join(changed))
    print("Connect only effortCommand on the Isaac Articulation Controller node.")


if __name__ == "__main__":
    main()

"""Create the GUI camera before Isaac Lab attaches the new scene to Kit."""

from pxr import UsdGeom


def ensure_viewer_camera(stage):
    # Isaac Sim's set_camera_view assumes this prim already exists. A new
    # stage may be attached before Kit has created its default camera.
    path = "/OmniverseKit_Persp"
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        if not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Viewport path {path} exists but is not a Camera")
        return
    UsdGeom.Camera.Define(stage, path)

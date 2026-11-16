"""Constant-speed tracking of the fixed start–goal segment; no terrain input."""
import math


def commands(start,goal,position,yaw,speed,lookahead=1.0,heading_gain=1.5,max_yaw=1.5):
    values=(*start[:2],*goal[:2],*position[:2],yaw,speed,lookahead,heading_gain,max_yaw)
    if not all(math.isfinite(v) for v in values) or min(speed,lookahead,heading_gain,max_yaw)<=0:
        raise ValueError('Invalid line-controller input')
    dx,dy=goal[0]-start[0],goal[1]-start[1];length=math.hypot(dx,dy)
    if length<=0:
        raise ValueError('Start and goal must differ')
    ux,uy=dx/length,dy/length
    px,py=position[0]-start[0],position[1]-start[1]
    progress=px*ux+py*uy
    cross_track=-px*uy+py*ux
    target=min(length,max(0.,progress+lookahead))
    heading=math.atan2(start[1]+target*uy-position[1],start[0]+target*ux-position[0])
    error=math.atan2(math.sin(heading-yaw),math.cos(heading-yaw))
    yaw_rate=max(-max_yaw,min(max_yaw,heading_gain*error))
    return dict(forward_m_s=speed,yaw_rad_s=yaw_rate,cross_track_m=cross_track,
                line_progress_m=progress,heading_error_rad=error)

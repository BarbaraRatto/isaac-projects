"""Track the recorded route using its original forward-command time series."""
import bisect
import math


class RouteTracker:
    def __init__(self,trace,goal,lookahead=.35,max_yaw=1.5):
        if lookahead<=0:
            raise ValueError('Use a positive lookahead')
        self.trace=trace
        self.points=[(float(r['x_m']),float(r['y_m'])) for r in trace]
        self.goal=tuple(goal);self.lookahead=lookahead;self.max_yaw=max_yaw
        self.lengths=[math.dist(a,b) for a,b in zip(self.points,self.points[1:])]
        self.arc=[0.]
        for length in self.lengths:self.arc.append(self.arc[-1]+length)
        self.progress=0.

    def projection(self,position,local=False):
        candidates=[]
        for i,(a,b,length) in enumerate(zip(self.points,self.points[1:],self.lengths)):
            if length<1e-9:continue
            if local and (self.arc[i+1]<self.progress-.15 or self.arc[i]>self.progress+1.):continue
            dx,dy=b[0]-a[0],b[1]-a[1]
            t=max(0.,min(1.,((position[0]-a[0])*dx+(position[1]-a[1])*dy)/(length*length)))
            q=(a[0]+t*dx,a[1]+t*dy)
            candidates.append((math.dist(position[:2],q),self.arc[i]+t*length))
        if not candidates:raise ValueError('Reference route has no usable segment')
        return min(candidates)

    def point_at(self,distance):
        if distance>=self.arc[-1]:return self.points[-1]
        i=min(bisect.bisect_right(self.arc,distance)-1,len(self.lengths)-1)
        while self.lengths[i]<1e-9 and i<len(self.lengths)-1:i+=1
        t=(distance-self.arc[i])/max(self.lengths[i],1e-9)
        a,b=self.points[i:i+2]
        return (a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))

    def control(self,position,yaw,step):
        distance,progress=self.projection(position,local=True)
        self.progress=max(self.progress,progress)
        index=min(step+1,len(self.trace)-1)
        row=self.trace[index]
        speed=float(row['command_forward_m_s'])
        target=self.point_at(self.progress+self.lookahead)
        # Extend only beyond the recorded endpoint to meet the original goal criterion.
        endpoint_extension=self.progress>=self.arc[-1]-.05
        if endpoint_extension:target=self.goal
        dx,dy=target[0]-position[0],target[1]-position[1]
        error=math.atan2(math.sin(math.atan2(dy,dx)-yaw),math.cos(math.atan2(dy,dx)-yaw))
        target_distance=max(math.hypot(dx,dy),.1)
        yaw_rate=max(-self.max_yaw,min(self.max_yaw,2*speed*math.sin(error)/target_distance))
        return dict(forward_m_s=speed,yaw_rad_s=yaw_rate,reference_step=index,
            reference_time_s=float(row['time_s']),source_yaw_command_rad_s=float(row['command_yaw_rad_s']),
            tracking_error_m=distance,route_progress_m=self.progress,
            target_x_m=target[0],target_y_m=target[1],
            speed_profile_extended=int(step+1>=len(self.trace)),endpoint_extension=int(endpoint_extension))

"""Short physical audit of the exact test1 command/controller chain. No RL training."""
import argparse,json,math,csv,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 app=AppLauncher(args).app
 try:
  import torch
  import isaaclab.sim as sim_utils
  from isaaclab.assets import Articulation
  from isaacsim.core.utils.prims import create_prim
  from isaaclab.utils.math import quat_apply_inverse,yaw_quat
  from train_general_big_map_test1.scene import spot_cfg
  from train_general_big_map_test1.actions import navigation_commands
  from train_general_big_map_test1.locomotion import BatchedSpotVelocityController
  from train_general_big_map_test1.layouts import TRAIN_LAYOUTS
  sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
  create_prim('/World/Map','Xform',usd_path=str(TRAIN_LAYOUTS[0].usd))
  cfg=spot_cfg();cfg.prim_path='/World/Spot';robot=Articulation(cfg)
  sim.reset();robot.update(.005)
  controller=BatchedSpotVelocityController(robot,1,sim.device)
  def yaw():
   q=robot.data.root_quat_w[0];return float(torch.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
  def tick(action):
   command=navigation_commands(torch.tensor([action],device=sim.device))
   for _ in range(40):
    if not app.is_running():raise KeyboardInterrupt
    controller.apply(command);robot.write_data_to_sim();sim.step(render=False);robot.update(.005)
   if float(robot.data.projected_gravity_b[0,2])>-.5:raise RuntimeError('Robot fell in flat-ground coherence check')
  root=robot.data.default_root_state.clone();root[:,:3]=torch.tensor([[0.,0.,.75]],device=sim.device);root[:,3:7]=torch.tensor([[1.,0.,0.,0.]],device=sim.device);root[:,7:]=0
  robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.joint_vel));robot.reset();robot.update(.005)
  for _ in range(10):tick((-1.,0.))
  stance=robot.data.root_state_w.clone();joints=robot.data.joint_pos.clone()
  def reset(angle):
   root=stance.clone();root[:,:2]=0;root[:,3:7]=torch.tensor([[math.cos(angle/2),0.,0.,math.sin(angle/2)]],device=sim.device);root[:,7:]=0
   robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(joints,torch.zeros_like(joints));robot.reset();robot.update(.005);controller.reset(torch.tensor([0],device=sim.device));controller.counter=0
   for _ in range(5):tick((-1.,0.))
  checks=[];samples=[]
  # Exact observation expression used by test1: inverse yaw rotation of goal minus base.
  for heading in [0.,math.pi/2,math.pi,-math.pi/2]:
   q=torch.tensor([[math.cos(heading/2),0.,0.,math.sin(heading/2)]],device=sim.device)
   base=torch.tensor([[11.,-7.,.5]],device=sim.device)
   for local in [(2.,0.),(0.,2.),(0.,-2.),(-2.,0.)]:
    dx=math.cos(heading)*local[0]-math.sin(heading)*local[1];dy=math.sin(heading)*local[0]+math.cos(heading)*local[1]
    goal=base+torch.tensor([[dx,dy,0.]],device=sim.device)
    result=quat_apply_inverse(yaw_quat(q),goal-base)[0,:2]
    assert torch.allclose(result,torch.tensor(local,device=sim.device),atol=2e-6)
  checks.append(dict(test='relative_goal_16_cases',passed=True))
  for sign in [-1,1]:
   reset(0.);initial=yaw()
   for _ in range(8):tick((-1.,sign*.4))
   delta=math.atan2(math.sin(yaw()-initial),math.cos(yaw()-initial))
   checks.append(dict(test=f'yaw_sign_{sign}',yaw_change_rad=delta,passed=sign*delta>.2))
  for heading in [0.,math.pi/2,math.pi,-math.pi/2]:
   reset(heading);start=robot.data.root_pos_w[0,:2].clone();actual_yaw=yaw()
   for _ in range(6):tick((-.375,0.)) # +0.5 m/s
   delta=robot.data.root_pos_w[0,:2]-start
   forward=float(delta[0]*math.cos(actual_yaw)+delta[1]*math.sin(actual_yaw))
   side=float(-delta[0]*math.sin(actual_yaw)+delta[1]*math.cos(actual_yaw))
   checks.append(dict(test=f'forward_heading_{heading:.3f}',forward_m=forward,lateral_m=side,passed=forward>.25 and abs(side)<.25))
  for name,goal in [('front',(1.4,0.)),('left',(0.,.8)),('right',(0.,-.8)),('behind',(-1.4,0.))]:
   reset(0.);target=torch.tensor([[*goal,0.]],device=sim.device)
   initial=float((target[0,:2]-robot.data.root_pos_w[0,:2]).norm());distance=initial
   for step in range(75):
    relative=quat_apply_inverse(yaw_quat(robot.data.root_quat_w),target-robot.data.root_pos_w)[0,:2]
    error=float(torch.atan2(relative[1],relative[0]));distance=float(relative.norm())
    if distance<.35:break
    vx=.4*max(0.,math.cos(error)) if abs(error)<.45 else 0.
    wz=max(-1.,min(1.,2*error))
    tick((vx/.8-1,wz/1.5))
    xy=robot.data.root_pos_w[0,:2].tolist()
    distance=float((target[0,:2]-robot.data.root_pos_w[0,:2]).norm())
    samples.append(dict(test=name,step=step+1,x=xy[0],y=xy[1],goal_x=goal[0],goal_y=goal[1],yaw=yaw(),error=error,vx=vx,wz=wz,distance=distance))
   checks.append(dict(test=f'go_to_{name}',initial_distance_m=initial,final_distance_m=distance,passed=distance<.35))
  passed=all(x['passed'] for x in checks)
  report=dict(passed=passed,checks=checks,note='One robot on asphalt of frozen v2 map; actual test1 action mapping and locomotion weights. No DINO, PPO or learned navigation policy. Does not validate rough-terrain traversability.')
  (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  with (args.output/'trajectories.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
  print('[COHERENCE]',json.dumps(report),flush=True)
  if not passed:raise RuntimeError('Coherence audit failed; inspect report before creating next training')
 finally:app.close(skip_cleanup=args.headless)
if __name__=='__main__':main()

"""Contracts for asynchronous alignment, exact RL budget and independent optimizers."""
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path
import sys
import math
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test3.alignment import InitialAlignment, heading_error
from train_general_big_map_test3.actions import navigation_commands
from train_general_big_map_test3.config import ALIGN_CFG, PPO_CFG, OBSERVATION_SIZE, CRITIC_SIZE
from train_general_big_map_test3.ppo import PPO, packed_gae, explained_variance
from train_general_big_map_test3.rollouts import collect_rollout
from train_general_big_map_test3.test_offline import synthetic_observations, environment_method

torch.set_num_threads(2)


def test_alignment_commands_wrap_sign_stability_and_independent_resets():
    machine=InitialAlignment(3,'cpu')
    machine.active[2]=False
    policy=torch.tensor([[.4,.5],[.3,.2],[.2,-.4]])
    error=heading_error(torch.tensor([[0.,1.],[0.,-1.],[-1.,-.01]]),torch.zeros(3,2),torch.zeros(3))
    command=navigation_commands(machine.actions(policy,error))
    torch.testing.assert_close(command[:2,0],torch.zeros(2))
    assert command[0,2]>.0 and command[1,2]<.0 and abs(error[2])<math.pi
    torch.testing.assert_close(command[2],navigation_commands(policy)[2])
    for e in (.1,.3,.1,.1): machine.advance(torch.tensor([e,.6,0.]))
    assert machine.active.tolist()==[True,True,False]  # out-of-tolerance reset the streak
    machine.advance(torch.tensor([.1,.6,0.]))
    assert machine.active.tolist()==[False,True,False]
    assert machine.succeeded.tolist()==[True,False,False]
    machine.reset(torch.tensor([2]))
    assert machine.active.tolist()==[False,True,True]
    assert machine.steps.tolist()==[5,5,0]


def test_alignment_timeout_and_success_on_deadline():
    m=InitialAlignment(2,'cpu',replace(ALIGN_CFG,max_seconds=.6))
    for _ in range(3):m.advance(torch.tensor([.01,1.]))
    assert m.succeeded.tolist()==[True,False]
    assert m.failed.tolist()==[False,True]
    assert not m.active.any()


def test_packed_gae_respects_env_identity_terminals_and_final_bootstrap():
    # Env0 has a terminal followed by a new episode; env1 has a nonterminal
    # rollout cut. Ordering interleaves robots, alignment rows are absent.
    rewards=torch.tensor([[1.,2.],[10.,20.],[3.,4.],[30.,40.],[5.,6.]])
    values=torch.zeros_like(rewards)
    ends=torch.tensor([False,False,True,False,False])
    next_v=torch.tensor([[0.,0.],[0.,0.],[999.,999.],[7.,8.],[9.,10.]])
    ids=torch.tensor([0,1,0,1,0])
    advantage,returns=packed_gae(rewards,values,ends,next_v,ids,.9,1.)
    expected=torch.tensor([[3.7,5.6],[42.67,62.48],[3.,4.],[36.3,47.2],[13.1,15.]])
    torch.testing.assert_close(returns,expected)
    torch.testing.assert_close(advantage,expected)


class FakeEnv:
    def __init__(self): self.t=0
    def observation(self):
        a=torch.zeros(2,OBSERVATION_SIZE)
        a[:,0]=torch.tensor([self.t,100+self.t])
        return dict(policy=a,critic=torch.cat((a,torch.zeros(2,24)),1))
    def step(self,actions):
        masks=[torch.tensor([False,True]),torch.tensor([True,False]),torch.tensor([True,True])]
        mask=masks[self.t]
        self.t+=1
        return self.observation(),None,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),dict(
            policy_mask=mask,reward_components=torch.full((2,2),float(self.t)),episodes=[])


class FakePolicy:
    normalizer=NS(critic=lambda x:x)
    def critic(self,x):return x[:,:1].repeat(1,2)
    def act(self,actor,critic):
        return torch.zeros(2,2),torch.zeros(2,2),torch.zeros(2),self.critic(critic)


def test_collector_excludes_alignment_and_preserves_each_next_state():
    e=FakeEnv()
    b,_,_,s=collect_rollout(e,FakePolicy(),e.observation(),'cpu',3,10)
    assert b['env_ids'].tolist()==[1,0,0,1]
    assert b['actor'][:,0].tolist()==[100,1,2,102]
    assert b['next_values'][:,0].tolist()==[101,2,3,103]
    assert s['alignment_transitions']==2 and s['rollout_transitions']==4
    assert not s['discarded_final_samples']
    e=FakeEnv()
    b,_,_,s=collect_rollout(e,FakePolicy(),e.observation(),'cpu',3,3)
    assert len(b['actor'])==3 and s['discarded_final_samples']==1
    assert b['actor'][:,0].tolist()==[100,1,2]


def test_critic_keeps_learning_when_actor_stops_and_privileges_do_not_leak():
    learner=PPO(settings=replace(PPO_CFG,epochs=2,batch_size=8))
    a,c=synthetic_observations(16)
    learner.policy.normalizer.update(a,c)
    with torch.no_grad(): _,latent,logp,v=learner.policy.act(a,c)
    old_actor={k:v.clone() for k,v in learner.policy.actor.state_dict().items()}
    old_std=learner.policy.log_std.detach().clone()
    old_critic={k:v.clone() for k,v in learner.policy.critic.state_dict().items()}
    batch=dict(actor=a,critic=c,latent=latent,log_prob=logp-1.,values=v,
               rewards=torch.ones(16,2)*20,terminals=torch.ones(16,dtype=torch.bool),
               next_values=torch.zeros(16,2),env_ids=torch.arange(16))
    metrics=learner.update(batch)
    assert metrics['early_stop']==1 and metrics['actor_minibatch_updates']==0
    assert metrics['critic_minibatch_updates']==4
    assert metrics['nav_mse_after']<metrics['nav_mse_before']
    assert all(torch.equal(v,learner.policy.actor.state_dict()[k]) for k,v in old_actor.items())
    assert torch.equal(old_std,learner.policy.log_std)
    assert any(not torch.equal(v,learner.policy.critic.state_dict()[k]) for k,v in old_critic.items())
    assert metrics['nav_ev_after'] is None  # constant target; not fabricated as zero
    assert sum(p.numel() for p in learner.policy.actor.parameters())==4039906
    assert not ({id(p) for p in learner.actor_parameters}&{id(p) for p in learner.critic_parameters})


def test_alignment_failure_terminates_and_is_penalized():
    from train_general_big_map_test3.config import NAV_CFG
    done=environment_method('_get_dones',terrain_height=lambda _:torch.zeros(1))
    e=NS(nav=NAV_CFG,geometry=NS(world_x_bounds=(-3,39),world_y_bounds=(-1.5,19.5)),
         scene=NS(env_origins=torch.zeros(1,3)),
         robot=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.6]]),projected_gravity_b=torch.tensor([[0.,0.,-1.]]))),
         alignment=NS(failed=torch.tensor([True])),episode_length_buf=torch.tensor([60]),deadline_steps=torch.tensor([150]),
         _distance=lambda:torch.tensor([2.]))
    terminal,_=done(e)
    assert terminal.item() and e.alignment_timeout.item() and not e.timeout.item()

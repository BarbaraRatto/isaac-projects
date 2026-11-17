"""Overlay original routes and measured swapped-map routes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from references import HERE,rows

COLORS=('#166CA6','#D47716','#20815A','#95469F','#C34B59')


def plot_results(output,layout,cases):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})
    palette=json.loads((HERE/'inputs/terrain_palette.json').read_text())
    samples=rows(output/'trajectories.csv');records=rows(output/'episodes.csv')
    references={c['mission']['mission_id']:rows(HERE/'inputs'/c['reference_file']) for c in cases}
    def draw(axis,selected):
        for r,row in enumerate(layout.grid):
            for c,name in enumerate(row):
                x,y=2*c-1,2*r-1
                axis.add_patch(Rectangle((x,y),2,2,facecolor=palette[name]['color'],edgecolor='white',lw=1.2))
                axis.text(x+.1,y+1.8,f'{r},{c}',fontsize=10,color='#526071')
                axis.text(x+1,y+1,palette[name]['label'].capitalize(),ha='center',va='center',fontsize=17,color='#465262')
        handles=[]
        for i,case in enumerate(selected):
            mission=case['mission'];name=mission['mission_id'];color=COLORS[i%5]
            ref=references[name];actual=[r for r in samples if r['mission_id']==name]
            axis.plot([float(r['x_m']) for r in ref],[float(r['y_m']) for r in ref],color=color,lw=1.5,ls='--',alpha=.7)
            axis.plot([float(r['x_m']) for r in actual],[float(r['y_m']) for r in actual],color=color,lw=2.4)
            x,y=mission['start_x_m'],mission['start_y_m']
            axis.scatter(x,y,s=40,color=color,edgecolor='white',zorder=5)
            axis.annotate(name,(x,y),xytext=(5,6+13*i),textcoords='offset points',color=color,fontsize=11)
            axis.scatter(mission['goal_x_m'],mission['goal_y_m'],marker='*',s=150,color=color,edgecolor='white',zorder=5)
            result=next(r for r in records if r['mission_id']==name)
            if result['reason']!='goal':
                axis.scatter(float(actual[-1]['x_m']),float(actual[-1]['y_m']),marker='x',s=70,color=color,zorder=6)
            handles.append(Line2D([],[],color=color,lw=2.8,label=f"{name} · {result['reason']}"))
        handles.extend([Line2D([],[],color='#4F5860',lw=2.4,label='Measured route on swapped map'),
                        Line2D([],[],color='#4F5860',lw=1.5,ls='--',label='Original reference route')])
        axis.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=11)
        axis.set(xlim=(-1,5),ylim=(-1,5),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
        axis.set_xticks((0,2,4));axis.set_yticks((0,2,4))
    fig,axis=plt.subplots(figsize=(12,8),constrained_layout=True)
    draw(axis,cases);axis.set_title(f"Original routes on rocks · {cases[0]['group']}",fontsize=16)
    for extension in ('pdf','png'):fig.savefig(output/f'routes_overlay.{extension}',dpi=170)
    plt.close(fig)
    with PdfPages(output/'routes_per_mission.pdf') as route_pdf, PdfPages(output/'commands_and_tracking.pdf') as command_pdf:
        for case in cases:
            name=case['mission']['mission_id'];data=[r for r in samples if r['mission_id']==name]
            fig,axis=plt.subplots(figsize=(12,8),constrained_layout=True);draw(axis,[case])
            axis.set_title(f'{name} · original reference and measured rock route',fontsize=16)
            route_pdf.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
            time=[float(r['time_s']) for r in data]
            ref=references[name]
            axes[0].step([float(r['time_s']) for r in ref],[float(r['command_forward_m_s']) for r in ref],where='pre',label='Original forward command',ls='--')
            axes[0].step(time,[float(r['command_forward_m_s']) for r in data],where='pre',label='Applied forward command')
            axes[0].plot(time,[float(r['speed_xy_m_s']) for r in data],label='Measured planar speed')
            axes[0].set_ylabel('Speed [m/s]');axes[0].legend()
            axes[1].plot(time,[float(r['command_yaw_rad_s']) for r in data],label='Tracking yaw command')
            axes[1].plot(time,[float(r['source_yaw_command_rad_s']) for r in data],label='Original yaw command',ls='--')
            axes[1].set_ylabel('Yaw rate [rad/s]');axes[1].legend()
            axes[2].plot(time,[float(r['tracking_error_m']) for r in data])
            axes[2].set(ylabel='Distance from reference [m]',xlabel='Simulated time [s]')
            fig.suptitle(name);command_pdf.savefig(fig);plt.close(fig)

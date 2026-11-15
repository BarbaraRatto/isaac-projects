"""Mission plans and recorded routes on the seven-terrain map."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from missions import HERE, METADATA, load_suite

STYLES={'t1_asphalt':('Asphalt','#C4C3C0'),'t2_slippery':('Ice','#B7DEF4'),
        't3_ramp':('Ramp','#A4A3A0'),'t4_fine_gravel':('Gravel','#E3D2C5'),
        't5_large_rocks':('Rocks','#BFAEA8'),'t6_obstacles':('Obstacles','#858481'),
        't7_stairs':('Stairs','#EEE9DD')}
COLORS=['#166CA6','#D47716','#20815A','#95469F','#C34B59',
        '#167C86','#8E5A3A','#3C4B99','#B44384','#6D7825']


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
                         'font.size':13,'pdf.fonttype':42})


def draw_map(axis,meta,scale=1):
    for r,row in enumerate(meta['grid']):
        for c,name in enumerate(row):
            label,color=STYLES[name]
            x,y=6*c-3,3*r-1.5
            axis.add_patch(Rectangle((x,y),6,3,facecolor=color,edgecolor='white',lw=1.1))
            axis.text(x+.2,y+2.6,f'{r},{c}',fontsize=12*scale,color='#4F5356',va='top')
            axis.text(x+3,y+1.5,label,fontsize=18*scale,color='#3F4141',ha='center',va='center')
    axis.set(xlim=(-3,39),ylim=(-1.5,19.5),aspect='equal')
    axis.set_xticks(range(0,40,6)); axis.set_yticks(range(0,20,3))
    axis.tick_params(labelsize=14*scale)
    axis.set_xlabel('X [m]',fontsize=16*scale)
    axis.set_ylabel('Y [m]',fontsize=16*scale)


def draw_mission(axis,case,rows=None):
    color=COLORS[(case['episode']-1)%10]
    a,b=case['mission']['start'],case['mission']['goal']
    if rows is None:
        axis.plot([a[0],b[0]],[a[1],b[1]],'--',color=color,lw=2,zorder=4)
    elif rows:
        axis.plot([float(r['x_m']) for r in rows],[float(r['y_m']) for r in rows],color=color,lw=2.3,zorder=4)
        axis.scatter(float(rows[-1]['x_m']),float(rows[-1]['y_m']),marker='x',color=color,s=40,zorder=6)
    axis.scatter(a[0],a[1],marker='o',color=color,edgecolor='white',s=40,zorder=5)
    axis.annotate(f"E{case['episode']}",xy=(a[0],a[1]),xytext=(7,8),textcoords='offset points',
                  fontsize=13,fontfamily='DejaVu Sans',color=color,fontweight='bold',zorder=8,
                  bbox=dict(boxstyle='square,pad=0.14',facecolor='white',edgecolor='none',alpha=.96))
    axis.scatter(b[0],b[1],marker='*',color=color,edgecolor='white',s=180,zorder=6)
    axis.add_patch(Circle(b[:2],.55,fill=False,ls='--',color=color,lw=.8,zorder=3))


def pages(document,metadata,target,paths=None,results=None):
    cases=document['cases']
    with PdfPages(target) as pdf:
        for offset in range(0,len(cases),4):
            figure,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
            for axis,case in zip(axes.flat,cases[offset:offset+4]):
                draw_map(axis,metadata,.65)
                draw_mission(axis,case,None if paths is None else paths.get(case['episode'],[]))
                caption=f"Ep {case['episode']} · {case['distance_group_m']} m · {case['category'].replace('_',' ')}"
                if results and case['episode'] in results:
                    row=results[case['episode']]
                    caption+=f"\n{row['reason']} · {float(row['energy_j']):.0f} J · {float(row['path_m']):.2f} m · {float(row['time_s']):.1f} s"
                axis.set_title(caption,fontsize=12)
            for axis in list(axes.flat)[len(cases[offset:offset+4]):]:
                axis.set_visible(False)
            figure.suptitle('Planned start–goal segments — not robot trajectories' if paths is None else 'Recorded trajectories — all outcomes',fontsize=18)
            pdf.savefig(figure)
            if paths is None and offset==0:
                figure.savefig(target.with_name(target.stem+'_preview.png'),dpi=140)
            plt.close(figure)


def plan(document):
    style()
    metadata=json.loads(METADATA.read_text())
    for distance in (18,):
        subset=dict(document,cases=[c for c in document['cases'] if c['distance_group_m']==distance])
        if subset['cases']:
            pages(subset,metadata,HERE/f'missions_plan_{distance}m.pdf')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def plot_results(output,document):
    style()
    metadata=json.loads((output/'inputs'/METADATA.name).read_text())
    episodes=read_csv(output/'episodes.csv')
    trajectories=read_csv(output/'trajectories.csv')
    results={int(row['episode']):row for row in episodes}
    paths={}
    for row in trajectories:
        paths.setdefault(int(row['episode']),[]).append(row)
    pages(document,metadata,output/'routes_per_episode.pdf',paths,results)
    with PdfPages(output/'routes_by_category.pdf') as pdf:
        groups=sorted({(c['distance_group_m'],c['category']) for c in document['cases']})
        for distance,category in groups:
            grouped=[c for c in document['cases'] if (c['distance_group_m'],c['category'])==(distance,category)]
            for offset in range(0,len(grouped),10):
                fig,axis=plt.subplots(figsize=(15,9),constrained_layout=True)
                draw_map(axis,metadata)
                handles=[]
                for case in grouped[offset:offset+10]:
                    if case['episode'] not in results:
                        continue
                    draw_mission(axis,case,paths.get(case['episode'],[]))
                    handles.append(Line2D([],[],color=COLORS[(case['episode']-1)%10],lw=2.3,
                        label=f"E{case['episode']}: {results[case['episode']]['reason']}"))
                batch=offset//10+1
                axis.set_title(f"{distance} m · {category.replace('_',' ')} · group {batch}",fontsize=20)
                axis.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=5,frameon=False,fontsize=11)
                pdf.savefig(fig,bbox_inches='tight')
                fig.savefig(output/f'routes_{distance}m_{category}_group_{batch:02d}.png',dpi=160,bbox_inches='tight')
                plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(15,10),constrained_layout=True)
    numbers=[int(r['episode']) for r in episodes]
    for axis,key,label in zip(axes,('path_m','time_s','energy_j'),('Traveled distance [m]','Simulated time [s]','Absolute joint work [J]')):
        axis.bar(numbers,[float(r[key]) for r in episodes],color=[COLORS[(n-1)%10] for n in numbers])
        axis.set_ylabel(label)
        axis.set_xticks(numbers[::max(1,len(numbers)//25)])
        axis.grid(axis='y',alpha=.2)
    axes[-1].set_xlabel('Episode')
    fig.suptitle('Mission measurements — all outcomes; incomplete missions are not energy savings')
    for ext in ('pdf','png'):
        fig.savefig(output/f'mission_metrics.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.plan:
        plan(load_suite())
    elif args.run:
        config=json.loads((args.run/'evaluation_config.json').read_text())
        document=json.loads((args.run/'inputs'/config['mission_file']).read_text())
        plot_results(args.run,document)
    else:
        parser.error('Choose --plan or --run PATH')

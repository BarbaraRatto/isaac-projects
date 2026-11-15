"""Plot the selected illustrative episodes with their collection IDs."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
CASE=HERE.parents[1]
STYLES={'t1_asphalt':('Asphalt','#C4C3C0'),'t2_slippery':('Ice','#B7DEF4'),
        't3_ramp':('Ramp','#A4A3A0'),'t4_fine_gravel':('Gravel','#E3D2C5'),
        't5_large_rocks':('Rocks','#BFAEA8'),'t6_obstacles':('Obstacles','#858481'),
        't7_stairs':('Stairs','#EEE9DD')}
COLORS=['#166CA6','#D47716','#20815A','#95469F','#C34B59','#167C86','#8E5A3A','#3C4B99','#B44384','#6D7825']


def main():
    document=json.loads((HERE/'mission_list.json').read_text())
    if hashlib.sha256((CASE/'maps/real_terrains_v2_eval_01.usd').read_bytes()).hexdigest()!=document['map_sha256']:
        raise ValueError('The shared map differs from the archived collection')
    grid=json.loads((CASE/'maps/real_terrains_v2_eval_01_metadata.json').read_text())['grid']
    slots=[s for s in document['slots'] if s['status']=='selected']
    if not slots:
        return
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})
    def draw(axis,items):
        for r,row in enumerate(grid):
            for c,name in enumerate(row):
                label,color=STYLES[name]
                x,y=6*c-3,3*r-1.5
                axis.add_patch(Rectangle((x,y),6,3,facecolor=color,edgecolor='white',lw=1.1))
                axis.text(x+.2,y+2.6,f'{r},{c}',fontsize=12,color='#4F5356',va='top')
                axis.text(x+3,y+1.5,label,fontsize=18,color='#3F4141',ha='center',va='center')
        for slot in items:
            n=slot['episode']; color=COLORS[(n-1)%len(COLORS)]
            with (HERE/slot['episode_directory']/'trajectories.csv').open() as stream:
                data=list(csv.DictReader(stream))
            m=slot['case']['mission']; a,b=m['start'],m['goal']
            axis.plot([float(r['x_m']) for r in data],[float(r['y_m']) for r in data],color=color,lw=2.3,zorder=4,label=f'E{n}'+(' ['+slot['result']['reason']+']' if slot['result']['reason']!='goal' else '')+(' (tentative)' if slot.get('review_status')=='tentative' else ''))
            axis.scatter(a[0],a[1],s=40,color=color,edgecolor='white',zorder=5)
            axis.annotate(f'E{n}',xy=(a[0],a[1]),xytext=(7,8),textcoords='offset points',fontsize=14,fontfamily='DejaVu Sans',
                          color=color,fontweight='bold',zorder=8,bbox=dict(boxstyle='square,pad=0.14',facecolor='white',edgecolor='none',alpha=.96))
            axis.scatter(b[0],b[1],marker='*',s=180,color=color,edgecolor='white',zorder=5)
            axis.scatter(float(data[-1]['x_m']),float(data[-1]['y_m']),marker='x',s=40,color=color,zorder=6)
        axis.set(xlim=(-3,39),ylim=(-1.5,19.5),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
        axis.set_xticks(range(0,40,6));axis.set_yticks(range(0,20,3));axis.tick_params(labelsize=14)
        axis.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=12,ncol=2 if len(items)>24 else 1)
    figure,axis=plt.subplots(figsize=(15,8.5),constrained_layout=True)
    draw(axis,slots)
    axis.set_title(f'Selected illustrative episodes · {len(slots)} archived',fontsize=20)
    for ext in ('png','pdf'):
        figure.savefig(HERE/f'selected_routes.{ext}',dpi=170)
    plt.close(figure)
    for category in document['category_targets']:
        items=[slot for slot in slots if slot['case']['category']==category]
        if not items:
            continue
        figure,axis=plt.subplots(figsize=(15,8.5),constrained_layout=True)
        draw(axis,items)
        title=category.replace('_',' ').title()
        axis.set_title(f'Selected routes · {title} · {len(items)} episodes',fontsize=20)
        for ext in ('png','pdf'):
            figure.savefig(HERE/f'selected_routes_{category}.{ext}',dpi=170)
        plt.close(figure)
    with PdfPages(HERE/'selected_routes_per_episode.pdf') as pdf:
        for slot in slots:
            figure,axis=plt.subplots(figsize=(15,8.5),constrained_layout=True)
            draw(axis,[slot])
            axis.set_title(f"Selected episode E{slot['episode']} · {slot['case']['category'].replace('_',' ')}"+(' · tentative' if slot.get('review_status')=='tentative' else ''),fontsize=20)
            pdf.savefig(figure)
            plt.close(figure)


if __name__=='__main__':
    main()

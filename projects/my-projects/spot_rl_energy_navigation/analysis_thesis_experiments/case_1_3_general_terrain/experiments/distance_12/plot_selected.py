"""Plot archived twelve-metre routes and highlighted figure candidates."""
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
TITLES={'regular':'Regular','ice':'Ice','stairs':'Stairs','surface_choice':'Surface choice'}


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    document=json.loads((HERE/'mission_list.json').read_text())
    if hashlib.sha256((CASE/'maps/real_terrains_v2_eval_01.usd').read_bytes()).hexdigest()!=document['map_sha256']:
        raise ValueError('The shared map differs from the archived collection')
    grid=json.loads((CASE/'maps/real_terrains_v2_eval_01_metadata.json').read_text())['grid']
    slots=[s for s in document['slots'] if s['status']=='selected']
    if not slots:
        return
    preferred=[s for s in slots if s.get('preferred_for_plot',False)]
    route_colors={}
    for category in document['category_targets']:
        items=sorted((s for s in slots if s['case']['category']==category),
                     key=lambda s:(not s.get('preferred_for_plot',False),s['source_episode']))
        for index,slot in enumerate(items):
            route_colors[slot['episode']]=COLORS[index%len(COLORS)]
    traces={s['episode']:read_rows(HERE/s['episode_directory']/'trajectories.csv') for s in slots}
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})

    def draw(axis,items,collection_labels=False):
        for r,row in enumerate(grid):
            for c,name in enumerate(row):
                label,color=STYLES[name]
                x,y=6*c-3,3*r-1.5
                axis.add_patch(Rectangle((x,y),6,3,facecolor=color,edgecolor='white',lw=1.1))
                axis.text(x+.2,y+2.6,f'{r},{c}',fontsize=10,color='#4F5356',va='top')
                axis.text(x+3,y+1.5,label,fontsize=16,color='#3F4141',ha='center',va='center')
        for slot in items:
            n=slot['source_episode'];color=route_colors[slot['episode']]
            data=traces[slot['episode']]
            m=slot['case']['mission'];a,b=m['start'],m['goal']
            label=(f'E{slot["episode"]} (Ep {n})' if collection_labels else f'Ep {n} (E{slot["episode"]})')
            if slot['result']['reason']!='goal':
                label+=f' [{slot["result"]["reason"]}]'
            if slot.get('review_status')=='tentative':
                label+=' (tentative)'
            axis.plot([float(r['x_m']) for r in data],[float(r['y_m']) for r in data],
                      color=color,lw=2.3,zorder=4,label=label)
            axis.scatter(a[0],a[1],s=40,color=color,edgecolor='white',zorder=5)
            label_id=slot['episode'] if collection_labels else n
            axis.annotate(f'E{label_id}',xy=(a[0],a[1]),xytext=(7,8),textcoords='offset points',
                          fontsize=12,fontfamily='DejaVu Sans',color=color,fontweight='bold',zorder=8,
                          bbox=dict(boxstyle='square,pad=0.14',facecolor='white',edgecolor='none',alpha=.96))
            axis.scatter(b[0],b[1],marker='*',s=180,color=color,edgecolor='white',zorder=5)
            axis.scatter(float(data[-1]['x_m']),float(data[-1]['y_m']),marker='x',s=40,color=color,zorder=6)
        axis.set(xlim=(-3,39),ylim=(-1.5,19.5),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
        axis.set_xticks(range(0,40,6));axis.set_yticks(range(0,20,3));axis.tick_params(labelsize=13)
        if items:
            axis.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=10,
                        ncol=2 if len(items)>14 else 1)

    def overview(items,stem,title,collection_labels=False):
        figure,axes=plt.subplots(2,2,figsize=(30,18),constrained_layout=True)
        for axis,category in zip(axes.flat,document['category_targets']):
            subset=[s for s in items if s['case']['category']==category]
            draw(axis,subset,collection_labels=collection_labels)
            axis.set_title(f'{TITLES[category]} · {len(subset)} episodes',fontsize=20)
        label_note=('Labels: collection IDs; legend parentheses: original try1 episode IDs' if collection_labels
                    else 'Labels: try1 episode IDs; legend parentheses: collection IDs')
        figure.suptitle(f'{title} · 12 m · {len(items)} episodes\n{label_note}',fontsize=22)
        for ext in ('png','pdf'):
            figure.savefig(HERE/f'{stem}.{ext}',dpi=150)
        plt.close(figure)

    overview(slots,'selected_routes','Selected illustrative routes',collection_labels=True)
    overview(preferred,'preferred_routes','Highlighted candidates for route figures')
    for items,target in ((slots,HERE/'selected_routes_per_episode.pdf'),
                         (preferred,HERE/'preferred/routes_per_episode.pdf')):
        target.parent.mkdir(exist_ok=True)
        with PdfPages(target) as pdf:
            for slot in items:
                figure,axis=plt.subplots(figsize=(15,8.5),constrained_layout=True)
                draw(axis,[slot])
                axis.set_title(f'Try1 Ep {slot["source_episode"]} · Collection E{slot["episode"]} · '
                               f'{TITLES[slot["case"]["category"]]}',fontsize=20)
                pdf.savefig(figure)
                plt.close(figure)


if __name__=='__main__':
    main()

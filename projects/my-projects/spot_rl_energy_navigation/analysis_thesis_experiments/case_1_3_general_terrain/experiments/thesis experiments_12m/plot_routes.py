"""Plot the forty v2 selections without changing the original collection."""
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
CASE=HERE.parents[1]
SOURCE=HERE.parent/'distance_12'
sys.path.insert(0,str(SOURCE))
from plot_selected import STYLES, TITLES, COLORS, read_rows


def main(show_only=False):
    document=json.loads((HERE/'selected_missions_v2.json').read_text())
    source=(HERE/document['source_collection']).resolve()
    terrain=CASE/'maps/real_terrains_v2_eval_01.usd'
    if hashlib.sha256(terrain.read_bytes()).hexdigest()!=document['map_sha256']:
        raise ValueError('Map differs from the selection snapshot')
    grid=json.loads(terrain.with_name(terrain.stem+'_metadata.json').read_text())['grid']
    slots=document['slots']
    if len(slots)!=40 or len({s['episode'] for s in slots})!=40:
        raise ValueError('Expected forty different archived episodes')
    if [s['thesis_episode'] for s in slots]!=list(range(1,41)):
        raise ValueError('Thesis episode IDs must run from 1 to 40')
    traces={}
    colors={}
    for category,ids in document['selected_ids'].items():
        subset=[s for s in slots if s['case']['category']==category]
        if len(subset)!=10 or [s['episode'] for s in subset]!=ids:
            raise ValueError(f'Invalid category selection: {category}')
        for index,slot in enumerate(subset):
            n=slot['episode']
            data=read_rows(source/slot['episode_directory']/'trajectories.csv')
            if not data or int(data[-1]['terminal'])!=1 or any(int(r['episode'])!=n for r in data):
                raise ValueError(f'Invalid archived trajectory for E{n}')
            if abs(float(data[-1]['energy_j'])-float(slot['result']['energy_j']))>.001:
                raise ValueError(f'Terminal energy mismatch for E{n}')
            traces[n]=data;colors[n]=COLORS[index]
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})

    def draw(axis,items,category_colors=False):
        for r,row in enumerate(grid):
            for c,name in enumerate(row):
                label,color=STYLES[name];x,y=6*c-3,3*r-1.5
                axis.add_patch(Rectangle((x,y),6,3,facecolor=color,edgecolor='white',lw=1.1))
                axis.text(x+.2,y+2.6,f'{r},{c}',fontsize=12,color='#4F5356',va='top')
                axis.text(x+3,y+1.5,label,fontsize=18,color='#3F4141',ha='center',va='center')
        handles=[]
        for slot in items:
            collection_id=slot['episode'];n=slot['thesis_episode']
            color=show_colors[slot['case']['category']] if category_colors else colors[collection_id]
            data=traces[collection_id]
            mission=slot['case']['mission'];a,b=mission['start'],mission['goal']
            axis.plot([float(row['x_m']) for row in data],[float(row['y_m']) for row in data],
                      color=color,lw=2.6,zorder=4)
            axis.scatter(a[0],a[1],s=55,color=color,edgecolor='white',zorder=5)
            axis.annotate(f'E{n}',xy=a[:2],xytext=(7,8),textcoords='offset points',fontsize=14,
                          fontfamily='DejaVu Sans',fontweight='bold',color=color,zorder=8,
                          bbox=dict(boxstyle='square,pad=0.14',facecolor='white',edgecolor='none',alpha=.96))
            axis.scatter(b[0],b[1],s=200,marker='*',color=color,edgecolor='white',zorder=5)
            reason=slot['result']['reason']
            if reason!='goal':
                axis.scatter(float(data[-1]['x_m']),float(data[-1]['y_m']),s=95,marker='x',color=color,lw=2.4,zorder=6)
            label=(f'{TITLES[slot["case"]["category"]]} · ' if category_colors else '')+f'E{n} (Ep {slot["source_episode"]})'
            if reason!='goal':
                label+=f' [{reason}]'
            handles.append(Line2D([],[],color=color,lw=2.6,label=label))
        for marker,label in [('o','Start'),('*','Goal'),('x','Failure position')]:
            handles.append(Line2D([],[],color='#454B50',marker=marker,linestyle='none',markersize=10,label=label))
        axis.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,
                    title='Episode ID (original try1 ID)',title_fontsize=14,fontsize=12,labelspacing=1.0)
        axis.set(xlim=(-3,39),ylim=(-1.5,19.5),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
        axis.set_xticks(range(0,40,6));axis.set_yticks(range(0,20,3));axis.tick_params(labelsize=14)
        axis.xaxis.label.set_size(16);axis.yaxis.label.set_size(16)

    def save(figure,stem):
        for extension in ('pdf','png'):
            figure.savefig(HERE/f'{stem}.{extension}',dpi=170)
        plt.close(figure)

    if not show_only:
        for index,category in enumerate(document['category_order'],1):
            subset=[s for s in slots if s['case']['category']==category]
            figure,axis=plt.subplots(figsize=(17,9),constrained_layout=True)
            draw(axis,subset)
            axis.set_title(f'Experiments · 12 m · {TITLES[category]} · 10 episodes',fontsize=21,pad=18)
            stem=f'exp_12m_cat{index}_{category}'
            save(figure,stem)
            print(stem)
        figure,axes=plt.subplots(2,2,figsize=(32,19),constrained_layout=True)
        for axis,category in zip(axes.flat,document['category_order']):
            draw(axis,[s for s in slots if s['case']['category']==category])
            axis.set_title(f'{TITLES[category]} · 10 episodes',fontsize=21)
        figure.suptitle('Experiments · 12 m · 40 episodes\nLabels: thesis episode IDs; legend parentheses: original try1 IDs',fontsize=24)
        save(figure,'exp_all_together')
        print('exp_all_together')

    show_document=json.loads((HERE/'show_routes.json').read_text())
    show_lookup={s['episode']:s for s in slots}
    show_colors={item['category']:item['color'] for item in show_document['episodes']}
    show_slots=[]
    for item in show_document['episodes']:
        slot=show_lookup[item['collection_episode']]
        if slot['source_episode']!=item['source_episode']:
            raise ValueError('Display selection source ID changed')
        item['thesis_episode']=slot['thesis_episode']
        show_slots.append(slot)
    show_slots.sort(key=lambda s:s['thesis_episode'])
    figure,axis=plt.subplots(figsize=(18,9),constrained_layout=True)
    draw(axis,show_slots,category_colors=True)
    axis.set_title(f'Show routes · 12 m · {len(show_slots)} episodes',fontsize=21,pad=18)
    save(figure,'show_routes')
    show_document['label_note']='Thesis episode IDs E1–E40 on the map; original try1 IDs in the legend. Collection IDs preserved as provenance.'
    (HERE/'show_routes.json').write_text(json.dumps(show_document,indent=2)+'\n')
    print('show_routes')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-only',action='store_true')
    main(show_only=parser.parse_args().show_only)

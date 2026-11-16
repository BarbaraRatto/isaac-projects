"""Plot the display selection with current collection IDs and source IDs."""
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_selected import HERE, CASE, STYLES, TITLES, read_rows


def main():
    document=json.loads((HERE/'mission_list.json').read_text())
    selection=json.loads((HERE/'show_preferred_selection.json').read_text())
    lookup={s['source_episode']:s for s in document['slots'] if s['status']=='selected'
            and s['source_run']==selection['source_run']}
    wanted=selection['source_episodes']
    if len(set(wanted))!=len(wanted) or not set(wanted)<=set(lookup):
        raise ValueError('Display selection contains duplicate or missing source episodes')
    slots=[lookup[n] for n in wanted]
    colors=selection['category_colors']
    terrain=CASE/'maps/real_terrains_v2_eval_01.usd'
    if hashlib.sha256(terrain.read_bytes()).hexdigest()!=document['map_sha256']:
        raise ValueError('Map differs from the archived collection')
    grid=json.loads(terrain.with_name(terrain.stem+'_metadata.json').read_text())['grid']
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})
    figure,axis=plt.subplots(figsize=(18,9),constrained_layout=True)
    for r,row in enumerate(grid):
        for c,name in enumerate(row):
            label,color=STYLES[name]
            x,y=6*c-3,3*r-1.5
            axis.add_patch(Rectangle((x,y),6,3,facecolor=color,edgecolor='white',lw=1.1))
            axis.text(x+.2,y+2.6,f'{r},{c}',fontsize=12,color='#4F5356',va='top')
            axis.text(x+3,y+1.5,label,fontsize=18,color='#3F4141',ha='center',va='center')
    handles=[]
    mapping=[]
    for category,color in colors.items():
        for slot in slots:
            if slot['case']['category']!=category:
                continue
            n=slot['episode'];source_id=slot['source_episode']
            data=read_rows(HERE/slot['episode_directory']/'trajectories.csv')
            if not data or int(data[-1]['terminal'])!=1:
                raise ValueError(f'Missing terminal state for collection episode {n}')
            mission=slot['case']['mission'];a,b=mission['start'],mission['goal']
            axis.plot([float(row['x_m']) for row in data],[float(row['y_m']) for row in data],
                      color=color,lw=2.6,zorder=4)
            axis.scatter(a[0],a[1],s=55,color=color,edgecolor='white',zorder=5)
            axis.annotate(f'E{n}',xy=a[:2],xytext=(7,8),textcoords='offset points',fontsize=14,
                          fontfamily='DejaVu Sans',fontweight='bold',color=color,zorder=8,
                          bbox=dict(boxstyle='square,pad=0.14',facecolor='white',edgecolor='none',alpha=.96))
            axis.scatter(b[0],b[1],s=200,marker='*',color=color,edgecolor='white',zorder=5)
            reason=slot['result']['reason']
            if reason=='fallen':
                axis.scatter(float(data[-1]['x_m']),float(data[-1]['y_m']),s=95,marker='x',color=color,lw=2.4,zorder=6)
            label=f'{TITLES[category]} · E{n} (Ep {source_id})'
            if reason!='goal':
                label+=f' [{reason}]'
            handles.append(Line2D([],[],color=color,lw=2.6,label=label))
            mapping.append(dict(collection_episode=n,source_episode=source_id,category=category,
                                reason=reason,color=color,episode_directory=slot['episode_directory']))
    for marker,label in [('o','Start'),('*','Goal'),('x','Fall position')]:
        handles.append(Line2D([],[],color='#454B50',marker=marker,linestyle='none',markersize=11,label=label))
    axis.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,
                title='Collection ID (original try1 ID)',title_fontsize=15,fontsize=13,labelspacing=1.05)
    axis.set(xlim=(-3,39),ylim=(-1.5,19.5),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
    axis.set_xticks(range(0,40,6));axis.set_yticks(range(0,20,3));axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(16);axis.yaxis.label.set_size(16)
    axis.set_title('Show preferred routes · 12 m · 10 episodes',fontsize=21,pad=18)
    for extension in ('pdf','png'):
        path=HERE/f'show_preferred_routes.{extension}'
        figure.savefig(path,dpi=170)
        print(path)
    (HERE/'show_preferred_routes.json').write_text(json.dumps(dict(source_run=selection['source_run'],
        label_note='Current collection IDs on the map; original try1 IDs in the legend.',episodes=mapping),indent=2)+'\n')
    plt.close(figure)


if __name__=='__main__':
    main()

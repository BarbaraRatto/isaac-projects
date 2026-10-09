# Medium distance choice

Esperimento separato sul **USD completo da 49 celle**. Spot parte dal centro
della rampa `Cell_1_4_t3_ramp`, `(24, 3)` m, e deve arrivare al centro delle
rocce `Cell_5_0_t5_large_rocks`, `(0, 15)` m. La distanza in linea d'aria è
26,83 m. Lo yaw iniziale punta verso il goal (circa 153,4°). La partenza
varia di ±0,05 m su ciascun asse a ogni reset. Non vengono forniti waypoint,
etichette dei terreni né un percorso alla policy. L'asset USD non viene
modificato.

La linea diretta passa vicino o sopra ghiaccio e scale. Raggiungere il goal
può richiedere una deviazione, ma il successo su questa singola coppia
start/goal non dimostra ancora che Spot sappia generalizzare o riconoscere
il percorso energeticamente migliore in una scena nuova.

## Come funziona

32 robot raccolgono esperienza in copie isolate della scena completa e
aggiornano **una sola** policy PPO condivisa. Ogni robot ha una ZED X simulata
RGB-D 640×360 inclinata di 15° verso il basso. Un solo DINOv2-small
congelato elabora le immagini in microbatch di quattro; la policy riceve
le feature DINO correnti e 11 valori di stato/goal. L'atlante
`runs/ice_choice_atlas.npz` fornisce solo la base PCA per comprimere le
feature a 32 componenti: le feature e le posizioni memorizzate nell'atlante
non vengono usate come osservazioni. La policy IsaacRobotics già esistente
continua a controllare i giunti. PPO comanda `vx`, `vy` e velocità angolare.

| Parametro | Valore |
| --- | ---: |
| Fisica | 0,005 s per passo (200 Hz) |
| Decisione RL | ogni 0,2 s (40 passi fisici) |
| Comandi massimi | `vx` ±1,6 m/s; `vy` ±0,9 m/s; rotazione ±1,5 rad/s |
| Durata massima episodio | 120 s, cioè 600 decisioni |
| Goal raggiunto | distanza dal centro del goal ≤0,55 m |
| Assestamento iniziale | 2 s, esclusi dall'energia dell'episodio |
| Transizioni totali predefinite | 450.560, sommate su tutti i robot |
| PPO | 64 passi per robot e aggiornamento (2.048 transizioni con 32 robot), batch 256, 5 epoche, rete [128,128], learning rate 0,0003, gamma 0,999; PPO su CPU |
| Reward per decisione | `+3 × progresso_m − 0,01 × energia_J − 0,01`; `+100` al goal, `−25` se cade/esce |

L'energia è il lavoro meccanico stimato dai momenti misurati dal solver:
`sum(abs(coppia × velocità_angolare_giunto)) × dt`. La penalità energetica
viene applicata a ogni azione e i joule vengono sommati nell'intero episodio.
Il training massimizza reward futura scontata, quindi la policy non esegue
un'ottimizzazione matematica esatta del totale dei joule. Il CSV di
valutazione riporta il **totale** consumato da ciascun episodio.

I flag `ice_entry`, `stairs_entry` e gli altri tipi di terreno indicano che
la posizione XY di almeno un piede è entrata nella cella durante l'episodio.
Sono dati di analisi, non etichette consegnate alla policy e non confermano
una forza di contatto col terreno.

## Comandi dalla cartella `my-projects`

1. Verifica tecnica breve, **senza training**. Controlla caricamento delle 32
scene complete, Spot, ZED, DINO e forma delle osservazioni:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/train.py --headless --num-envs 32 --probe-steps 8
```

2. Solo dopo che il probe ha terminato senza errori, avvia il training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/train.py --headless --num-envs 32 --timesteps 450560
```

Il training crea una nuova cartella `runs/medium_online_32env_<data>/`
con `config.json`, `episodes.csv`, `timing.csv`, `summary.json`, checkpoint
ogni 10.240 transizioni circa e `policy_final.zip` al termine. Se premi
`Ctrl+C`, salva un `policy_interrupted_<passi>_steps.zip`; attendi la riga
`checkpoint=...` prima di chiudere. `--timesteps` indica il **totale**
delle transizioni, non le decisioni di un singolo robot. Al ritmo precedente
con 32 scene, 450.560 transizioni richiederebbero circa 6 ore e 45 minuti
di training; il tempo reale può cambiare con il nuovo PPO e con il carico
GPU/CPU.

3. Per vedere il checkpoint e salvare 20 episodi, sostituisci `<RUN>` con la
cartella indicata dal training. Senza `--headless` apre la simulazione:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/play.py --checkpoint spot_rl_energy_navigation/medium_distance_choice/runs/<RUN>/policy_final.zip --episodes 20
```

La valutazione crea `results/medium_eval_<data>.csv`. Il file contiene
successo, caduta/uscita o timeout, joule totali, distanza, decisioni e
celle attraversate. Confronta il consumo dei **soli episodi arrivati**.

Per riprendere un checkpoint, usa lo stesso numero di robot e lo stesso
atlante PCA, indicando un totale `--timesteps` maggiore dei passi già
salvati:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/train.py --headless --num-envs 32 --timesteps 450560 --resume spot_rl_energy_navigation/medium_distance_choice/runs/<RUN>/policy_interrupted_<PASSI>_steps.zip
```

Il probe serve a rilevare errori tecnici e memoria insufficiente **prima**
del training lungo. La precedente prova su 32 copie del USD completo ha
funzionato su questo PC; tuttavia il probe non può garantire che un processo
resti stabile per molte ore.

## Grafici dei percorsi

`play.py` registra ora, per ogni decisione RL, la posizione XY **del corpo** di
Spot prima di un eventuale reset automatico e l'energia accumulata. Dopo i
20 episodi crea cinque file con lo stesso prefisso in `results/`:

- `medium_eval_<data>.csv`: riepilogo degli episodi;
- `medium_eval_<data>_trajectory.csv`: punti XY, tempo ed energia, inclusi
  il punto iniziale e quello finale degli episodi terminati;
- `medium_eval_<data>_grid.csv`: le 49 celle lette dall'USD in uso;
- `medium_eval_<data>_routes_outcomes.svg`: arrivi verdi e fallimenti rossi;
- `medium_eval_<data>_routes_successes.svg`: ogni arrivo con un colore diverso.

I grafici sono file SVG vettoriali apribili in un browser o nell'anteprima
immagini. La griglia è disegnata con coordinate e terreni letti dall'USD;
X cresce verso destra e Y verso l'alto. Un punto della traiettoria viene
registrato ogni 0,2 s di simulazione. Le linee rappresentano il percorso del
corpo, non le singole impronte dei piedi. Il precedente CSV di 20 episodi non
contiene questi punti: per ottenere i grafici occorre **ripetere soltanto la
valutazione**, non il training.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/play.py --checkpoint spot_rl_energy_navigation/medium_distance_choice/runs/medium_online_32env_20261007_013818/policy_final.zip --episodes 20 --seed 0
```

Per ricreare i grafici da CSV già registrati, senza aprire Isaac Sim:

```bash
python3 spot_rl_energy_navigation/medium_distance_choice/plot_trajectories.py --summary spot_rl_energy_navigation/medium_distance_choice/results/medium_eval_<data>.csv
```

`--seed 0` rende ripetibili le **nuove** valutazioni con la stessa versione dei
file e del simulatore. Non garantisce di ricostruire esattamente i 20 episodi
precedenti, che non avevano un seed esplicito né una traiettoria salvata.

## Tre percorsi disegnati a mano: confronto guidato

La foto con le linee rosa, arancione e gialla è stata trascritta in
`three_route_plan.py`; `draw_three_routes.py` crea la mappa
`results/medium_three_routes_from_sketch.png` usando i terreni letti dall'USD.
Le coordinate sono una **stima dal disegno prospettico**: controllare la mappa
prima di misurarle. La rosa evita il ghiaccio, passa per il centro delle celle
con ostacoli `(r3,c3)` e `(r4,c1)` e segue la linea `y=10,5 m` fra le due
celle con scale. La linea è solo geometrica: il USD potrebbe non avere una
striscia percorribile abbastanza larga per tutti i piedi di Spot. L'arancione
attraversa la rampa `(r2,c1)` e le rocce `(r3,c1)`; la sua linea centrale
resta fuori da ghiaccio, scale e ostacoli. La gialla si allarga verso
l'asfalto della colonna 0. Le lunghezze geometriche sono circa 28,79 m
(rosa), 30,31 m (arancione) e 31,85 m (gialla). L'arancione è quindi 1,54 m
più corta della gialla, mentre la rosa è la più corta delle tre. Le celle
toccate dai piedi e la fattibilità si possono conoscere solo in simulazione.

`compare_three_routes.py` segue ogni linea con Spot e lo stesso controllore
di giunti già usato nel progetto. Non allena RL: comanda velocità di avanzamento
e rotazione con un limite identico di velocità su tutte le vie. La camera e
DINO non servono a un percorso prescritto, quindi sono spenti. Vengono
registrati energia meccanica misurata, percorso effettivo, arrivo, terreni
toccati e scostamento dalla linea. Le medie energetiche includono **solo**
prove arrivate e che hanno seguito abbastanza da vicino la linea assegnata.
Per la rosa si verifica inoltre il passaggio vicino al centro delle due celle
con ostacoli e alla linea fra le scale; per l'arancione si richiede che nessun
piede entri nelle celle di ghiaccio, scale o ostacoli. I fallimenti rimangono
nel CSV. Questi flag indicano posizione dei piedi, non forza di contatto.

Comando di controllo rapido con 3 prove per via, dalla cartella `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/medium_distance_choice/compare_three_routes.py --headless --episodes 3 --speed 0.8
```

Per 20 prove per via, sostituire `--episodes 3` con `--episodes 20`.
Senza `--headless` si apre la simulazione. I file
`results/guided_three_routes_<data>.csv` e
`results/guided_three_routes_<data>_trajectory.csv` contengono risultati
e coordinate reali del corpo. Nessun test è stato eseguito durante la
preparazione di questi file.

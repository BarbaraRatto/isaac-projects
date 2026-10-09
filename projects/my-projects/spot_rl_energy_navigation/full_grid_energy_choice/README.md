# Navigazione energetica sul USD completo

Esperimento separato dal precedente `ramp_energy_choice`. Ogni ambiente carica
**tutte le 49 celle originali** di `terrain_generator/real_terrains.usd`; non
crea una variante ridotta e non modifica il USD.

## Geometria e azioni

Nel USD gli indici dei nomi delle celle partono da zero. La partenza è
`Cell_2_1_t3_ramp` (riga 2, colonna 1), a `(3.7, 6.8)` m: 0,7 m dai
bordi sinistro e superiore della cella. Il goal è in
`Cell_4_1_t6_obstacles` (riga 4, colonna 1), a `(3.7, 11.2)` m:
0,7 m dai bordi sinistro e inferiore. Tra i due punti, nella riga 3,
`Cell_3_0` è asfalto e `Cell_3_1` è rocce. Lo yaw iniziale è 90°,
verso il goal; a ogni reset la partenza varia solo di ±0,05 m.
Il goal ha raggio 0,55 m.

Una policy PPO sceglie `vx`, `vy`, `wz` ogni 0,2 s. I limiti dei comandi sono
rispettivamente **±1,6 m/s**, **±0,9 m/s**, **±1,5 rad/s**. L'esistente policy
IsaacRobotics controlla i giunti. La velocità fisica ottenuta può essere
inferiore al comando.

Ogni Spot ha una ZED X simulata RGB-D 640×360 inclinata di 15° verso il basso.
Un DINOv2-small congelato, condiviso fra gli ambienti, estrae le feature in
tempo reale. Si riusano soltanto media e assi PCA del file già esistente
`spot_rl_energy_navigation/runs/ice_choice_atlas.npz` per comprimere le
feature da 384 a 32 valori. Le posizioni e le feature memorizzate in quel
file non vengono date alla policy. La griglia visiva corrente copre tutte
le 49 celle; il USD fisico non è ridotto.

Reward come nel precedente compito energetico: progresso `+3/m`, energia
meccanica misurata `−0,005/J`, tempo `−0,01` per azione, goal `+25`, caduta o
uscita dal tabellone `−12`. Episodio massimo 80 s. I 2 s iniziali di
assestamento non contano nell'energia del percorso. Se Spot cade già durante
l'assestamento, l'ambiente termina con un errore esplicito.

## Comandi dalla cartella `my-projects`

1. Prova tecnica con 32 Spot e otto azioni nulle. Non allena né salva
checkpoint; verifica robot, celle, camere e osservazioni DINO. La base PCA
esiste già dall'esperimento del ghiaccio:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/train.py --headless --num-envs 32 --probe-steps 8
```

2. Se la prova termina senza errori, lancia il training con lo stesso numero
di Spot:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/train.py --headless --num-envs 32 --timesteps 102400
```

Il USD completo può esaurire la VRAM con 32 copie. In quel caso rifai **entrambi**
i comandi con `--num-envs 4`; il training non parte dopo un probe fallito.
Ogni training crea una
cartella nuova in `runs/` con `config.json`, `episodes.csv`, `timing.csv`,
checkpoint intermedi e `policy_final.zip`. `Ctrl+C` durante `learn` salva un
checkpoint interrotto; attendi la riga `checkpoint=...` prima di chiudere.

## Confronto energetico guidato delle due vie

Questa prova usa un solo Spot e lo stesso controllore locomotorio, senza
allenamento, camera o DINO. La via `direct` attraversa le rocce della cella
`Cell_3_1`. La via `asphalt` segue i waypoint `(2.1, 7.8)` e `(2.1, 10.3)`
all'interno dell'asfalto `Cell_3_0`, poi torna verso il goal. Entrambe usano
lo stesso limite di velocità. Il test registra l'energia totale, la distanza,
il successo e quanti metri il centro del robot percorre sulle due celle
intermedie. Confronta l'energia media degli episodi che **arrivano e seguono
la via assegnata**; il file CSV conserva anche le prove non valide e i
fallimenti.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/check_goal.py --headless --episodes 20 --speed 0.8 --routes direct asphalt
```

Senza `--headless` si vede la simulazione. La cella `Cell_4_0` a sinistra del
goal contiene rocce: la deviazione evita le rocce della riga 3, ma non
garantisce zero contatti con rocce vicino al goal. Un contatto registrato
con un piede sul confine non implica che il centro del robot abbia
attraversato la cella di rocce intermedia.

Per vedere la policy dopo il training, sostituisci `<RUN>` con il nome della
cartella del training. Senza `--headless` apre Isaac Sim e salva un CSV:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/play.py --checkpoint spot_rl_energy_navigation/full_grid_energy_choice/runs/<RUN>/policy_final.zip --episodes 20
```

`rocks_entry`, `asphalt_entry` e `obstacle_entry` indicano se almeno un piede
ha toccato le aree corrispondenti; non sono premi o penalità della reward.
Questo esperimento usa ancora un'unica coppia partenza-goal: il successo non
dimostra da solo che la policy usi le feature visive invece di memorizzare
una traiettoria.

## Nuovo confronto: partenza e goal sui confini

Il confronto `--scenario border` usa il USD completo e un solo Spot, senza
training, camera o DINO. Parte esattamente da `(3, 6)` m, sul confine tra
asfalto `Cell_2_0` e rampa `Cell_2_1`; il goal è `(3, 10.5)` m, all'incrocio
di quattro celle. Lo yaw iniziale è 90° e non c'è variazione casuale della
partenza. Il raggio di successo è 0,55 m.

La via sinistra attraversa l'asfalto di `Cell_3_0` con waypoint
`(2.2, 7.8)` e `(2.2, 10.0)`. La via destra passa dalla rampa e attraversa
le rocce di `Cell_3_1` con waypoint speculari `(3.8, 7.8)` e `(3.8, 10.0)`.
Le traiettorie nominali hanno entrambe circa 5,11 m. Il CSV registra le
distanze realmente percorse, i joule e se il centro del robot ha seguito
il terreno previsto. Confrontare soltanto gli arrivi validi. La partenza
sul bordo può fare toccare entrambi i materiali ai piedi; il goal è
raggiunto entro il suo raggio, senza dover calpestare l'incrocio esatto.

Dalla cartella `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/check_goal.py --headless --scenario border --episodes 20 --speed 0.8
```

Il file `results/guided_border_check_<data>.csv` viene creato al termine.
Togliere `--headless` per osservare la prova. Questo comando non avvia alcun
training e non modifica i checkpoint esistenti.

### Via centrale sul confine

Per misurare la terza via dagli stessi punti, senza ripetere i due percorsi
precedenti, usa:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/full_grid_energy_choice/check_goal.py --headless --scenario border --routes center --episodes 20 --speed 0.8
```

`center` comanda il goal lungo `x=3`. Oltre a energia e percorso, il CSV
registra la distanza percorsa dal centro del corpo entro ±0,35 m dal confine,
lo scostamento massimo e la frazione di azioni in cui le posizioni XY
dei piedi sono distribuite sui due lati della riga 3. Una prova conta come `route_followed=1` solo
se il centro resta entro 0,4 m dal confine e almeno l'80% del tratto nella
riga 3 è entro ±0,35 m. `mixed_feet_fraction` nella sintesi indica quanto
spesso i piedi si trovavano simultaneamente sui due lati. È una misura
geometrica: non conferma che tutti i piedi fossero in contatto col terreno.
Il solo attraversamento di entrambe le celle in momenti diversi non basta
a definirla una camminata a cavallo. Confronta i joule della via centrale con le precedenti due vie
solo fra gli arrivi validi.

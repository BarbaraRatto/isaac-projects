# Spot RL energy navigation — verifiche e primo training

Questa cartella è separata dal vecchio esperimento `ros2_ws/src/spot_rl_controller`.
Si riutilizzano il robot
`IsaacRobotics/assets/spot.usd`, i pesi della policy di locomozione IsaacRobotics e
`terrain_generator/real_terrains.usd`. Il vecchio controller `SingleArticulation`
non funziona direttamente nel contesto di Isaac Lab; `locomotion.py` riproduce le
sue 48 osservazioni e i comandi di posizione dei giunti tramite `Articulation`.
ROS 2, DINOv2, camera e Nav2 non sono coinvolti in questi controlli.

## Prova semplice: un cammino

Dalla directory `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/check_locomotion.py --headless
```

Togliere `--headless` per aprire la finestra di Isaac Sim. Spot parte da
`(-1.3, 8.25, 0.75)` sulla cella asfaltata, resta a comando zero per 1 s,
cammina per 3 s con comando `vx=0.9 m/s` e si ferma per 1 s. La fisica gira a
200 Hz, come in `IsaacRobotics/applications/spot_warehouse.py`. La policy dei
giunti era stata addestrata con un passo fisico di 500 Hz.

Una prova headless ha completato 1000 passi; lo spostamento orizzontale totale
era 2.79 m e l'altezza finale della base 0.46 m. I joule stampati da questo
script comprendono anche atterraggio e arresto: non usarli come misura del
solo terreno durante la camminata.

## Step 2: confronto tra tre terreni

Togli `--headless` per vedere la simulazione. La modalità grafica carica una
camera di viewport dedicata e può essere sensibilmente più lenta; le prove
procedono automaticamente e la finestra si chiude al termine.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/terrain_speed_study/compare_terrains.py --headless
```

Il programma trova nel USD una cella di asfalto (`t1_asphalt`), ghiaia fine
(`t4_fine_gravel`) e rocce grigie (`t5_large_rocks`). Per ciascuna esegue tre
prove con partenze laterali leggermente diverse ma corrispondenti. Attende 2 s
per l'assestamento, poi comanda `vx=0.9 m/s` fino a circa 2 m di percorso;
interrompe la misura se Spot cade, se la base o un piede esce dalla cella, o se supera 5 s di cammino.
L'atterraggio e l'arresto sono esclusi dal consumo del tratto misurato. La
prova semplice continua ad attendere soltanto 1 s.

Il CSV viene scritto in `results/terrain_comparison.csv` solo quando tutte le prove sono finite; un errore durante la simulazione conserva il CSV precedente. Una nuova esecuzione riuscita lo sostituisce. Contiene stato della
prova, distanza, durata, velocità media, posizione e due stime energetiche separate:
`estimated_energy_j` e `estimated_energy_j_per_m` usano `robot.data.applied_torque`;
`measured_energy_j` e `measured_energy_j_per_m` usano le forze ai giunti
proiettate sulla direzione del moto, lette dal solver PhysX tramite
`robot.root_physx_view.get_dof_projected_joint_forces()`. Per entrambe, la
potenza meccanica è `sum(abs(torque * joint_velocity))` e l'energia è
integrata con il metodo dei trapezi sugli stessi passi di cammino. Le colonne
`estimated_mean_abs_torque_nm` e `measured_mean_abs_torque_nm` mostrano la
media del valore assoluto della coppia sui 12 giunti e sui passi di cammino.
Con gli attuatori impliciti, `applied_torque` è una stima di Isaac Lab,
non una lettura diretta del solver. La misura del solver è vicina a quella
usata dal precedente nodo ROS, ma non misura il consumo elettrico reale.
Le superfici differiscono per attrito e geometria, quindi questo
benchmark non separa gli effetti delle due proprietà.

Nella prova del 30 settembre 2026 tutte le nove camminate sono state completate.
Le medie esplorative con la coppia stimata erano 175.73 J/m (asfalto),
186.43 J/m (ghiaia) e 187.52 J/m (rocce). Con la coppia misurata dal solver,
le medie dello stesso benchmark sono 153.53, 161.47 e 163.32 J/m. Prima della misura la velocità residua era sotto 0.015 m/s
in ogni prova. Ghiaia e rocce hanno valori vicini: questi nove percorsi non
bastano per affermare una differenza affidabile fra loro.

Per una prova rapida con una sola ripetizione per terreno:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/terrain_speed_study/compare_terrains.py --headless --repeats 1
```

### Scansione energetica con i nuovi limiti di velocità

La tabella precedente a 0,5 / 0,7 / 0,9 m/s è conservata in
`results/terrain_speeds_20261006_163528_summary.csv`. Per completarla,
`compare_terrain_speeds.py` prova ora, per impostazione predefinita, **1,1 / 1,3 /
1,6 m/s comandati**. Mantiene gli stessi sette terreni, una cella fissa per
terreno, tre prove laterali, 2 m di cammino misurato e 2 s di assestamento.
Usa la coppia misurata dal solver PhysX e scrive file nuovi con timestamp,
lasciando intatti quelli precedenti. Da `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/terrain_speed_study/compare_terrain_speeds.py --headless
python3 spot_rl_energy_navigation/terrain_speed_study/format_terrain_speed_table.py
```

Il secondo comando legge i CSV esistenti e stampa una tabella Markdown con
le sei velocità; salva anche `results/terrain_speeds_<data>_combined.md`.
La velocità nella tabella è il **comando**, mentre il CSV contiene la velocità
fisica media raggiunta. La cella più economica viene evidenziata solo se
completa tutte e tre le prove. Se Spot cade o lascia la cella a una velocità,
il risultato resta documentato ma non viene trattato come velocità ottima.
Non è un training RL e non usa camera o DINO.

## Step 3: prima policy RL di navigazione

Questa è una prova iniziale con **un solo Spot sulla cella di asfalto già presente
nel USD**. Non usa ancora DINOv2, ROS 2, Nav2, etichette di terreno o una mappa
dei costi. Serve a verificare che l'apprendimento del comando di velocità e la
misura energetica funzionino prima di aggiungere la percezione dei terreni.

- `navigation_config.py`: tempi, limiti dei comandi, posizione relativa del goal
  e pesi iniziali della ricompensa.
- `navigation_env.py`: ambiente Gymnasium su Isaac Lab. Legge il goal nel
  riferimento di Spot, la velocità e l'orientamento della base e il comando
  precedente. L'azione RL ha tre componenti: velocità avanti, laterale e
  angolare. La policy di locomozione in `locomotion.py` resta congelata e
  controlla i giunti.
- `train_navigation.py`: PPO di Stable Baselines3. Salva checkpoint, episodi e
  configurazione in `runs/navigation_YYYYMMDD_HHMMSS/`.
- `play_navigation.py`: riproduce una policy salvata e stampa esito, distanza,
  energia e ricompensa di ogni episodio.

Ogni azione RL dura 0,2 s, pari a 40 passi fisici da 0,005 s. La ricompensa
premia l'avvicinamento e il raggiungimento del goal; sottrae il lavoro meccanico
integrato da `get_dof_projected_joint_forces()`, il tempo trascorso e una
penalità per caduta o uscita dalla cella. L'energia dei 2 s iniziali di
assestamento è esclusa. Il comando avanti è firmato: un'azione zero corrisponde
a velocità richiesta zero, mentre valori positivi e negativi comandano avanti
e indietro. Il goal nel riferimento del robot è diviso per 2 m nell'osservazione.
I coefficienti della ricompensa restano sperimentali.

Da `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/train_navigation.py --headless --timesteps 20000
```

Il training salva `policy_final.zip` nella directory mostrata a terminale.
Il secondo training è stato eseguito per 20 096 transizioni; il checkpoint è in
`runs/navigation_20261001_093538/policy_final.zip`. La valutazione
deterministica con i seed 1000–1009 ha raggiunto il goal in 10 episodi su 10,
con lavoro meccanico medio di circa 490 J per episodio. Il primo checkpoint
riusciva in 1/10; il comando fisso `vx=0.9 m/s` riusciva in 6/10 sugli stessi
seed. Il consumo dei soli successi del comando fisso era anch'esso circa
489 J: questo esperimento verifica il raggiungimento del goal, ma non dimostra
ancora un vantaggio energetico della policy.

Per verificare il checkpoint nuovo nella finestra grafica:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/play_navigation.py --checkpoint spot_rl_energy_navigation/runs/navigation_20261001_093538/policy_final.zip --episodes 10
```

Il training può richiedere tempo perché questa prima versione simula un solo
robot. Il checkpoint iniziale resta disponibile per confronto. Questa prova
non può dimostrare una scelta energetica tra terreni: le feature visive
entreranno nello step successivo.


## Step 4: costo di DINOv2 nel ciclo Isaac Lab

**Archivio storico:** gli script e i risultati dei primi tentativi descritti
in questa sezione e nella successiva sono stati spostati in
`old_useless_attempt/`. I comandi e i percorsi riportati sotto documentano le
prove originali e non sono istruzioni eseguibili dalla posizione attuale. Gli
esperimenti correnti sono descritti nella sezione «Scelta rocce–ghiaccio–rampe».


`benchmark_dino.py` misura il tempo di un ciclo con un solo Spot e comandi
identici di `vx=0.9 m/s`. Ogni comando dura 0,2 s simulati; Spot viene
riposizionato ogni 10 comandi e attende 2 s per assestarsi. Il tempo totale
misurato comprende questi reset, come avverrebbe in un training, mentre la
colonna `simulation_seconds` conta soltanto il tempo dei comandi. Questa è una
**prova di costo del ciclo**, non un addestramento PPO. Lo step 3 e il suo
checkpoint restano invariati.

La prova attuale monta sul corpo di Spot il **vero asset USD ZED X** già
usato in `IsaacRobotics/applications/spot_camera.py` e acquisisce l'RGB della
`CameraRight` a 640×360. Non avvia ROS 2 e usa solo il canale RGB; l'asset ha
anche un'altra camera e un IMU, ma questa misura non usa stereo o profondità.
Il frame di controllo è in `results/dino_sample_rgb.png`. La modalità `online`
riusa `spot_terrain_gridmap/dino_feature_extractor.py` con
`facebook/dinov2-small`, input 518×518 e output 37×37×384. DINO resta
congelato. La modalità `cached` tiene **la ZED attiva** e legge matrici salvate
da `runs/zed_feature_samples.npy`, senza ricalcolare DINO.

Le matrici salvate sono i primi 16 output della modalità `online`, poi ripetuti
in sequenza: **non sono ancora associati alla posa corrente né garantiti
identici ai frame della seconda esecuzione**. La cache serve solo a misurare
il costo della lettura, non è l'osservazione pronta per un training RL.

Da `my-projects`, eseguire in ordine:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/benchmark_dino.py --mode baseline --headless
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/benchmark_dino.py --mode camera --headless
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/benchmark_dino.py --mode online --headless
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/benchmark_dino.py --mode cached --headless
```

Le ultime tre modalità abilitano il rendering della ZED anche in headless.
Ogni esecuzione aggiunge una riga a `results/dino_timing_zed.csv`; la modalità
`online` prepara la cache richiesta da `cached`. Per un aggiornamento DINO ogni
0,4 s simulati, usare `--image-every 2` con `online` e con `cached`.

Prova ZED del 1 ottobre 2026: 40 azioni misurate da 0,2 s ciascuna e 4 azioni
di warm-up. Il tempo di avvio di Isaac Sim non è incluso in `wall_seconds`.

| Modalità | Camera attiva | Tempo per 40 azioni | Azioni/s | Tempo medio dell'operazione sulle feature |
| --- | --- | ---: | ---: | ---: |
| Senza camera | No | 10,416 s | 3,84 | — |
| Solo ZED X | Sì | 6,142 s | 6,51 | — |
| ZED X + DINO online | Sì | 8,372 s | 4,78 | 54,96 ms per inferenza |
| ZED X + feature salvate | Sì | 6,164 s | 6,49 | 0,28 ms per lettura |

La riga senza camera è anormalmente più lenta di quella con ZED: questa singola
esecuzione non permette di quantificare con affidabilità il solo costo della
camera. Il confronto più omogeneo è `online` contro `cached`, entrambe con ZED
attiva e la stessa sequenza di comandi. DINO aumenta il tempo misurato di circa
2,21 s su 40 azioni. Il caricamento del modello richiede altri 2,61 s fuori
dalla finestra misurata. Il 95° percentile dell'inferenza è 69,24 ms e quello
della lettura è 0,33 ms. Le letture della memoria GPU (circa 5,15–5,29 GB)
includono anche altri usi della GPU, non il solo processo.

`results/dino_timing.csv` conserva la **vecchia prova con camera pinhole
generica**: in quella prova `cached` non rendeva la camera, quindi il confronto
era meno omogeneo. Entrambe le prove misurano tempi con comandi fissi: non sono
training PPO e non confrontano ancora qualità delle policy. Per addestrare con
feature precalcolate bisogna costruire la corrispondenza tra feature, vista e
posizione di Spot; per addestrare con DINO online serve passare le stesse
osservazioni spaziali alla policy.

### Confronto durante due veri training PPO

`train_dino_cost.py` esegue il training PPO dello step 3 per 8.192 azioni.
Entrambe le modalità montano e rendono la ZED X destra a 640×360 a ogni
azione di 0,2 s. La modalità `online` estrae DINOv2 dal fotogramma appena
acquisito; `cached` acquisisce comunque il fotogramma e legge una delle 16
matrici già salvate in `runs/zed_feature_samples.npy`. Seed, reward, rete PPO
e osservazioni (11 valori di stato e goal) restano identici. Il confronto
misura soltanto il costo di calcolo durante il training: **le feature non
sono ancora input della policy**, quindi i checkpoint non risolvono la scelta
energetica fra terreni. La cache non è associata alla vista corrente.

Da `my-projects`, eseguire le due modalità con lo stesso numero di azioni:

```bash
TERM=xterm /home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/train_dino_cost.py --mode cached --timesteps 8192 --seed 0 --headless
TERM=xterm /home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/train_dino_cost.py --mode online --timesteps 8192 --seed 0 --headless
```

Ogni esecuzione crea una directory `runs/dino_cost_<modalità>_<data>/`
con `timing.json`, `episodes.monitor.csv`, `config.json` e `policy_final.zip`.
`timing.json` contiene durata del training, transizioni/s, tempo medio e 95°
percentile di acquisizione RGB e operazione sulle feature. Il caricamento di
DINO è riportato separatamente e non è incluso nella durata del training.

Risultati del 2 ottobre 2026, una esecuzione per modalità, 8.192 azioni PPO
ciascuna, ZED X attiva in entrambi i casi:

| Modalità | Training | Azioni/s | Operazione sulle feature, media / p95 |
| --- | ---: | ---: | ---: |
| Feature salvate | 1.050,1 s (17,5 min) | 7,80 | 0,25 / 0,29 ms |
| DINO online | 1.532,5 s (25,5 min) | 5,35 | 54,68 / 67,92 ms |

DINO online ha richiesto 482,4 s in più (+45,9% sul tempo del training
con cache). Il caricamento iniziale del modello, 2,09 s, è escluso dai tempi
di training. Sono state registrate 8.192 operazioni sulle feature per run.
La lettura del buffer RGB dopo il rendering ha richiesto mediamente
0,242 ms nel run con cache e 0,245 ms in quello online; il costo del
rendering è incluso nel tempo totale di training. È una misura singola per modalità: il rapporto
può variare con il carico del computer. Poiché la policy non riceve le
feature, questi checkpoint non confrontano qualità di navigazione né
percezione del terreno.

## Step 5: riferimento misurato per due percorsi

`two_route_scene.py` ricompone **celle già presenti** in `real_terrains.usd` in due
corridoi paralleli, lasciando intatto il file originale. La via diretta contiene
quattro celle `t6_obstacles` e misura 33,00 m tra i waypoint; il bypass usa asfalto
`t1_asphalt` e misura 34,66 m (+5,0%). Partenza e goal sono comuni. I materiali
visivi e fisici restano quelli delle celle originali. `route_config.py` contiene
geometria, waypoint e limiti delle prove. Questa disposizione è un test
controllato della scelta di percorso, non ancora un ambiente di addestramento.

`compare_routes.py` usa lo stesso controllore di locomozione congelato degli
step precedenti. Guida Spot su ciascun percorso, misura l'energia meccanica
`sum(abs(torque * joint_velocity))` con la coppia del solver PhysX e interrompe
la prova se cade, esce dal corridoio o raggiunge il goal. I 2 s di assestamento
prima della partenza sono esclusi. `status=fall` significa che l'energia
riportata riguarda **solo il tratto compiuto** e non è confrontabile con
l'energia totale di un arrivo. Il CSV viene sostituito soltanto quando tutte le
prove programmate finiscono.

Da `my-projects`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/compare_routes.py --headless --repeats 3 --direct-speed 0.65 --bypass-speed 0.9
python3 spot_rl_energy_navigation/analyze_routes.py
```

Togliere `--headless` per seguire le camminate nella finestra di Isaac Sim.
`results/route_baselines.csv` contiene le prove appaiate, con partenza centrale
e due partenze spostate lateralmente di ±0,15 m. `results/route_speed_scan.csv`
conserva una scansione esplorativa di velocità fisse; queste prove usano la
partenza centrale. Gli altri CSV sperimentali nella cartella, se presenti,
riguardano configurazioni preliminari scartate e non vanno mescolati con
il riferimento finale.

Risultato del 1 ottobre 2026 sulle tre partenze: il diretto a 0,65 m/s arriva
in **1/3** casi e consuma 5673 J nell'unico arrivo. Il bypass a 0,9 m/s arriva
in **3/3** casi, con media 5446 J. Nell'unica partenza in cui arrivano entrambi,
il bypass consuma 254 J in meno. Nella scansione a partenza centrale il diretto
arriva anche a 0,6 m/s (6299 J) e 0,8 m/s (5867 J), ma cade a 0,7, 0,75 e
0,9 m/s; fra le velocità provate non batte il bypass. Questi dati definiscono
un **riferimento empirico per le traiettorie guidate e le velocità provate**.
La futura policy potrà usare traiettorie e velocità variabili, quindi questi
numeri non sono una dimostrazione dell'ottimo globale.

La risposta del benchmark è usata solo **per valutare** la futura policy.
Nessuna etichetta di percorso, classe di terreno o costo energetico viene
fornita alla policy attraverso questi file. Il confronto fra training con
DINOv2 online e feature precalcolate richiede ancora un'osservazione visiva
coerente con la posa corrente e due addestramenti nello stesso ambiente; il
benchmark dei tempi dello step 4 non costituisce quel confronto.

## Scelta rocce–ghiaccio–rampe nel USD originale: due training separati

Questo esperimento usa `terrain_generator/real_terrains.usd` **senza spostare le
celle**: partenza nelle rocce `Cell_1_3_t5_large_rocks` a (18, 3) m, goal
nell’asfalto `Cell_3_4_t1_asphalt` a (24, 9) m, orientamento iniziale 45°.
La linea diretta attraversa `Cell_2_3_t2_slippery`; la deviazione a destra
passa per due celle di rampa asfaltata. Spot sceglie liberamente `vx`, `vy`
e velocità angolare, con `|vx| <= 0.9 m/s`, e può anche provare il ghiaccio.
Il limite è 80 s simulati per episodio. Il controllore dei giunti IsaacRobotics
resta congelato. I 2 s iniziali di assestamento non entrano nella reward.

Tutti i comandi qui sotto sono **da eseguire dall'utente**, dalla directory
`my-projects`. Prima si prepara una volta l'atlante DINO della nuova scena:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/build_ice_choice_atlas.py --headless
```

L'output è `spot_rl_energy_navigation/runs/ice_choice_atlas.npz` e la procedura
stampa numero di celle osservate, tempo di preparazione e frazione di varianza
conservata nelle 32 componenti PCA. Se l'USD cambia, l'atlante va rigenerato
con `--overwrite`. Entrambi i training usano la **stessa** base PCA; la modalità
online calcola però le feature dall'immagine corrente a ogni azione.

I due training indipendenti, con stessi seed, timesteps, PPO, scena e camera
RGB-D attiva, sono:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_choice.py --headless --mode cached --timesteps 102400 --seed 0
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_choice.py --headless --mode online --timesteps 102400 --seed 0
```

Lanciali **uno alla volta**. Ciascuno crea una nuova cartella
`runs/ice_choice_<mode>_<data>/` con `config.json`, `episodes.monitor.csv`,
`timing.csv`, checkpoint intermedi, `policy_final.zip` e `summary.json`.
`timing.csv` misura decisioni/s e il tempo della sola inferenza DINO;
`summary.json` registra il tempo di training e il caricamento del modello.
La generazione dell'atlante è un costo separato da considerare per la modalità
precalcolata. La camera viene comunque renderizzata ogni azione anche in tale
modalità; la profondità limita le feature alle celle attualmente visibili.

Dopo i training, `play_ice_choice.py` può caricare un `policy_final.zip` per
visualizzare episodi **senza aggiornare la policy**; usare `--mode` uguale a
quello del training, `--checkpoint <percorso/policy_final.zip>` e omettere
`--headless` per aprire Isaac Sim. L'esito di ogni episodio viene salvato in
`results/ice_choice_<mode>_eval.csv`. I due training sono nuovi: i checkpoint
`first_choice_*` precedenti non sono confrontabili con questo esperimento.

L'atlante spaziale è un'approssimazione delle feature che DINO vedrebbe da ogni
posa: le immagini e le feature nei due modi non saranno identiche pixel per
pixel. Gli esiti sul ghiaccio misurano soprattutto apprendimento della sicurezza
(del terreno che causa cadute), non ancora ottimizzazione energetica fra due
percorsi entrambi percorribili.

## Quattro Spot in parallelo: DINO live

`train_ice_parallel.py` usa quattro copie isolate di `real_terrains.usd` nella
stessa simulazione. In ogni copia, Spot parte dalle rocce verso lo stesso goal,
con le stesse azioni (velocità ogni 0,2 s), lo stesso limite di 80 s e gli stessi
pesi della reward del training online precedente. Il controllore dei giunti
IsaacRobotics è congelato. Ogni Spot ha una camera con ottica ZED X CameraRight,
RGB e profondità 640×360, offset `(0.42, 0, 0.07)` m e pitch 15°. Un **unico**
DINOv2-small congelato elabora le quattro immagini in un batch. L'atlante NPZ
fornisce soltanto la base PCA; le feature consegnate alla policy provengono
sempre dalle immagini correnti.

I 2 s di assestamento vengono simulati una volta per i quattro robot; agli
azzeramenti successivi si riusa la posa e lo stato dei giunti assestati. Questo
evita che il reset di un robot fermi gli altri. È una piccola differenza dal
training precedente a robot singolo, da ricordare nel confronto dei tempi.
La policy PPO è una sola. Quattro robot raccolgono **quattro transizioni RL**
per azione simultanea. Il rollout resta di 256 transizioni per aggiornamento
PPO: 64 per robot, invece di 256 con un solo robot. Tutti i limiti, la reward,
l'energia meccanica da coppie misurate e l'osservazione locale restano uguali.

Dalla directory `my-projects`, eseguire prima la verifica **senza training**:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --probe-steps 8
```

La verifica avvia Isaac Sim, carica DINO e controlla quattro scene, quattro
camere, forma dell'osservazione `(4, 671)`, valori finiti e otto azioni
simultanee. `[PROBE] completed` conferma che è finita. Non produce checkpoint
né aggiorna la policy. Se compare un errore, non avviare il training: copiare
il traceback completo.

Dopo una verifica riuscita, il training è:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --timesteps 102400 --seed 0
```

`--timesteps` conta **transizioni totali dei quattro robot**, non passi per
robot. Si può interrompere con **un solo Ctrl+C**: lo script salva
`policy_interrupted_<passi>_steps.zip`. Durante la corsa salva anche un
checkpoint ogni 5120 transizioni totali. Ogni 1024 transizioni stampa il rate
misurato in transizioni/s e una stima del tempo restante; in `timing.csv` trovi
anche i tempi DINO e visione per robot. Il rate precedente con un solo Spot
online era circa 4 transizioni/s: il parallelismo è vantaggioso solo se il rate
**totale** qui cresce. Il guadagno non è garantito, perché camere e DINO
richiedono GPU. Ogni lancio crea una nuova cartella
`runs/ice_choice_online_4env_<data>/` con `config.json`, `timing.csv`,
`episodes.csv`, checkpoint e `summary.json`. La prova di otto azioni non crea
una cartella run.

Per riprendere un checkpoint dei **quattro robot** fino a un obiettivo totale
più alto, indicare `--resume <checkpoint.zip> --timesteps <totale>`: il totale
deve superare i passi già salvati. Non usare qui un checkpoint del training a
un solo robot.

La policy parallela usa la stessa osservazione di 671 valori e le stesse tre
azioni del task a un robot. Per vedere **senza ulteriore training** tre episodi
del checkpoint appena ottenuto, riusa il player esistente e sostituisci il
percorso al checkpoint con quello stampato dal training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/play_ice_choice.py --mode online --checkpoint spot_rl_energy_navigation/runs/ice_choice_online_4env_<data>/policy_final.zip --episodes 3 --output spot_rl_energy_navigation/results/ice_choice_online_4env_3episodes_eval.csv
```

Se hai interrotto il training, usa `policy_interrupted_<passi>_steps.zip` al
posto di `policy_final.zip`. Questo player apre un solo Spot per osservare bene
la traiettoria. `--episodes 3` può richiedere fino a 240 s **simulati** oltre
all'avvio e a DINO; il tempo reale dipende dal carico GPU. Per un confronto
quantitativo con i vecchi test a 20 episodi, ripeti con `--episodes 20`,
`--headless` e un nome CSV distinto. I checkpoint a 4 e a 1 robot, se
addestrati per numeri di transizioni diversi, confrontano solo in modo
preliminare la qualità delle policy; per il confronto dei tempi usa
`timing.csv` e le transizioni totali.

### Prova con 16 Spot

Lo script parallelo accetta `--num-envs 4`, `8` o `16`; omettendo l'opzione
resta a 4, come nel training completato. Il numero di transizioni per
aggiornamento PPO rimane **256 in totale**: con 16 Spot sono 16 per robot.
Ogni Spot ha una copia isolata del terreno e una camera ZED X RGB-D 640×360.
Il modello DINO è sempre **uno solo**, congelato; con
`--dino-batch-size 4` elabora 16 immagini in quattro passaggi da quattro per
limitare il picco di memoria GPU. Questa scelta preserva la risoluzione della
camera e l'osservazione, ma può limitare il guadagno di velocità.

Prima eseguire, da `my-projects`, soltanto la prova **senza training**:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --num-envs 16 --probe-steps 8
```

L'output deve terminare con `[PROBE] completed=8` e riportare
`observation=(16, 671)` e 16 conteggi di celle visive valide maggiori di
zero. La prova non produce una policy. Su una GPU da 8 GB, se la creazione
delle 16 camere esaurisce la memoria, interrompere qui e provare con
`--num-envs 8`; ridurre automaticamente la risoluzione renderebbe diverso il
compito visivo.

Se la prova riesce, lanciare il training **da zero**:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --num-envs 16 --timesteps 102400 --seed 0
```

I nuovi risultati finiscono in `runs/ice_choice_online_16env_<data>/`; le run
a 4 Spot restano intatte. Controllare il primo `[PARALLEL] transitions=1024`:
riporta le **transizioni complessive al secondo**, che vanno confrontate con
circa **11,5/s** della run a 4 Spot. Più robot non garantiscono più velocità:
rendering, proiezione delle feature e quattro passaggi DINO per azione
possono diventare il limite. `Ctrl+C` salva la policy corrente; i checkpoint
periodici sono ogni 5120 transizioni complessive.


### Prova con 32 Spot

Il parametro `--num-envs` accetta anche `32`. La policy raccoglie ancora
256 transizioni per aggiornamento PPO, ora 8 per Spot. Un solo modello DINO
con `--dino-batch-size 4` esegue otto passaggi per elaborare le 32 camere.
Sulla GPU da 8 GB la memoria del renderer può essere insufficiente anche se
DINO usa microbatch: eseguire **prima** la prova senza training.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --num-envs 32 --probe-steps 8
```

Solo se termina con `[PROBE] completed` e osservazione `(32, 671)`, si può
lanciare una nuova run da zero:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ice_choice/train_ice_parallel.py --headless --num-envs 32 --timesteps 102400 --seed 0
```

La run a 16 Spot resta indipendente. Se viene fermata con Ctrl+C, crea un
checkpoint `policy_interrupted_<passi>_steps.zip`, `summary.json` e conserva
`timing.csv` e `episodes.csv`. I checkpoint a 16 Spot non possono essere
passati a `--resume` con 32 ambienti: cambiano il numero di ambienti e i
passi per ambiente del rollout PPO. Per decidere se 32 conviene, confrontare
le **transizioni totali al secondo** con circa 17/s ottenute con 16 Spot.

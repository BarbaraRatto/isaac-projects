# Test10 SAC Re — laboratorio grigio/verde

Esperimento indipendente basato sul test7 SAC. Budget iniziale: **100.000
transizioni di navigazione complessive**, da zero, con 32 scene separate.
Il training non è stato avviato durante la preparazione dei file.

## Avvio

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/train.sh
```

Il comando controlla la mappa congelata, raccoglie viste RGB-D dei due
materiali, calibra e valida una nuova PCA384→32, quindi avvia SAC in un
processo separato. La PCA viene riutilizzata ai lanci successivi soltanto
se compatibile. Non utilizza la vecchia PCA a sette terreni.

Il comando usa Isaac Lab `/home/isaac/isaaclab/2.3.0/isaaclab.sh`, simulazione,
locomozione, DINO e PCA su `cuda:0`. Come nella versione iniziale basata sul
**test7**, agente SAC, replay, preprocessing RGB OpenCV, gridmap e geometria
sono sulla CPU. Sono state annullate le accelerazioni trasferite dal test9,
inclusi reset in blocco, resistenza vettorizzata e metriche differite.
La RTX 5060 locale ha 8 GB di VRAM; il replay da circa 2,44 GiB resta in RAM.

Restano invariati 32 robot, 100.000 transizioni, rete, reward, mappa,
missioni e PCA già calibrata. Il gestore esplicito di Ctrl+C e la diagnostica
degli errori sono mantenuti. Per scegliere un'altra installazione impostare
`GENERAL_MAP_ISAACLAB`. Le opzioni di `train.py` possono essere aggiunte a
`train.sh`; `--timesteps` indica il target totale.
Esattamente 100.000 campioni vengono accettati, senza richiedere multipli di 32.
Allineamento e calibrazione non vengono conteggiati nel budget SAC.

Monitoraggio in un altro terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/status.py
```

## Mappa e terreni

- Mappa **15×15 m**, 25 celle di **3×3 m**: 17 grigie e 8 verdi.
- Layout casuale con **seed 42**, congelato e uguale nelle 32 copie.
- Alcune celle verdi sono isolate, altre adiacenti; niente ghiaccio o scale.
- `maps/lab_15x15.usda` è autonomo, senza texture esterne o riferimenti alle
  mappe precedenti. `maps/layout.png` e `layout.pdf` mostrano la disposizione.
- `t1_asphalt` è il nome interno del grigio; `lab_green` è il verde.

| Proprietà | Grigio | Verde |
|---|---:|---:|
| Attrito statico / dinamico | 1,0 / 0,8 | 1,0 / 0,8 |
| Contatto | Rigido | Rigido |
| Quota del terreno | 0 m | 0 m |
| Resistenza orizzontale | Nessuna | 90 N·s/m |

Come nella prova manuale, `surface_drag.py` applica **F_xy = −90 v_xy**
alla base, al centro di massa, senza forza verticale o coppia aggiunta.
La velocità è quella reale nel mondo. La scelta della cella usa la posizione
XY della base, sottraendo l'origine della singola scena. Non usa i singoli
contatti dei piedi; il coefficiente cambia al passaggio della base sul bordo.
Il carico viene aggiornato a ogni passo fisico di 0,005 s, anche durante
l'allineamento, e azzerato sul grigio o fuori dalla mappa. È presente sia
nel training sia nella valutazione. Non è incorporato nel solo USD.

Il terreno è un modello sintetico di resistenza all'avanzamento, non un
modello fisico calibrato di un pavimento reale. Non sono sommati Joule
artificiali: l'energia resta l'integrale di Σ|τ q̇| dei giunti.

## Missioni e curriculum

Start e goal vengono ricampionati per ogni robot alla fine del suo episodio.
Non cambiano a ogni transizione e non sono uguali tra tutti i robot.
Gli endpoint sono interni alle celle con margine di 0,65 m. Training e
valutazione usano insiemi distinti di candidati sulla stessa mappa.

| Fase | Distanza A–B | Start e goal |
|---|---|---|
| 0 | 1,5–3 m | Entrambi sul grigio, anche in celle differenti |
| 1 | 3–6 m | Grigio e verde |
| 2 | 6–10 m | Grigio e verde |
| 3 | 10–14 m | Grigio e verde |

Il percorso può attraversare il verde anche quando gli endpoint sono grigi.
Le missioni non impongono un lato o una strada. Nelle fasi successive il
25% delle missioni di training riprende una fase precedente.

Valutazione deterministica ogni 10.000 transizioni e alla fine: 32 episodi,
uno per ambiente. Promozione dopo **due valutazioni consecutive ≥80%**
di successi della fase corrente, come test7. Il budget breve può concludersi
prima delle fasi più difficili. Nessuna promozione automatica a tempo.

Yaw iniziale casuale e allineamento fisico al goal prima della navigazione,
come test7. Timeout per missione: clip(15+4×distanza, 30, 90) secondi.
Goal raggiunto entro 0,55 m; caduta per inclinazione ≥90° o contatto del
torso con il terreno, persistenti per 1 s. Altezza della base diagnostica.

## Rete, input e reward invariati dal test7

Input actor: 15 stati e 81 celle robocentriche ×40 valori = **3255**.
Ogni cella ha PCA32, quattro geometrie, due validità e due età. La memoria
visiva mantiene le osservazioni precedenti nello stesso episodio e si
azzera al reset. Goal nel frame yaw del robot, diviso per 10 senza scaler
dinamico. Altri scaler dinamici con maschere, congelati in valutazione.

- Stato: 15→64→32.
- Mappa: 3240→1024→512→256→128.
- Fusione: 160→128→64→media/log-std delle due azioni.
- Comandi: vx da 0 a 1,6 m/s e wz da −1,5 a +1,5 rad/s.
- SAC asimmetrico con due Q indipendenti; privilegi come test7 (attrito,
  contatti, forze e slip delle quattro zampe), senza etichetta colore/costo
  nell'actor. Nessun nuovo input privilegiato per la resistenza.
- Replay 100.000, warmup 4096, batch 256, learning rate 3e−4,
  gamma 0,999, tau 0,005, un aggiornamento per vector step di navigazione.

Reward per azione di navigazione (0,2 s), con i pesi originali del test7:

`r = 30(d_pre − d_post) − 0,02 − E_step/100 + 1000 I_goal − 1000 I_failure + r_heading`

La penalità di orientamento è nulla fino a 110° e aumenta linearmente
fino a −2 punti/s a 180°, come in `rewards.py`. Non impedisce deviazioni.
I fallimenti includono caduta, uscita mappa, timeout e timeout allineamento.
La penalità temporale può favorire il grigio anche per il minor tempo di
attraversamento: la scelta non dimostra da sola un criterio solo energetico.

## Test e grafici dopo il training

Per confrontare la policy best sulle stesse missioni del precedente test,
con due episodi per categoria di distanza più quello finale sul verde:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/test_best_9.sh
```

Usa `policy_best.pt` della run completata più recente, senza sostituirla
con la final. Riprende esattamente gli episodi **1, 2, 5, 6, 9, 10, 12, 13,
15** della suite da 15, mantenendo start, goal, yaw iniziale e deadline.
Sono rinumerati da 1 a 9; il campo `source_episode` nei risultati conserva
il numero originale. Nessun nuovo campionamento o aggiornamento dei pesi.

Test dedicato di 15 episodi, con un solo robot visibile e il modello finale
più recente selezionato automaticamente:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/test_15.sh
```

| Episodi | Endpoint | Distanza A–B |
|---|---|---|
| 1–4 | Grigio → grigio | 1,5–3 m |
| 5–8 | Grigio → grigio | 5,5–6,5 m (circa 2 celle) |
| 9–11 | Grigio → grigio | 8,5–9,5 m (circa 3 celle) |
| 12–14 | Grigio → grigio | 11,5–12,5 m (circa 4 celle) |
| 15 | Grigio → verde | 3,90 m |

Le distanze sono euclidee, non il numero di celle attraversate dal percorso.
Gli endpoint sono diversi e appartengono ai candidati di valutazione, con
seed 20000. Il piano è congelato in `artifacts/missions_15.json` e la tabella
con le coordinate è in `artifacts/missions_15.md`. La suite viene copiata
nella cartella dei risultati. Start e goal sul grigio non obbligano il
robot a restare sul grigio lungo tutto il tragitto.
Rete e scaler sono congelati, nessun aggiornamento o nuova PCA. Allineamento,
carico sul verde e timeout rimangono quelli del training. Le missioni sono
indipendenti dalla fase raggiunta dal checkpoint: se il training si è fermato
in fase 0, quelle lunghe verificano la generalizzazione oltre le distanze apprese.

Il test standard con missioni casuali della fase selezionata resta disponibile:

Il modello finale più recente viene selezionato automaticamente:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/test.sh
```

Esegue 9 missioni deterministiche, con seed 20000, nella stessa mappa,
alla fase dell'ultima valutazione di training. Per forzare una fase:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/test.sh --stage 1
```

Salva `episodes.csv`, `trajectories.csv`, `fall_diagnostics.csv` e
`summary.json` in `results/evaluation_...`. Per grafici di start, goal e
percorsi, indicando la cartella restituita dal test:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/analyze_paths.py \
  spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/results/evaluation_DATA_ORA
```

## Salvataggi e verifica

`runs/` contiene checkpoint periodici, best per fase, finale, scaler,
PCA, log e sorgenti. Il finale e l'interruzione con Ctrl+C salvano anche
il replay necessario alla ripresa: conservare **l'intera cartella della run**.
I checkpoint intermedi non hanno il replay corrispondente; non garantiscono
una ripresa completa dopo interruzione brusca, come il test7 di base.

Per interrompere usare Ctrl+C una sola volta e attendere il messaggio di
salvataggio di `policy_interrupted.pt` e il ritorno del prompt. La scrittura
del replay può richiedere tempo. Ora SIGINT e SIGTERM sono gestiti
esplicitamente dopo l'avvio di Kit: compare `[SAC STOP] Richiesta ricevuta`,
si completa il passo corrente, poi si salva. La richiesta viene controllata
anche durante le valutazioni, senza aspettare tutti gli episodi. Altri segnali
di arresto durante il salvataggio non sollevano una seconda interruzione.
Errori Python vengono stampati prima della chiusura di Kit e registrati
in `failures/failure_*.json`. Errori nativi che terminano il processo o un
SIGKILL non consentono al gestore Python di salvare.
Per riprendere con la configurazione ripristinata:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re/train.sh \
  --resume /PERCORSO_DELLA_RUN/policy_interrupted.pt
```

Sono ripristinati policy, due Q, target Q, ottimizzatori, alpha, scaler,
replay, contatori e curriculum. Il target resta 100.000 transizioni totali,
quindi vengono raccolte soltanto quelle mancanti. Si crea una nuova cartella
di run; lo stato fisico degli episodi viene reinizializzato. La continuazione
non promette identità numerica bit per bit rispetto a un processo ininterrotto.

Controlli offline (nessun training o simulazione):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh -m pytest -q \
  spot_rl_energy_navigation/train_general_big_map_test10_SAC_Re
```

La calibrazione con immagini reali della simulazione e la prova fisica
richiedono la GPU e vengono eseguite quando si lancia il comando di avvio.

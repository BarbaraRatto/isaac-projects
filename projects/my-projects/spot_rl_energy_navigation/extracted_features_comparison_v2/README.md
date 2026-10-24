# Feature comparison v2 — camera di Spot, sette terreni

Esperimento indipendente in questa cartella. Non importa moduli dalla v1 o
dagli esperimenti RL. Contiene copie locali di Spot USD, controllore locomotorio,
pesi DINO, board, texture e codice di acquisizione/analisi. Le librerie Isaac,
PyTorch/Transformers e Python restano dipendenze installate. Il materiale
standard `OmniPBR.mdl` è fornito dal renderer Isaac Sim.

**Stato della preparazione:** nessuna acquisizione, simulazione o inferenza
DINO eseguita. Sono stati verificati file, asset USD e calcoli con fixture
sintetiche. La dinamica di Spot, la copertura semantica e le inquadrature
vanno controllate con il pilota prima di interpretare risultati.

## Scena e acquisizione

- Sette classi: **asphalt, ice, ramp, gravel, rocks, obstacles, stairs**.
- Board nuovo con **14 tasselli 6×3 m**, due donatori distinti per classe,
  estratti da `real_terrains.usd`. Le mesh, i materiali e le texture originali
  sono mantenuti, applicando solo traslazioni. La seconda fila permuta le
  classi rispetto alla prima. Manifest e hash sono in `assets/manifest.json`.
- Le due file hanno centri Y=0 e Y=7 m, con corsie di accesso in asfalto.
  I tasselli della stessa fila sono adiacenti: le immagini possono contenere
  più terreni. Le corsie non sono etichettate come campioni di asfalto.
- Un solo Spot fisico usa il controllore articolare congelato, con la stessa
  struttura di osservazioni, frequenza di controllo e attuatori dei test recenti.
  Nessun SAC/PPO, policy di navigazione o training RL viene eseguito.
- La camera segue **posizione e orientamento reali del corpo**: offset
  `(0.42, 0, 0.07)` m e inclinazione relativa di 15°, ottiche ZED X, RGB-D
  640×360. Oscillazioni, inclinazioni e parti del robot visibili non vengono
  eliminate dalle immagini RGB.
- Le pose assestate vengono ottenute resettando Spot nelle corsie e lasciando
  stabilizzare la fisica. Le camminate partono dalle corsie e avanzano verso
  ciascun tassello a 0,3 m/s, con piccole correzioni di yaw e velocità laterale
  zero. Non si presume che tutti i percorsi su scale/ostacoli riescano.
- Una caduta interrompe quell'episodio e viene registrata; la raccolta continua
  sugli altri. `episodes_failed.json` e la verifica della copertura devono
  essere controllati, senza scambiare episodi mancanti per risultati positivi.

| Modalità | Tasselli | Pose assestate | Camminate | Frame massimi previsti |
| --- | ---: | ---: | ---: | ---: |
| `pilot` | 7, uno per classe | 28 | 7 × 6 frame | **70** |
| `full` | 14, due per classe | 168 | 28 × 6 frame | **336** |

Luce iniziale uguale al training (distant light, intensità 3000). La v2 non
corregge automaticamente la resa chiara della ghiaia. Si vuole prima valutare
ciò che vede la camera nelle condizioni correnti. Posizioni, tempi e velocità
sono modificabili in `config.json`; l'illuminazione è configurabile ma la
raccolta predefinita non include variazioni di luce.

## Etichette, proiezione e celle

Il renderer produce una segmentazione semantica indipendente da DINO, con
etichette delle mesh dei terreni. Spot, corsie e sfondo vengono esclusi dai
campioni etichettati. Si salvano RGB, profondità, maschera int8 (`-1` ignoto,
`0..6` nell'ordine delle classi), intrinseci, posa della camera, stato del corpo,
episodio, tipo di acquisizione e tassello target.

La proiezione usa profondità e camera effettive. Una patch è considerata pura
quando nove punti proiettati concordano su classe e tassello, lontano dai bordi.
Il controllo è geometrico/semantico: i token DINO conservano contesto globale.
Non si applica il vecchio limite superiore assoluto di altezza alla pipeline
principale, perché potrebbe eliminare rampe/scale. Si escludono i fianchi
inferiori dei tasselli tramite bordo e quota minima.

Si mantiene una griglia fissa da 0,5 m ancorata al board. Le celle sono medie
dei vettori raw e vengono etichettate solo con purezza almeno 95%; robot,
sfondo e celle miste contano contro tale purezza. Si analizzano quattro casi:

| Nome | Significato |
| --- | --- |
| `patches` | Token DINO384 delle patch pure. |
| `cells` | Celle osservate valide, etichettate, di tutta la porzione di board visibile. |
| `front_cells` | Sottoinsieme intercettato dai 20 campioni frontali, usando la posa reale di Spot e maschera 50°. |
| `front_cells_rl_filter` | Stesso campionamento, ricalcolando le medie dopo il vecchio filtro `-0,1 < z < 0,6` m. |

Le celle frontali duplicate contano una volta. Non si usano celle sconosciute.
Il quarto caso serve a diagnosticare eventuali perdite del filtro geometrico;
non è la riproduzione completa di un particolare ambiente RL. Non vengono
modificati né riaddestrati gli esperimenti di navigazione.

## Separazione training/valutazione del classificatore

- **Pilota:** separazione di gruppi di pose/percorsi; tutti i frame di una
  camminata e tutti i campioni della stessa immagine restano dallo stesso lato.
  I donatori dei terreni sono gli stessi: non è generalizzazione a nuovi asset.
- **Completa:** prima fila di donatori per allenare il classificatore lineare,
  seconda fila per valutarlo. Per evitare contaminazione un'immagine contribuisce
  solo campioni della stessa partizione del tassello target. Campioni visibili
  dell'altra fila vengono esclusi dal dataset etichettato.
- Si verifica esplicitamente che frame ed episodi non compaiano in entrambi
  gli insiemi. Le differenze di contesto globale fra file rimangono un limite.
- Il classificatore usa tutte le 384 dimensioni, normalizzazione L2, nessuna
  PCA e nessun riaddestramento DINO. Riferimento casuale: **14,3%**.
- Matrici 7×7, tutte le 21 coppie, distribuzioni e AUC. PCA solo nei grafici.
  Se mancano classi in una rappresentazione, il report indica copertura
  insufficiente invece di presentare un'accuratezza ingannevole a sette classi.

## File principali

| File | Funzione |
| --- | --- |
| `acquire.py` | Unico programma che avvia Isaac Sim. |
| `scene.py`, `locomotion.py`, `routes.py` | Spot/camera, controllore e piano di acquisizione. |
| `terrain_labels.py`, `projection.py` | Etichette del renderer, proiezione e celle. |
| `check_dataset.py` | Integrità, copertura e anteprime delle etichette. |
| `dino_features.py`, `extract.py` | Estrazione offline da pesi DINO locali. |
| `compare.py`, `report.py` | Confronti e report, CPU. |
| `common.py`, `config.json` | Percorsi, hash e configurazione locale. |
| `setup_analysis.sh`, `analysis_python.sh` | Ambiente CPU isolato nella `.venv` di questa cartella. |
| `test_offline.py` | Controlli sintetici e USD senza simulatore/GPU. |

## Procedura da my-projects

Prima preparare l'ambiente di analisi locale (una volta sola). Installa solo
le dipendenze CPU elencate in `requirements-analysis.txt`; non modifica Python
di sistema, Isaac o la `.venv` della v1:

```bash
bash spot_rl_energy_navigation/extracted_features_comparison_v2/setup_analysis.sh
```

Controllo degli asset, senza simulatore:

```bash
bash spot_rl_energy_navigation/extracted_features_comparison_v2/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/check_dataset.py
```

**Solo quando si può usare Isaac Sim**, acquisizione pilota con finestra:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/extracted_features_comparison_v2/acquire.py --mode pilot --output data/raw/pilot_01
```

Aggiungere `--headless` per rendering senza finestra. Il programma chiude Isaac
al termine senza attendere job Replicator inesistenti. Gli errori Python vengono
stampati prima della chiusura e salvati in `logs/`. Nessun output viene sovrascritto.

Verifica e anteprime delle maschere, prima di estrarre DINO:

```bash
bash spot_rl_energy_navigation/extracted_features_comparison_v2/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/check_dataset.py --dataset data/raw/pilot_01 --previews
```

Aprire RGB e immagini in `data/raw/pilot_01/previews/`: i colori devono seguire
i terreni e non il corpo di Spot. Controllare `episodes_failed.json` e i conteggi.
`complete.json` certifica la fine della registrazione, non la copertura sufficiente.

Estrazione offline, senza Isaac Sim o GPU:

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/extract.py --dataset data/raw/pilot_01 --device cpu
```

Confronto e report:

```bash
bash spot_rl_energy_navigation/extracted_features_comparison_v2/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/compare.py --features data/features/pilot_01
bash spot_rl_energy_navigation/extracted_features_comparison_v2/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/report.py --results results/pilot_01
```

Per analizzare solo pose assestate, aggiungere `--kind stance` a `compare.py`:
l'output diventa `results/pilot_01_stance/`. `--kind walking` è indicato per la
raccolta completa: il pilota ha una sola direzione di camminata e non basta
per una valutazione indipendente dei soli percorsi.

Dopo un pilota valido, raccolta completa (nuovo dataset):

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/extracted_features_comparison_v2/acquire.py --mode full --output data/raw/full_01
```

Ripetere controllo, estrazione e confronto sostituendo `pilot_01` con `full_01`.
La classificazione distingue categorie visive/geometriche; asfalto, rampa e
ostacoli condividono parte del materiale originale. Non è garantito che RGB
da solo separi tutte le categorie né che le feature predicano energia.

## Controlli offline

```bash
bash spot_rl_energy_navigation/extracted_features_comparison_v2/analysis_python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/test_offline.py
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh spot_rl_energy_navigation/extracted_features_comparison_v2/test_offline.py OfflineChecks.test_usd_optics_and_local_board
```

La prima esecuzione verifica semantica, percorsi, aggregazione e separazione
dei dati con fixture sintetiche. La seconda apre solo file USD e una camera
USD in memoria: non avvia Isaac Sim, non carica DINO o la policy locomotoria.

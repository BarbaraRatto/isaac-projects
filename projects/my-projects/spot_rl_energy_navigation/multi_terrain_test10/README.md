# Test 10 — PCA e PPO asimmetrico a due teste

Esperimento autonomo di navigazione energetica di Spot in Isaac Lab 2.3.0.
Il controllore articolare rimane congelato. Non richiede Stable-Baselines3:
il trainer PyTorch implementa PPO con due ritorni e due GAE separati.

## Avvio completo

Dalla directory `my-projects`:

```bash
bash spot_rl_energy_navigation/multi_terrain_test10/train_test10.sh
```

Il comando esegue, in ordine:

1. Se manca `artifacts/pca32.npz`, avvia la raccolta RGB-D con un controllore
   a waypoint, adatta la PCA e chiude quella sessione di Isaac Sim.
2. Avvia un nuovo training PPO di **500.000 transizioni complessive**.

Usa `/home/isaac/isaaclab/2.3.0/isaaclab.sh`, simulazione e DINO su `cuda:0`,
PPO su CPU, 32 ambienti, modalità headless. Per cambiare l'installazione:
`TEST10_ISAACLAB=/percorso/isaaclab.sh bash .../train_test10.sh`.
I percorsi degli script e degli asset sono risolti rispetto ai file, non alla cwd.

La raccolta PCA e gli episodi di valutazione **non contano** nelle 500.000
transizioni di training. Queste sono 122 rollout da 4.096 e un ultimo rollout
da 288 transizioni (9 passi per ciascuno dei 32 ambienti).

## Comandi separati

Preparazione PCA, da eseguire una sola volta per la configurazione:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test10/prepare_pca.py \
  --headless --device cuda:0
```

La raccolta predefinita usa 4 robot, 1.500 passi vettoriali, un campionamento
ogni 5 passi, due percorsi perimetrali e uno diagonale. Sono 6.000 transizioni
di raccolta, senza apprendimento RL. Un reservoir per materiale limita il numero
di campioni a 10.000. La PCA viene adattata con lo stesso numero di campioni per
ciascuno dei quattro terreni presenti (asfalto, ghiaccio, ghiaia e rocce).
La classificazione del tassello serve solo a bilanciare il dataset di calibrazione;
nessuna etichetta entra nell'actor.

Se un materiale ha meno di 250 campioni, la preparazione termina con un messaggio
di copertura insufficiente: aumentare `--collect-steps`, per esempio a 3000.
La varianza spiegata viene stampata e salvata; 32 componenti è una scelta del test,
non una soglia di varianza garantita. Si salvano anche i campioni effettivamente
usati, in `pca32.samples.npz`, e i metadati in `pca32.json`.
Un file PCA esistente non viene sovrascritto: usare `--output` per una nuova raccolta.

Controllo breve in simulazione, **senza aggiornamenti PPO**, dopo il fitting PCA:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test10/train.py \
  --headless --device cuda:0 --probe-steps 10
```

Training, con PCA già disponibile:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test10/train.py \
  --headless --device cuda:0 --agent-device cpu --timesteps 500000
```

Per una PCA in un'altra posizione aggiungere `--pca /percorso/pca32.npz`.
`--eval-every 0` disabilita la valutazione durante il training. Il trainer salva
comunque policy finale e log. `--agent-device cuda:0` sposta anche PPO sulla GPU.

## Mappe e missione

Entrambe le mappe piccole speculari sono usate per il training, 16 ambienti ciascuna:

- `terrain_generator/multi_terrain_binary_test2/train/rocks_gravel__right_rocks.usd`
- `terrain_generator/multi_terrain_binary_test2/test/rocks_gravel__right_gravel.usd`

Il nome della directory `test` del secondo asset non ne cambia il ruolo:
qui è una mappa di training. La mappa grande non viene caricata.
Partenza `(0, 0)`, goal `(4, 4)`, jitter iniziale ±0,05 m e yaw verso il goal.
Ogni missione dura al massimo 40 s. Raggio goal 0,55 m. Timeout, caduta e uscita
dai limiti sono terminali veri; entrambi i ritorni si interrompono senza bootstrap.
Il tempo residuo è osservato dall'actor.

Fisica 200 Hz; locomozione congelata 20 Hz; navigazione 5 Hz.
Le due azioni normalizzate diventano `vx=(a0+1)*0.8`, `vy=0`, `wz=a1*1.5`.
Con `tanh` lo stop è approssimabile arbitrariamente; il comando normalizzato zero
corrisponde a 0,8 m/s, non allo stop.

## Osservazioni e reti

L'actor riceve 775 valori: stato15 seguito da 20 celle da 38 valori.

Stato: goal relativo2, velocità lineare3 e angolare3 nel body frame, gravità
proiettata3, azione precedente2, potenza meccanica filtrata1, tempo residuo1.
Il filtro di potenza ha costante temporale 1 s e usa solo l'intervallo già trascorso.

Cella: feature PCA32, altezza relativa1, pendenza longitudinale1, trasversale1,
rugosità1, validità visiva1 e geometrica1. Griglia senza memoria temporale,
orizzontale e orientata con lo yaw. Centri x=(0.5, 1, 1.5, 2),
y=(-1, -0.5, 0, 0.5, 1), celle da 0,5 m; ordine x esterno, y interno.
I punti sono assegnati direttamente a queste aree, senza filtro angolare fisso.

DINOv2-small congelato → PCA congelata 384→32 (senza whitening) → media per cella.
Pesi locali predefiniti: `extracted_features_comparison_v2/models/dinov2`.
La directory deve contenere `config.json`, `preprocessor_config.json` e
`model.safetensors`; si può selezionare un'altra directory con `--dino-model`.
Gli hash dei pesi, del preprocessing e delle mappe sono verificati rispetto alla PCA
e ai checkpoint. Non viene scaricato automaticamente un altro modello.

La geometria usa depth `distance_to_image_plane` sottocampionata ogni 4 pixel:
retroproiezione con intrinseci e posa ROS della camera; altezza relativa alla base;
fit del piano locale; pendenze firmate; RMS delle distanze ortogonali dal piano.
Sono richiesti almeno 10 punti e varianza spaziale XY sufficiente (autovalore minimo
1e-4 m²). Non viene applicato un filtro sulla quota assoluta del mondo.
Ritorni vicini ai centri dei link del robot sono esclusi con sfere conservative
di raggio 8 cm: è un filtro geometrico approssimato da verificare visivamente nel probe.
La camera viene sincronizzata e renderizzata dopo l'ultimo passo fisico, anche dopo reset.

Il critic riceve 799 valori: osservazione actor775 + informazioni privilegiate24.
Per i piedi nell'ordine FL, FR, HL, HR: attriti statico/dinamico8, contatti4,
forze normali4, velocità tangenziali nel riferimento yaw8. Quattro sensori separati
filtrano i contatti contro i nove collider del terreno. I materiali sono letti
dall'USD composto; il coefficiente combinato piede-terreno segue le regole PhysX.
Se un piede tocca più tasselli si usa la media pesata con il modulo della forza
normale per tassello. Velocità tangenziale al punto medio di contatto e normale
risultante sono una sintesi dei contatti, non una ricostruzione di ogni microcontatto.
Senza contatto, tutti i corrispondenti valori sono zero.

Reti separate, attivazioni ELU:

- Actor: `775 → 512 → 256 → 128 → 64 → 2`, Gaussiane diagonali e due log-std apprese,
  azioni trasformate con tanh e log-probabilità corretta col Jacobiano.
- Critic: `799 → 512 → 256 → 128 → 64 → 2`, teste navigazione e costo energetico.

Normalizzazione incrementale delle sole variabili continue; statistiche condivise
fra celle e aggiornate solo su valori validi. Maschere, gravità, azioni e tempo
residuo non vengono standardizzati. Le statistiche rimangono fisse per l'intero
rollout e il suo aggiornamento PPO; vengono aggiornate dopo usando i dati del rollout.
Sono congelate in valutazione e contenute nel checkpoint e nell'actor esportato.

## Obiettivo e PPO

Lavoro assoluto articolare: integrale di `sum(abs(tau * joint_velocity))`, usando
`get_dof_projected_joint_forces()` e integrazione trapezoidale su tutti i 40 intervalli
fisici. Il periodo iniziale di assestamento non conta. Questa grandezza non è energia
elettrica della batteria e conta positivamente anche lavoro meccanico negativo.

`r_nav = 3*(d_pre-d_post) - 0.02 + 100*success - 100*failure`.
`cost_energy = step_energy_j / 100`. Timeout incluso in failure.
`A_actor = A_nav - A_energy`; normalizzazione solo dopo questa combinazione.
La loss di ciascuna testa è MSE sui propri ritorni, con coefficiente 0,5.

PPO: rollout 128 passi/ambiente, minibatch256, 10 epoche, lr=3e-4,
gamma=0.999, GAE lambda=0.95, clip=0.2, entropia=0.005,
grad norm massimo=0.5, target KL=0.02. L'entropia della distribuzione trasformata
è stimata con un campione riparametrizzato. Il target KL può interrompere in anticipo
le epoche. Si campionano dati nuovi dopo ogni rollout; nessun replay buffer.

## Output, valutazione e ripresa

Per controllare un training già in corso, aprire un **secondo terminale** nella
directory `my-projects` ed eseguire:

```bash
watch -n 10 python3 spot_rl_energy_navigation/multi_terrain_test10/status.py
```

Questo comando legge soltanto i CSV: non avvia Isaac Sim e non modifica il processo
di training. Mostra avanzamento, ETA, successi recenti, cause di fallimento e valutazioni
deterministiche. Per scegliere una run specifica aggiungere `--run-dir /percorso/run`.
Senza `watch -n 10` produce una singola lettura. Ctrl+C chiude soltanto il monitor.
Il log di training viene aggiornato a fine rollout, non a ogni azione: l'ETA è calcolata
all'ultimo rollout registrato e risente delle valutazioni periodiche e del carico macchina.
Il monitor legge i file e non verifica se il processo sia ancora attivo: l'età dell'ultimo
aggiornamento aiuta a riconoscere log che non avanzano.

Ogni training crea `runs/test10_ppo_<data>/`, stampato all'avvio. Contiene:

- `config.json`, `architecture.txt`, `sources/`, `pca32.npz`;
- `episodes.csv`: esito, energia, durata in passi, percorso, distanza finale, ritorni;
- `training.csv`: loss separate, KL, entropia, saturazione, celle valide e tempi;
- `evaluations.csv` e JSON dettagliati;
- checkpoint periodici `policy_<transizioni>.pt`, `policy_best.pt`, `policy_final.pt`;
- `actor.pt`: TorchScript, solo inferenza, input **raw** `[N,775]`, output `[N,2]` normalizzato.

I checkpoint periodici scattano al primo confine di rollout successivo a ogni 50.000
transizioni. La valutazione scatta analogamente ogni 100.000 e alla fine: un episodio
per ciascuno dei 32 ambienti, deterministico, stesso seed. Il modello migliore è scelto
prima per tasso di successo, poi per energia media dei successi. Nessuna osservazione
o ricompensa di valutazione aggiorna PPO o le statistiche. Si riutilizza la scena per
non raddoppiare memoria GPU; dopo la valutazione si iniziano nuove missioni di training.
Gli episodi di training non ancora conclusi al confine di valutazione vengono quindi
interrotti senza entrare nel log degli episodi completati.

Valutazione manuale, cinque episodi per ciascuna mappa, con finestra grafica:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test10/play.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test10/runs/NOME_RUN/policy_final.pt \
  --episodes 5 --device cuda:0
```

Sostituire `NOME_RUN` con la directory stampata dal training. Aggiungere `--headless`
per non aprire la GUI. Le due mappe sono simulate insieme; il viewport predefinito
inquadra il primo ambiente. CSV di episodi e traiettorie e JSON sono salvati in `results/`.

Ctrl+C salva `policy_interrupted.pt` quando il ciclo di training intercetta l'interruzione.
Un rollout incompleto viene scartato; le transizioni già raccolte restano conteggiate.
Ripresa fino al target complessivo di 500.000:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test10/train.py \
  --resume spot_rl_energy_navigation/multi_terrain_test10/runs/NOME_RUN/policy_interrupted.pt \
  --timesteps 500000 --headless --device cuda:0
```

La ripresa carica reti, ottimizzatore, statistiche, PCA, contatori e RNG e crea una
nuova directory. Riparte da nuovi episodi: non ripristina lo stato fisico dei robot.
La PCA esterna non serve per riprendere, perché è inclusa nel checkpoint.

## Verifiche senza simulatore

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  spot_rl_energy_navigation/multi_terrain_test10/test_offline.py
```

Le verifiche numeriche non importano Isaac Sim, non caricano DINO e non eseguono
training robotico. Controllano griglia, PCA, maschere, GAE, distribuzione delle azioni,
isolamento delle osservazioni privilegiate, checkpoint, export e budget esatto.
Il probe in simulazione resta necessario per verificare rendering e contatti sul runtime.

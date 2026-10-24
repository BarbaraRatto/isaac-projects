# Train general big map — test 1

Training da zero su una copia immutata di `terrain_generator/real_terrains_v2.usd`.
Il generatore NON viene eseguito. 32 scene separate nello stesso processo Isaac Sim,
con la medesima mappa 42 × 21 m (49 celle da 6 × 3 m), distanziate di 80 m.
Robot visibili, collisioni tra ambienti filtrate, camere limitate a 20 m per non
inquadrare gli altri ambienti. Nessuna ricerca o verifica di raggiungibilità.

## Avvio

Dalla directory `my-projects`:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test1/train.sh
```

600.000 di transizioni totali: 146 rollout da 4096 e uno da 1984 (62 passi
vettoriali). Simulazione e DINO su CUDA:0, PPO su CPU, 4 thread PyTorch.
Nessun checkpoint di test12/test13 viene caricato. La PCA32 rimane quella già
calibrata, senza nuovo fitting. I pesi DINO e il controllore locomotorio sono congelati.
La compatibilità dei pesi/preprocessing PCA viene verificata; non si assume che
la PCA sia ottimale per tutti i materiali della mappa grande.

Monitor da un secondo terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test1/status.py
```

La stima del tempo arriva dopo il primo rollout. Inizializzazione delle 32 mappe,
rendering, osservazioni e valutazioni hanno costi da verificare sulla macchina.
Non è garantito lo stesso throughput delle mappe piccole.

Controllo tecnico facoltativo con Isaac Sim, da lanciare esplicitamente:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test1/train.sh --probe-steps 10
```

Non allena PPO; crea le 32 scene e verifica la pipeline completa con azioni fisse.
Non è stato eseguito durante la preparazione del folder.

## Missioni e reset

- A: tutti i terreni tranne ghiaccio, comprese scale.
- B: tutti i terreni tranne ghiaccio e scale.
- A e B sono diversi per ogni robot e rimangono fissi per l'episodio.
- Al termine cambia solo la missione del robot interessato; gli altri continuano.
- Orientamento iniziale uniforme in [-pi, pi].
- Banco di 4200 punti XY casuali (100 per cella ammessa), verificati offline sulla
  geometria USD: 3360 riservati al training, 840 alle valutazioni. Le coppie e lo
  yaw vengono campionati ai reset; le posizioni vengono scelte da questo banco
  finito, senza interpolazioni o jitter che invaliderebbero i controlli locali.
- Materiale di spawn estratto uniformemente fra i sei ammessi, poi punto casuale;
  B casuale fra i punti ammessi nella fascia di distanza. Se una coppia non esiste
  nella fascia, si ripete il campionamento (nessun controllo del percorso).
- Margine dalle pareti della cella 0,65 m. Campionamento geometrico 11×11 su una
  regione quadrata di semilato 0,55 m; rilievo massimo locale 0,16 m. Questi controlli
  riducono spawn problematici, ma non certificano stabilità dinamica o attraversabilità.
- All'avvio si calibra per 2 s una postura in piedi su asfalto. Nei reset si riusa
  tale postura, ruotata e traslata sopra la quota massima locale con 2 cm di margine,
  e si azzerano velocità, azione precedente, memoria e statistiche del solo robot.
  Non si eseguono passi fisici extra durante i reset parziali: non avanzano episodi
  degli altri robot fuori dal buffer PPO. L'eventuale assestamento dopo il reset
  avviene nei normali passi e la sua energia è inclusa nelle metriche.
- Goal raggiunto entro 0,55 m. Confini globali con margine 0,2 m. Caduta valutata
  rispetto alla quota reale sotto alla base e alla gravità proiettata.

La maschera dei materiali e il banco di punti servono esclusivamente al campionatore.
L'actor non riceve la mappa USD, etichette, traiettorie pianificate o attriti veri.

## Curriculum e valutazione

Tre fasce di distanza EUCLIDEA A–B: 3–6 m, 6–12 m, 12–20 m. Nessun calcolo della
lunghezza del percorso praticabile. Si comincia dalla prima; dopo due valutazioni
consecutive con almeno l'80% di successi nella fascia corrente si passa alla
successiva. Il 25% delle nuove missioni viene estratto dalle fasce precedenti,
il 75% dalla fascia corrente (inizialmente 100% dalla prima). Nessuna regressione
automatica. Un episodio già in corso mantiene missione e scadenza alla promozione.
Se il successo non sale, il curriculum può restare alla prima fascia per tutta la run.

Deadline per missione: `clip(15 + 4*distanza_m, 30, 90)` secondi, arrotondata al passo
successivo di 0,2 s. Timeout è un fallimento terminale vero, non una truncation.
Il tempo residuo viene normalizzato rispetto alla deadline dell'episodio.

Ogni circa 50.000 transizioni, al confine del rollout, e alla fine: 96 episodi
deterministici (32 per fascia). Coppie, yaw e ordine sono riproducibili e indipendenti
dai reset degli altri robot; seed valutazione 710031 e punti riservati. Stesso insieme
in tutte le valutazioni, anche prima delle promozioni. Le valutazioni non entrano
nel buffer PPO né nelle statistiche di normalizzazione. Il training riparte con
nuove missioni dopo la valutazione. La valutazione finale si esegue anche con
`--eval-every 0` (che disabilita soltanto le valutazioni intermedie).

Questa è generalizzazione a missioni sulla stessa mappa, non una dimostrazione di
generalizzazione a nuove mappe. La promozione usa solo il successo nella fascia
corrente. Il checkpoint migliore massimizza il successo complessivo sulle 96 missioni,
a parità minimizza l'energia media dei successi: confrontare anche i risultati per
fascia e le stesse missioni riuscite, per evitare distorsioni dovute a fallimenti.

## Rete, reward e memoria

Actor 3255 → 512 → 256 → 128 → 64 → 2, ELU, Gaussian tanh.
Critic indipendente 3279 → 512 → 256 → 128 → 64 → 2, teste navigazione ed energia.
PPO: gamma 0,999, lambda GAE 0,95, clip 0,2, learning rate 3e-4, 10 epoche,
minibatch 256, entropia 0,005, value coefficient 0,5 per testa, target KL 0,02.

Reward per passo:

`3*(d_precedente-d_attuale) - 0.02 + 150*goal - 300*fallimento - E_abs_J/100`

Peso energetico fisso 1, senza curriculum energetico. Energia meccanica assoluta
misurata dalle forze articolari PhysX e dalle velocità, integrata a 200 Hz con
trapezi; non è energia elettrica. Successo e fallimento mutuamente esclusivi.
Velocità comandata vx 0–1,6 m/s, vy=0, yaw ±1,5 rad/s; navigazione 5 Hz,
controllore congelato 20 Hz, fisica 200 Hz.

15 valori di stato: goal XY robot-centrico, velocità lineare/angolare, gravità,
azione precedente, potenza filtrata e tempo residuo. 81 celle robot-centriche,
9×9 passo 0,5 m, centri da −2 a +2 m. Per cella: PCA32, altezza relativa,
pendenza X/Y, rugosità, maschere visiva/geometrica, età delle due misure.
Memoria separata per episodio, bin mondiali di 10 cm, ultima misura per bin,
nessuna cancellazione per sola uscita dal campo visivo. Età normalizzate a 90 s,
nessun riempimento con verità del simulatore. Registrazione con posa simulata esatta.
Critic: stessi dati più attriti statici/dinamici delle quattro zampe, contatti,
forze e slip letti su tutti i 49 gruppi di terreno. Dimensione privilegi sempre 24.
La finestra locale non garantisce pianificazione globale o gestione di lunghi vicoli ciechi.

## Output, ripresa e test grafico

`runs/general_map_ppo_<timestamp>/`: configurazione, sorgenti, PCA, metriche CSV,
valutazioni JSON con risultati per fascia, `curriculum.json`, checkpoint periodici,
`policy_latest.pt` dopo le valutazioni, `policy_best.pt`, `policy_final.pt`, `actor.pt`.
`episodes.csv` include coordinate A/B, materiali, fascia, deadline, energia e risultato.
Il checkpoint conserva anche livello/streak del curriculum e RNG. Ripresa con nuove
scene e nuovi episodi, non continuazione esatta delle traiettorie interrotte:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test1/train.sh \
  --resume spot_rl_energy_navigation/train_general_big_map_test1/runs/<run>/policy_latest.pt
```

Il budget è il totale finale, non transizioni aggiuntive.

Test GUI: un robot, nove missioni sui punti riservati, fasce alternate:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/train_general_big_map_test1/play.py \
  --checkpoint spot_rl_energy_navigation/train_general_big_map_test1/runs/<run>/policy_final.pt \
  --episodes 9 --device cuda:0
```

CSV e JSON in `results/`. `--num-envs` consente valutazioni parallele. Il default
seed del play è 20000, distinto dal set fisso di monitoraggio durante il training.
Nessun marker del goal viene aggiunto alle immagini della policy.

## Riproducibilità e verifiche

Mappa e banco sono congelati, con hash nel manifest e nei checkpoint. Texture,
robot, camera e controllore restano asset condivisi del progetto. Il generatore
copiato come testo documenta il file presente al momento della preparazione, ma
l'asset autorevole è l'USD copiato; non si presume che rigenerarlo produca lo stesso mondo.
`prepare_map.py` è il preparatore offline usato una volta: rifiuta la sovrascrittura
del banco esistente e non è richiamato dal launcher. Non è necessario eseguirlo.

```bash
python3 -m pytest spot_rl_energy_navigation/train_general_big_map_test1/test_*.py -q
```

Controlli offline su budget, campionamento e split, curriculum, reset parziali,
reward/energia, contatti con 49 celle, memoria, PPO, checkpoint/export e asset USD.
La creazione del folder non avvia simulazione né training: la validazione fisica
sui 32 ambienti resta da effettuare con il launcher o il probe indicato sopra.

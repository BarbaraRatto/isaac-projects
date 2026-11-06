# Test10 SAC — curriculum misto e scaler congelati

Training da zero su 128 copie separate della stessa mappa del test9. Budget
predefinito 1.500.000 transizioni **di navigazione**, SAC asimmetrico con twin Q,
replay 100.000, batch 256, 16 aggiornamenti per passo vettoriale con campioni
validi, learning rate 3e-4, gamma 0.999. L'allineamento iniziale non entra nel replay.

Actor: stato 15→64→32; gridmap 3240→1024→512→256→128; fusione 160→128→64.
Due comandi: vx 0–1,6 m/s, wz ±1,5 rad/s; vy=0. I quattro output interni SAC
parametrizzano media e deviazione standard dei due comandi. Q1/Q2 hanno pesi
indipendenti e 24 privilegi di contatto, non disponibili all'actor.

## Avvio unico

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/train.sh
```

Il launcher:

1. verifica mappe/PCA e il banco di missioni già preparato offline;
2. se mancano gli scaler, avvia la calibrazione RGBD reale su 32 robot, senza
   RL e senza rifare PCA; salva le statistiche solo dopo tutti i controlli;
3. verifica gli scaler congelati e avvia il training da zero su 128 robot.

Camera, DINO, PCA, preprocessing equivalente a OpenCV, gridmap, actor/critic
e replay vengono eseguiti su CUDA. Missioni/log/checkpoint e gestione scena
restano dove necessario su CPU/disco. Default DINO batch 128; calibrazione 32.
Nessun training dei test precedenti è caricato senza `--resume` esplicito.

Il launcher usa il runtime locale Isaac Lab 2.3.0, oppure il `cloud/env.sh`
già installato se presente. La cartella è autonoma rispetto al codice degli
altri test, ma usa i normali asset condivisi Spot, camera, texture e pesi DINO
del progetto. Copiare questa cartella sulla GPU prima di lanciarne il comando.
Non occorre rigenerare i terreni.

Esempio cloud con directory separata e nome nuovo:

```bash
tmux new -s test10
cd /root/spot-test9-sac-cloud
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/train.sh \
  --run-dir /root/test10-runs/run_01
```

`/root` è persistente soltanto se lo prevede il provider. Il percorso può
essere sostituito con un volume persistente. Un `--run-dir` esistente viene
rifiutato. Pianificare spazio per replay, due recovery e checkpoint periodici.

Per misurare i componenti aggiungere `--profile-timing`: introduce sincronizzazioni
CUDA e un certo overhead. Il preprocessing verifica uguaglianza esatta con
OpenCV/processor sui primi frame reali, anche durante la calibrazione.

## Scaler immutabili

| Gruppo | Trasformazione |
|---|---|
| Goal XY orizzontale robocentrico | /10 m, senza centratura o clipping |
| Velocità lineare XYZ, frame base | /2 m/s |
| Velocità angolare XYZ, frame base | /2 rad/s |
| Potenza meccanica media mobile | /400 W |
| PCA32 di ogni cella | media/std calibrate **sulle celle finali**, non sulle patch |
| Altezza relativa alla base | /0,7 m |
| Pendenze X/Y | /(π/2) rad |
| Rugosità | /0,05 m |
| Attrito statico/dinamico piedi, solo critic | /1 |
| Forze piedi, solo critic | /300 N |
| Slip XY piedi, solo critic | /1,5 m/s |

Gravità proiettata, azione precedente, validità delle celle e contatti restano
invariati; tempo residuo ed età restano fra 0 e 1 (età/90 s). Nessuna statistica
running è aggiornata da training, replay, valutazione o ripresa. Gli scaler
sono incorporati nei checkpoint/actor esportato e salvati anche nella run.
Gli altri gruppi scalati hanno protezione [-10,10], con percentuali di clipping
nel monitor; i riferimenti fisici non sono limiti massimi.

La calibrazione bilancia 2.000 celle valide per ciascuno dei sette materiali,
più il 20% di celle di bordo/miste. La validazione usa 500 celle/materiale e
pose disgiunte. Sono richieste molte posizioni distinte, anche per ghiaccio
e scale osservati da terreno sicuro. Preprocessing, PCA e identità degli asset
sono vincolati agli scaler. Se copertura o clipping non superano i controlli,
il training non parte. Per più acquisizioni avviare deliberatamente
`calibrate_scalers.py` con `--train-views-per-material` e
`--validation-views-per-material` maggiori, prima che esista lo scaler finale.

La PCA32 originale copre già tutti e sette i materiali e viene riutilizzata
con la sua provenienza. Non si riusano gli scaler del test8 o del test9.

## Missioni e curriculum

Start/goal sono costanti per episodio e rinnovati a ogni reset, indipendenti
fra robot. Ghiaccio e scale sono esclusi da entrambi. La rete riceve soltanto
stato, goal relativo e gridmap ottenuta da RGBD: non riceve categoria, etichette,
percorso pianificato o costi di materiale.

| Fase | Distanza XY | Direct / right / left / energy / free | Valutazione |
|---|---|---|---|
| 0 | dirette 1,5–3; deviazioni 2–4 m | 50 / 25 / 25 / 0 / 0 % | ogni 10k |
| 1 | 3–6 m | 50 / 25 / 25 / 0 / 0 % | ogni 10k |
| 2 | 6–9 m | 24 / 20 / 20 / 16 / 20 % | ogni 20k |
| 3 | 9–12 m | 24 / 20 / 20 / 16 / 20 % | ogni 30k |
| 4 | 12–20 m | 15 / 12,5 / 12,5 / 10 / 50 % | ogni 30k |

La parte guidata conserva le proporzioni concordate; `free` estrae A/B senza
condizionare presenza del pericolo, lato o costo. Rimangono distanza, endpoint
ammessi e connessione del corridoio di riferimento. Le classi `right/left`
indicano il lato della deviazione geometrica di riferimento rispetto ad A→B,
non un'azione prescritta. Le scelte energetiche comprendono linea diretta
sicura e alternativa sicura più lunga su materiali differenti.

Il preparatore offline usa la mappa nota **solo per selezionare missioni**:
griglia 0,25 m, margine 0,55 m da ghiaccio/scale/bordi, niente attraversamenti
diagonali fra zone bloccate. La fase 0 richiede un attraversamento del pericolo
vicino e un passaggio locale: è una condizione geometrica; la copertura RGBD
effettiva rimane quella misurata dalla camera, senza inventare celle valide.
Altri materiali mantengono la geometria reale: un corridoio 2D non è una
garanzia di stabilità del controllore. I percorsi non vengono eseguiti come teacher.

Per scegliere scenari `energy` si usa soltanto un **proxy di materiale**, non
una misura in Joule o un'etichetta di percorso ottimo. Il proxy non compare in
osservazioni/reward. La policy paga esclusivamente l'energia misurata e può
scegliere qualunque percorso: la superiorità energetica va verificata sui risultati.

Timeout: clip(15 + 4 × lunghezza del riferimento sicuro, 30, 120) secondi,
comprensivo dell'allineamento. Il percorso e la sua lunghezza non sono input
della policy; la frazione di tempo residuo rimane l'input già previsto.

Promozione dopo **due valutazioni consecutive**: ≥90% direct e ≥80% in
**ciascuna** altra categoria attiva, incluse energy/free. Una classe mancante
non può promuovere. I 128 episodi deterministici sono bilanciati per categoria
con arrotondamento; i candidati di valutazione sono separati da quelli di training.
La nuova cadenza decorre dalla valutazione/promozione. `--eval-every N`
è un override esplicito; senza opzione viene applicata la tabella.

Al budget finale si aggiunge un test indipendente con 128 A/B liberi, seed
distinto, sull'ultima fase valutata: `evaluation_free_final.json`. Non influisce
su promozione, replay o scaler. Non dimostra generalizzazione a mappe nuove:
questo test usa ancora una sola disposizione dei materiali.

`artifacts/mission_examples.pdf` mostra start/goal e riferimenti di esempio.
Non sono grafici di una policy allenata. `missions.json` riporta conteggi e limiti.
Non modificare il banco congelato durante una run.

## Reward invariata

Per passo di navigazione di 0,2 s:

`30*(d_precedente-d_attuale) - 0.10 - 10*(E_passo/100 J) + orientamento + terminale`

Goal +1000; fallimento −1000. Orientamento zero fino a 110°, poi fino a
−0,4/passo a 180°. Nessun bonus di deviazione, lato o materiale.
1.000 J costano 100 punti. L'allineamento algoritmico non entra nel replay;
i log distinguono energia di allineamento, navigazione e totale.
Caduta: tilt ≥90° per 1 s oppure contatto corpo-terreno >1 N per 1 s.

## Monitor, ripresa, test e grafici

Run locali automatiche:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC/status.py
```

Per la run cloud dell'esempio:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC/status.py \
  --run-dir /root/test10-runs/run_01
```

Stop: Ctrl+C una volta nel terminale del training, attendere `Saved ...policy_interrupted.pt`
e il prompt. Recovery completa iniziale, ogni 30 minuti, alla promozione e allo
stop/fine; due generazioni verificate e conservate, modello + replay abbinati.
Conservare tutta la run, compresa `recovery/`. Le copie modello ogni 10k non
contengono da sole tutto il necessario alla ripresa.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/train.sh \
  --resume /root/test10-runs/run_01 --run-dir /root/test10-runs/run_02
```

Il target è totale, non aggiuntivo. Scaler ripristinati dal checkpoint, nessuna
ricalibrazione; nuovi episodi fisici. Per cambiare robot/gradient steps serve
un esperimento diverso: il controllo di compatibilità rifiuta cambiamenti nascosti.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test.sh \
  --checkpoint /PERCORSO_RUN/policy_final.pt --episodes 12

bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test.sh \
  --checkpoint /PERCORSO_RUN/policy_final.pt --suite free --episodes 12

python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC/analyze_paths.py --help
```

Play salva start, goal, traiettorie, comandi, Joule ed eventi di caduta per
generare i grafici con `analyze_paths.py`. Gli scaler embedded consentono di
testare un checkpoint senza rifare la calibrazione sulla macchina di playback.

### Test con goal distanti, senza altro training

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test_long_goals.sh
```

Usa il checkpoint importato `checkpoint_trained_online/test10_relevant/policy_best_stage_1.pt`,
un robot visibile, 9 episodi e distanza euclidea start–goal fra 12 e 24 m.
Le celle della mappa misurano 6 m lungo X e 3 m lungo Y: quattro celle sono
circa 24 m orizzontali oppure 12 m verticali. Le posizioni vengono campionate
dal banco di valutazione, escludendo ghiaccio e scale come start/goal e
richiedendo un riferimento 2D raggiungibile; non si impone un lato di deviazione.
Il percorso effettivo può essere più lungo della distanza euclidea. Il timeout
resta calcolato dalla lunghezza del riferimento, con il limite originale di 120 s.

Il selettore `--stage 4` del launcher serve alla valutazione: non indica che
il checkpoint abbia completato quella fase. Le distanze personalizzate sono
registrate nei risultati, separatamente dalla fase raggiunta dal modello.
La policy usa gli stessi pesi, PCA e scaler incorporati nel checkpoint; nessun
aggiornamento o nuova calibrazione. I risultati finiscono in `results/evaluation_...`
con tutti i dati per i grafici dei percorsi.

Per goal attorno ai 24 m o per provare il modello finale:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test_long_goals.sh \
  --distance-range 22 26

bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test_long_goals.sh \
  --checkpoint spot_rl_energy_navigation/train_general_big_map_test10_SAC/checkpoint_trained_online/test10_relevant/policy_final.pt
```

`--headless` nasconde la simulazione, `--episodes N` cambia il numero di episodi.
L'anteprima offline delle 9 missioni predefinite è in
`analysis/long_goal_test_12_24m/mission_preview.csv`; non contiene traiettorie simulate.

## Verifica del codice

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh -m pytest -q \
  spot_rl_energy_navigation/train_general_big_map_test10_SAC
```

I test usano dati sintetici e simulator doubles; non avviano Isaac Sim. La
calibrazione RGBD reale e le prestazioni su GPU restano verifiche del primo
lancio sulla macchina di training. I due cambiamenti (curriculum/scaler) sono
contemporanei: questo esperimento non ne separa causalmente gli effetti.

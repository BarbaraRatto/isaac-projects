# Test14 SAC Re — versione locale, 32 Spot e 600k transizioni

Nuovo training **da zero**, derivato dal test13 SAC Re grigio/ghiaccio nero. Non crea archivi e non carica automaticamente policy precedenti. La mappa e i nove episodi di playback sono copie identiche del test13, per confrontare i risultati sulle stesse missioni.

| Impostazione | Valore |
| --- | --- |
| Budget | 600.000 transizioni di navigazione complessive |
| Robot | 32 scene indipendenti, stessa mappa |
| Aggiornamenti SAC | 4 per passo vettoriale contenente campioni di navigazione, dopo warmup |
| Rapporto nominale aggiornamenti/robot | 4/32, come 16/128 nel test13 online |
| Replay / batch / warmup | 100.000 / 256 / 4096 |
| Learning rate / gamma / tau | 0,0003 / 0,999 / 0,005 |
| Simulazione, locomozione, rendering, DINO/PCA | GPU CUDA:0 |
| SAC, replay, gridmap/geometria | CPU |
| RGB | OpenCV su CPU |
| Batch DINO | 4 |
| Valutazioni | 128 episodi complessivi: 4 per ognuno dei 32 robot |
| Frequenza valutazioni | Ogni 10k transizioni nelle fasi 0–1, poi 30k nelle fasi 2–3; anche alla fine |
| Promozione | Almeno 80% in due valutazioni consecutive |
| Recovery completa | Ogni 30 minuti; anche iniziale, promozione, arresto ordinato e finale; 2 generazioni verificate |
| Checkpoint ordinari | Ogni 10k transizioni |

## Avvio su questo PC

Eseguire dalla radice `projects/my-projects`, dove si trova `spot_rl_energy_navigation/`. Il launcher usa l'installazione locale `/home/isaac/isaaclab/2.3.0/isaaclab.sh`; non importa automaticamente l'ambiente cloud.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/train.sh
```

Verifica la mappa, calibra una PCA32 nuova su grigio e nero se non ne esiste una compatibile, poi avvia SAC in un processo separato. La PCA rimane congelata durante l'RL. La calibrazione iniziale non compare nel conteggio delle transizioni. Nessuna PCA o policy del test13 viene caricata automaticamente.

**Spazio disco:** conservando circa 60 checkpoint ordinari, best/final e due recovery con replay, prevedere almeno 20–25 GB liberi prima di partire e controllare lo spazio durante l'esecuzione. Non cancellare recovery di una run che si vuole riprendere. Il sistema ha controlli preventivi prima delle scritture complete; non evita la necessità di liberare spazio. Alla preparazione dei file erano disponibili soltanto circa 6,4 GB sul disco del progetto; nessun file precedente è stato eliminato.

## Monitor

Da un altro terminale, nella stessa directory di progetto:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/status.py
```

Per vedere il report completo, senza il limite verticale di watch:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/status.py
```

Il monitor seleziona la run con `training.csv` aggiornato più recentemente, mostra successi, ETA, backend, fase e recovery. Durante calibrazione/inizializzazione può non esserci ancora un log; durante valutazione o salvataggio i contatori possono temporaneamente restare fermi.

Stima iniziale: **15–20 ore**, non un benchmark della nuova configurazione. I precedenti training locali da 32 robot impiegavano circa 2,1 ore/100k con un solo aggiornamento SAC e 32 episodi per valutazione; qui sono previsti 4 aggiornamenti e 128 episodi. Il tempo reale dipende anche dalle fasi raggiunte. Usare il monitor per aggiornare la stima.

## Task, terreni e reward

Mappa piana 15×15 m, 25 celle da 3×3 m, stessa disposizione del test13. Prima riga in alto (Y maggiore):

```text
n g g n g
g n g g g
g n g n g
g g g n n
n g g g g
```

17 celle grigie, attrito statico 1 e dinamico 0,8; 8 nere, attrito statico 0,10 e dinamico 0,08 come il vecchio ghiaccio. Restituzione zero, nessuna cedevolezza o forza resistente aggiuntiva. Le 17 celle grigie sono connesse per adiacenza sui lati. L'attrito al contatto tiene conto anche del materiale dei piedi.

Start e goal casuali **sempre sul grigio**, margine 0,65 m dai bordi, candidati di valutazione separati da quelli di training. Curriculum: 1,5–3 m, 3–6 m, 6–10 m, 10–14 m. Dal livello 1, 25% di missioni dalle fasi precedenti. Nessun planner impone la deviazione e nessuna penalità è associata direttamente al colore o alla visita del nero. Allineamento algoritmico iniziale, deadline `clip(15+4*distanza,30,90)` s, raggio goal 0,55 m. Caduta con tilt ≥90° o contatto del tronco oltre 1 N persistenti per 1 s; altezza solo diagnostica.

Reward per passo: `30*progresso_m - 0,02 - energia_J/100`, `+1000` al goal, `-1000` al fallimento. Penalità di orientamento nulla fino a 110°, lineare fino a 180°, massimo 2 punti/s. Energia misurata dalle articolazioni, senza aggiunte artificiali.

Actor invariato: stato `15→64→32`, mappa `3240→1024→512→256→128`, fusione `160→128→64`, distribuzione per vx 0–1,6 m/s e wz ±1,5 rad/s. Critic gemelli asimmetrici con 24 input privilegiati. Gridmap robocentrica 9×9 da −2 a +2 m con PCA32, geometria, validità e anzianità, memoria delle misure viste. Goal /10 fisso, altri scaler dinamici nel training e congelati in valutazione.

## Arresto ordinato e ripresa

Ctrl+C una sola volta nel terminale del training, poi attendere salvataggio e chiusura. Per riprendere indicare la CARTELLA della run, non una policy ordinaria senza replay abbinato:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/train.sh \
  --resume spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/runs/NOME_RUN \
  --timesteps 600000
```

Il traguardo include le transizioni già completate. Si crea una nuova run; conservare anche la precedente. Il resume ripristina learner, ottimizzatori, scaler, curriculum, RNG e replay; gli episodi fisici vengono resettati. In caso di arresto brutale si riparte dall'ultima recovery completa verificata. Non usare checkpoint del test13: il test14 ha firma e configurazione differenti.

## A fine training: nove episodi e grafici

Aspettare il salvataggio finale, `summary.json` con `interrupted: false` e 600.000 transizioni; il solo 100% del monitor può precedere le scritture finali.

Test visibile con un robot e policy best dell'ultima run completata:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/test_9.sh
```

Missioni identiche al test13: 2 vicine, 2 a circa 6 m, 2 a 9 m, 2 a 12 m, una con ghiaccio sul segmento A–B. Tutti gli endpoint sul grigio e diversi fra loro. La policy decide il percorso. Per test headless aggiungere `--headless`; per scegliere un modello specifico aggiungere `--checkpoint /percorso/policy_final.pt` o altro checkpoint di questa run.

Grafici dell'ultimo test completato, senza riavviare Isaac Sim:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test14_SAC_Re/analyze_results.sh
```

Scrive in `results/NOME_TEST/analysis/`: panoramica, zoom per episodio, diagnostica PNG/PDF, CSV e report. Si può passare esplicitamente una directory di risultati dopo il nome dello script.

## File da conservare

Tutta `runs/` soprattutto `recovery/`, più `results/` con traiettorie e grafici, PCA/manifest in `artifacts/` e sorgenti del test14. Le run includono PCA, sorgenti, mappa, config, curriculum, CSV, valutazioni e checkpoint. Non conservare solo policy_final se si vuole poter riprendere il training completo.

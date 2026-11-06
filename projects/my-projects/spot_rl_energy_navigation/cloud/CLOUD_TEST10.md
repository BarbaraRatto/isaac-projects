# Test10 SAC — avvio sulla GPU online

**È sufficiente questo archivio.** Include codice del test10, mappa e tutte le
texture, banco missioni, PCA32, DINOv2, robot Spot, controllore di locomozione e
camera. Non caricare gli archivi dei test7/test9 o un secondo archivio dei terreni.
Il setup scarica Isaac Sim e le librerie; serve Internet per l'installazione.
Non contiene policy di navigazione già allenate.

## 1. Caricamento ed estrazione

Da Windows, usando IP e porta attuali della GPU:

```powershell
scp -P PORTA "C:\Users\babir\Downloads\spot-test10-sac-cloud.tar.gz" root@IP:/root/
```

La destinazione è `/root/`, con la barra finale. Sul server:

```bash
cd /root
tar -xzf spot-test10-sac-cloud.tar.gz
cd spot-test10-sac-cloud
```

Estrarre in una cartella nuova. Se hai copiato anche il file `.sha256`, puoi
verificare il trasferimento con `sha256sum -c spot-test10-sac-cloud.tar.gz.sha256`
prima di entrare nella cartella estratta. Il setup verifica comunque i checksum
dei file estratti.

## 2. Preparazione: scegli uno dei due comandi

**GPU nuova, software non installato:**

```bash
bash setup_test10.sh
```

Usa Ubuntu 22.04/24.04 x86_64 con GPU RTX e supporto grafico/Vulkan, driver
esposto dall'host e root/sudo per le librerie. Il setup installa lo stesso stack
del test9: Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11, PyTorch 2.7.0 CUDA 12.8.
Non installa i driver host. Prevedere circa 200 GB complessivi per installazione,
cache e risultati. Leggere e accettare la licenza NVIDIA quando richiesto.

**Stessa macchina, installazione del test9 ancora presente:**

```bash
bash setup_test10.sh --reuse-existing /root/spot-test9-sac-cloud
```

Questo riutilizza soltanto il software, senza caricare pesi, replay, scaler o
curriculum del test9. Non modifica il suo `env.sh` né le sue run. Non eliminare
la vecchia cartella: contiene ancora il runtime utilizzato dal test10.
L'opzione funziona anche con una copia integrale del runtime su un'altra macchina,
purché i percorsi salvati siano ancora validi; altrimenti usare il setup nuovo.

Entrambe le modalità verificano file, CUDA e DINO, poi avviano una prova tecnica
con **due robot e due passi**, senza calibrazione degli scaler e senza training.
Questa prova non certifica la memoria necessaria a 128 robot. Al primo avvio
shader ed estensioni possono richiedere alcuni minuti.

Solo verifica dei file, senza installare né avviare simulatori:

```bash
bash setup_test10.sh --check-only
```

Per saltare solo la prova del simulatore aggiungere `--skip-probe`. Per scegliere
un percorso software durante una nuova installazione:
`bash setup_test10.sh --runtime-dir /PERCORSO/runtime`.

## 3. Training

Terminare il training test9 prima di avviare test10 sulla stessa GPU. La
configurazione è **128 robot, DINO batch 128, 1.500.000 transizioni da zero**.
Simulazione, rendering, DINO/PCA, preprocessing RGB, gridmap, geometria, replay e
aggiornamenti SAC usano la GPU. Campionamento missioni, scrittura dei file e
parte dell'orchestrazione restano sulla CPU.

```bash
tmux new -s test10
bash start_test10.sh /root/test10-runs
```

Non servono altri parametri per attivare i backend GPU. Per scegliere un budget
diverso, aggiungere per esempio `--timesteps 600000`.

**Al primo avvio avviene prima la calibrazione RGBD degli scaler visivi**, con
32 robot e DINO batch 32. Usa celle valide dopo la proiezione e osservazioni di
tutti i materiali, con dati separati per calibrazione e verifica. La PCA
congelata non viene riaddestrata. Solo se la verifica degli scaler passa inizia
il training SAC. Fino a quel momento il monitor può indicare che attende il
primo log: l'avanzamento della calibrazione è nel terminale del training.
Non c'è una stima attendibile della durata della calibrazione prima della prova
sulla macchina noleggiata. Se la copertura o la verifica non bastano lo script si
ferma con un errore, senza iniziare un training con scaler incompleti.

Gli scaler sono poi tutti fissi. Curriculum misto con missioni dirette, deviazioni
a destra/sinistra, alternative energetiche e goal casuali; valutazioni ogni 10k
nelle fasi 0–1, 20k nella fase 2, 30k nelle fasi 3–4. Reward e dettagli completi
nel README del test10. L'energia è misurata durante la simulazione, mentre il
proxy dei materiali serve solo a preparare alcune missioni del banco.

`/root/test10-runs` è un esempio: usare il volume che il provider conserva.
Il nome di una directory non ne garantisce la persistenza. Il launcher crea una
run nuova e aggiorna `latest_run.txt`. Per staccarsi da tmux: **Ctrl+B, poi D**.
Per tornare: `tmux attach -t test10`.

## 4. Monitoraggio

In un secondo terminale, nella cartella estratta:

```bash
watch -n 10 bash monitor_test10.sh /root/test10-runs
```

Mostra transizioni, tempi stimati, valutazioni, curriculum, clipping degli input
e ultimo salvataggio completo. Per diagnosticare le prestazioni si può avviare
con `--profile-timing`; i timer sincronizzati possono rallentare il training.

## 5. Arresto e ripresa

Per un arresto ordinato premere una volta Ctrl+C nel terminale del training e
aspettare che finisca il salvataggio completo. Per la ripresa conservare **tutta
la run, inclusa `recovery/`**, non soltanto `policy_*.pt`.

```bash
bash start_test10.sh /root/test10-runs \
  --resume /root/test10-runs/NOME_RUN_INTERROTTA \
  --timesteps 1500000
```

Il budget indica il totale finale. Si riprende dall'ultima coppia verificata
modello/replay, con ottimizzatori, scaler, PCA, curriculum e RNG. I salvataggi
completi avvengono inizialmente, circa ogni 30 minuti, ai passaggi di curriculum,
all'arresto gestito e a fine training; si conservano due generazioni e si cerca
quella precedente se l'ultima è corrotta. Un arresto brutale può perdere i dati
successivi all'ultimo salvataggio. Gli episodi fisici vengono resettati alla
ripresa: non è una continuazione identica passo per passo.

La ripresa crea una nuova run. Non cambiare numero di robot o gradient-steps
rispetto al checkpoint. Non riprendere da checkpoint del test9: questo
esperimento ha un curriculum e una normalizzazione diversi.

Per continuare su un altro host installare lo stesso pacchetto, copiare
l'intera run e usare il comando precedente. Gli scaler sono anche incorporati
nei checkpoint: una ripresa non esegue una nuova calibrazione. Conservare inoltre
`artifacts/frozen_scalers.json` e `artifacts/frozen_scalers.samples.npz` della
cartella test10 per documentare la calibrazione e riutilizzare gli stessi scaler
nei successivi training da zero. Non eliminarli o rigenerarli durante una run.

## 6. Valutazione finale

```bash
source spot_rl_energy_navigation/cloud/env.sh
bash spot_rl_energy_navigation/train_general_big_map_test10_SAC/test.sh \
  --checkpoint /root/test10-runs/NOME_RUN/policy_final.pt \
  --headless --episodes 9
```

Per missioni casuali senza selezione in base agli ostacoli aggiungere `--suite free`.
Togliere `--headless` sul PC con display per vedere la simulazione. Per grafici e
tabella dei test, passare la cartella evaluation prodotta a:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test10_SAC/analyze_paths.py \
  /PERCORSO/EVALUATION
```

L'archivio è verificato offline per integrità e portabilità. Installazione,
sensori e calibrazione reali vengono verificati sulla GPU con i comandi sopra;
non è stato avviato alcun training per creare il pacchetto.

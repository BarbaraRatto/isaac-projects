# Aggiornamento GPU del test9 — 4 novembre 2026

Il piccolo archivio `spot-test9-gpu-update.tar.gz` aggiorna il pacchetto test9 già
installato. Non contiene terreni, pesi DINO, PCA, dipendenze o run: non bisogna
reinstallare Isaac Lab né scaricare di nuovo il pacchetto completo.

## Che cosa cambia

- RGB: ridimensionamento intero equivalente a OpenCV INTER_LINEAR e lookup table
  generata dal processor originale, eseguiti sulla GPU. Verifica obbligatoria
  di uguaglianza esatta sui primi due frame reali, prima di utilizzare il percorso.
  In caso di differenza il programma si ferma, senza sostituire silenziosamente
  gli input. Il backend OpenCV rimane disponibile.
- Replay: buffer e campionamento su CUDA; stesso schema NPZ dei vecchi snapshot.
  Scrittura dei backup in blocchi, senza duplicare tutto il replay nella RAM.
- Metriche SAC: restano tensor detached fino al log; una sola copia dei valori
  per il log. Restano i controlli di finitezza delle loss e delle metriche.
- Reset: posizionamento vettoriale sulla GPU e un trasferimento raggruppato dei
  dati di missione. Il campionatore NumPy e le sue regole/RNG restano quelli
  originali, per preservare la distribuzione delle missioni e i seed di valutazione.
- Tempi: `--profile-timing` misura fisica/locomozione, rendering/camera, RGB,
  DINO/PCA, geometria/gridmap, raccolta, SAC, valutazione e backup. `timing.json`
  e il monitor riportano i risultati. La misura sincronizza CUDA ai confini delle
  fasi e aggiunge overhead; serve a diagnosticare, non è una promessa di velocità.

Reward, architettura, scaler, PCA, materiali, curriculum, numero di robot e
rapporto aggiornamenti SAC/campioni restano quelli della run originale. Anche
la frequenza di valutazione resta quella originale (10.000 in questa run).
Campionare il replay con Torch CUDA mantiene la distribuzione uniforme ma usa
il generatore CUDA anziché NumPy: non è una continuazione bit per bit della
stessa traiettoria casuale. La ripresa ripristina lo stato di apprendimento
completo, poi avvia nuovi episodi come già previsto dal test9.

## 1. Caricare ed estrarre, senza fermare ancora il training

Dal PC Windows (usare IP e porta attuali della GPU):

```bat
scp -P 15262 "C:\Users\babir\Downloads\spot-test9-gpu-update.tar.gz" root@195.231.123.138:/root/
```

Sulla GPU:

```bash
cd /root
tar -xzf spot-test9-gpu-update.tar.gz
python3 /root/spot-test9-gpu-update/install_update.py /root/spot-test9-sac-cloud --check
```

`--check` è in sola lettura e può essere eseguito mentre il training continua.
Non copiare i file del payload manualmente dentro la cartella in uso.

## 2. Stop con salvataggio completo

```bash
tmux attach -t test9
```

Premere **Ctrl+C una sola volta** dentro il terminale che esegue il training.
Attendere il messaggio `[SAC] Saved .../policy_interrupted.pt` e il ritorno del
prompt della shell. Il segnale viene gestito al confine di un passo/update;
un backup di più GB può richiedere tempo. Non chiudere l'istanza e non usare
`kill -9`. Se il training termina normalmente nel frattempo, il file sarà
`policy_final.pt` e non servono altre transizioni se ha raggiunto il target.

## 3. Installazione e ripresa, dentro tmux

```bash
python3 /root/spot-test9-gpu-update/install_update.py /root/spot-test9-sac-cloud
cd /root/spot-test9-sac-cloud
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/resume_gpu.py /root/test9-runs
```

L'installer rifiuta un training test9 ancora attivo e salva una copia del codice
originale in `spot_rl_energy_navigation/cloud/update-backups/`. Verifica il
pacchetto base, gli asset e il payload; aggiorna i checksum del bundle.

Il launcher `resume_gpu.py`:

1. legge `latest_run.txt` e la configurazione originale;
2. con il Python del cloud, senza SimulationApp né aggiornamenti dei pesi,
   verifica uguaglianza RGB/normalizzazione/DINO, roundtrip NPZ e caricamento
   di learner, ottimizzatori, replay e curriculum del salvataggio reale;
3. confronta l'inferenza deterministica CPU/CUDA su osservazioni del replay;
4. solo dopo **PASS**, avvia una **nuova** run con `--resume` della precedente,
   RGB/replay/agent/mappe su CUDA e profilazione dei tempi;
5. conserva seed, batch DINO, robot, gradient steps, frequenze e target totale
   della configurazione originale. Per questa run restano 1.500.000 totali,
   128 robot, 16 aggiornamenti e DINO batch 128.

Se il controllo GPU fallisce, **non parte alcun nuovo training**: conservare
l'output per diagnosticare il problema. Il vecchio salvataggio resta intatto.
Il controllo scrive `/root/test9-runs/acceleration_validation.json`.

Per vedere prima i comandi senza eseguirli:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/resume_gpu.py /root/test9-runs --dry-run
```

Per scegliere una run diversa dal puntatore `latest_run.txt`, aggiungere
`--resume /root/test9-runs/NOME_RUN` al launcher.

## 4. Monitorare

In un altro terminale SSH:

```bash
cd /root/spot-test9-sac-cloud
watch -n 10 bash monitor_test9.sh /root/test9-runs
```

All'inizio controllare `[RECOVERY] Verified checkpoint selected`, il contatore
iniziale, la fase ripristinata e `[RGB] Real-frame equivalence PASSED`.
Il tempo trascorso e le statistiche di training della nuova run ripartono;
contatore transizioni, aggiornamenti SAC e curriculum vengono ripristinati.

I tempi nel monitor sono medi per chiamata. `sac_updates` include il gruppo di
16 aggiornamenti; `physics_locomotion` include i 40 intervalli fisici.
Le fasi visive sono annidate in raccolta/valutazione: non sommare tutte le righe.
Una valutazione ogni 10.000 transizioni continua a ridurre la velocità media
visualizzata, perché i suoi episodi non entrano nel contatore di training.

Per scollegarsi lasciando il training attivo: Ctrl+B, rilasciare, poi D.
La persistenza di `/root/test9-runs` resta una proprietà del servizio noleggiato,
non di tmux. Conservare l'intera run, compresa `recovery/`.

## Rollback del codice

Prima fermare anche il training aggiornato con Ctrl+C e attendere il salvataggio.
Usare il percorso di backup stampato dall'installer:

```bash
python3 /root/spot-test9-gpu-update/install_update.py /root/spot-test9-sac-cloud \
  --rollback /root/spot-test9-sac-cloud/spot_rl_energy_navigation/cloud/update-backups/NOME_BACKUP
```

Il vecchio lettore CPU può leggere anche gli snapshot NPZ scritti dal nuovo replay.
Il rollback non elimina le run. Per rilanciare il codice originale utilizzare
il comando precedente con `--resume` della run scelta e senza le nuove opzioni.

## Verifiche locali e limiti

L'ambiente di verifica usa Torch 2.7.0+cu128, OpenCV 4.11.0 e Transformers 5.8.1,
come il cloud. I controlli locali non avviano la simulazione. Su questo host
CUDA non è disponibile: test GPU e prestazioni end-to-end vanno verificati sulla
5090. Il controllo obbligatorio del launcher e quello sui frame reali servono
proprio a coprire questa differenza. Non è garantita una specifica accelerazione.

Riferimento del ridimensionamento intero: implementazione ufficiale OpenCV
4.11, `modules/imgproc/src/resize.cpp`:
https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/resize.cpp

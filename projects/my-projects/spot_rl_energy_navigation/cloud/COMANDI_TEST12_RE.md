# Test12 SAC Re — comandi completi sulla GPU online

Pacchetto unico: `spot-test12-sac-re-cloud.zip`. Include codice, mappa 15×15 m del test11, robot, controllore di locomozione, camera con dipendenze e pesi DINO. Non occorre un secondo archivio dei terreni, clonare GitHub o caricare checkpoint precedenti. Il setup scarica Isaac Sim/Isaac Lab e le librerie Python: serve Internet durante l'installazione.

Configurazione: SAC da zero, 128 Spot indipendenti, 1.500.000 transizioni totali, 16 aggiornamenti SAC per passo vettoriale con campioni di navigazione, DINO batch 128, backend CUDA. Goal /10 fisso, altri scaler dinamici come nel test11/test7. PCA calibrata automaticamente al primo avvio.

## 1. Avviare la VM

Scegliere Linux Ubuntu 22.04 o 24.04 x86_64 con accesso grafico NVIDIA/Vulkan, GPU RTX e driver compatibili con Isaac Sim 5.1 e PyTorch cu128. La RTX 5090 e la RTX 6000 Pro non richiedono codice differente. Prevedere spazio per runtime, checkpoint e almeno due recovery: indicativamente 100 GB di disco o più, controllando lo spazio disponibile durante il training. I 32 GB di VRAM della 5090 sono il profilo previsto; la disponibilità reale si verifica all'avvio.

Prendere dalla piattaforma **IP pubblico e porta SSH della nuova VM**. Nei comandi seguenti sostituire `IP_GPU` e `PORTA_SSH`; non riutilizzare automaticamente quelli della vecchia macchina.

I comandi assumono accesso come `root`. Per semplicità usano `/root/test12-re-runs`: questo percorso NON è garantito persistente se la VM viene cancellata. Se il fornitore offre un volume persistente, sostituire quel percorso in TUTTI i comandi, incluso scp finale, con il percorso reale del volume. Prima di rilasciare la VM scaricare comunque i risultati.

## 2. Caricare dal PC Windows

Salvare ZIP e relativo `.zip.sha256` in `C:\Users\babir\Downloads`. Eseguire in un terminale Windows (non nel terminale Linux della GPU):

```bat
scp -P PORTA_SSH "C:\Users\babir\Downloads\spot-test12-sac-re-cloud.zip" root@IP_GPU:/root/
scp -P PORTA_SSH "C:\Users\babir\Downloads\spot-test12-sac-re-cloud.zip.sha256" root@IP_GPU:/root/
ssh -p PORTA_SSH root@IP_GPU
```

La prima connessione può chiedere conferma della chiave SSH: confrontarla con la fingerprint comunicata dal fornitore. Usare `/root/`, non `/~` (che nel vecchio trasferimento era diventato il nome di un file).

## 3. Preparare Linux ed estrarre

Dopo il comando ssh si è nel terminale della GPU. Eseguire nell'ordine:

```bash
nvidia-smi
df -h /root
cd /root
sha256sum -c spot-test12-sac-re-cloud.zip.sha256
apt-get update
apt-get install -y unzip tmux
unzip spot-test12-sac-re-cloud.zip
cd /root/spot-test12-sac-re-cloud
bash setup_test12_re.sh --check-only
```

Il checksum deve rispondere `OK`, il controllo file deve terminare senza errori. Non estrarre sopra una copia già modificata da un training: per una VM nuova la directory non deve esistere. Non reinstallare il driver NVIDIA e non installare manualmente un CUDA Toolkit: il setup gestisce il runtime Python.

## 4. Installare dentro tmux

```bash
tmux new -s test12-re
```

Si apre una nuova sessione terminale. Al suo interno:

```bash
cd /root/spot-test12-sac-re-cloud
bash setup_test12_re.sh
```

Attendere il completamento senza errori. Scarica e installa il runtime, verifica CUDA e DINO e apre un controllo headless di due robot; NON avvia ancora il training. Log di installazione in `.spot-runtime/logs/`. Se l'installazione fallisce, correggere l'errore e rilanciare lo stesso setup, senza lanciare il training prima del controllo riuscito.

Se SSH si disconnette, riconnettersi dal PC con `ssh -p PORTA_SSH root@IP_GPU`, quindi eseguire `tmux attach -t test12-re`: non serve riavviare setup/training se è ancora in corso.

## 5. Avviare il training

Nella stessa sessione tmux, dopo il setup:

```bash
cd /root/spot-test12-sac-re-cloud
bash start_test12_re.sh /root/test12-re-runs \
  --device cuda:0 --agent-device cuda:0 \
  --num-envs 128 --dino-batch-size 128 --gradient-steps 16 \
  --map-backend cuda --rgb-backend cuda --replay-device cuda:0 \
  --timesteps 1500000
```

Questi argomenti sono già i default, ma sono esplicitati per rendere visibile tutta la configurazione. Il launcher verifica la mappa, calibra la PCA sui due materiali e avvia l'RL in un processo separato. La calibrazione iniziale richiede tempo e può non apparire ancora nel monitor del training.

La prima verifica RGB sui frame reali deve riportare equivalenza riuscita. Un errore di setup/CUDA/equivalenza non va ignorato. Per uscire da tmux lasciando il training attivo: premere **Ctrl+B**, rilasciare, poi **D**. NON Ctrl+C, che richiede l'arresto del training.

## 6. Monitorare

Aprire un secondo terminale Windows:

```bat
ssh -p PORTA_SSH root@IP_GPU
```

Nel nuovo terminale della GPU:

```bash
cd /root/spot-test12-sac-re-cloud
watch -n 10 bash monitor_test12_re.sh /root/test12-re-runs
```

Per vedere tutto il report, senza il limite verticale di watch, uscire da watch con Ctrl+C (ferma SOLO watch) e lanciare:

```bash
bash monitor_test12_re.sh /root/test12-re-runs
```

`-n 10` aggiorna la schermata ogni 10 secondi. Le valutazioni avvengono ogni 10k transizioni nelle fasi 0–1 e ogni 30k nelle fasi 2–3, su 128 episodi deterministici. Promozione con almeno 80% di successi in due valutazioni consecutive. Il monitor può non aggiornarsi durante una valutazione, un backup o l'inizializzazione: non significa da solo che il processo sia bloccato.

## 7. Arrestare e riprendere se necessario

Non è necessario farlo per un training che procede regolarmente. Per arrestarlo in modo ordinato:

```bash
tmux attach -t test12-re
```

Premere Ctrl+C **una volta** nel terminale del training; aspettare salvataggio completo e chiusura. Non usare kill -9 o spegnere la VM per fermarlo. Recovery ogni 30 minuti, all'inizio, alla promozione, all'interruzione ordinata e alla fine; si conservano due generazioni verificate. In caso di spegnimento brutale si perdono solo i progressi successivi all'ultima recovery valida, purché il disco sia ancora disponibile.

Per riprendere l'ultima run, dallo stesso progetto/runtime:

```bash
cd /root/spot-test12-sac-re-cloud
PREVIOUS_RUN=$(cat /root/test12-re-runs/latest_run.txt)
bash start_test12_re.sh /root/test12-re-runs \
  --resume "$PREVIOUS_RUN" --timesteps 1500000
```

Leggere `latest_run.txt` PRIMA di chiamare start, perché start aggiorna quel file alla nuova destinazione. Il traguardo 1.500.000 comprende le transizioni già fatte. Il resume verifica checkpoint e replay e può ripiegare sulla recovery precedente se l'ultima è incompleta. Gli episodi simulati vengono resettati: non è una ripresa dello stesso fotogramma, ma ripristina learner, ottimizzatori, scaler, curriculum, replay e RNG.

## 8. A fine training: verificare che abbia finito

Nel terminale della GPU:

```bash
cd /root/spot-test12-sac-re-cloud
bash monitor_test12_re.sh /root/test12-re-runs
RUN_DIR=$(cat /root/test12-re-runs/latest_run.txt)
cat "$RUN_DIR/summary.json"
cat "$RUN_DIR/recovery_status.json"
ls -lh "$RUN_DIR/policy_final.pt" "$RUN_DIR/pca32.npz"
```

Aspettare il messaggio di salvataggio finale, `summary.json` con `interrupted: false` e 1.500.000 transizioni, e recovery finale disponibile. Il solo 100% del contatore non garantisce che valutazione e scrittura finale siano terminate. `policy_best.pt` e `policy_best_stage_*.pt` sono i modelli selezionati dalle valutazioni e possono essere migliori del final.

## 9. Salvare sul PC PRIMA di spegnere o rilasciare la VM

### Copia completa raccomandata, per poter continuare il training

In Windows, creare una directory e scaricare TUTTA la radice delle run, incluse le run precedenti se si è fatto resume:

```bat
mkdir "C:\Users\babir\Downloads\backup_test12_re"
scp -r -P PORTA_SSH root@IP_GPU:/root/test12-re-runs "C:\Users\babir\Downloads\backup_test12_re"
scp -P PORTA_SSH root@IP_GPU:/root/spot-test12-sac-re-cloud/spot_rl_energy_navigation/train_general_big_map_test12_SAC_Re/artifacts/pca32_lab.npz "C:\Users\babir\Downloads\backup_test12_re"
scp -P PORTA_SSH root@IP_GPU:/root/spot-test12-sac-re-cloud/spot_rl_energy_navigation/train_general_big_map_test12_SAC_Re/artifacts/manifest.json "C:\Users\babir\Downloads\backup_test12_re"
```

`-r` è indispensabile per copiare una cartella. La copia completa può essere di molte decine di GB: aspettarne il completamento, controllare che la cartella locale contenga `recovery/` nelle run e conservare anche lo ZIP originale e il suo checksum. Non eliminare prima i file remoti.

I file PCA e manifest scaricati separatamente permettono di conservare la calibrazione del progetto, oltre alla PCA già incorporata nei checkpoint e in ogni run. Su un'altra macchina, setup del pacchetto originale va eseguito PRIMA di sostituire eventuali file con le copie post-calibrazione: il checksum del pacchetto descrive i file originali.

### Cosa è importante e perché

| File/cartella nella run | Uso |
| --- | --- |
| `recovery/` intera + `recovery_status.json` | Ripresa completa; checkpoint, replay abbinato, manifest/hash. Non tenere solo policy.pt |
| `policy_final.pt`, `policy_best.pt`, `policy_best_stage_*.pt` | Valutare e confrontare policy; il checkpoint best non include necessariamente un replay abbinato |
| `pca32.npz`, `config.json`, `curriculum.json`, `summary.json`, `sources/` | Calibrazione, configurazione e riproducibilità |
| `training.csv`, `episodes.csv`, `evaluations.csv`, tutti gli `evaluation_*.json` | Grafici e analisi di successi, energia e fallimenti |
| `actor.pt` | Actor esportato per inferenza |
| Checkpoint intermedi, eventuali `failure.json`, `timing.json` | Confronti nel tempo e diagnosi |

Per non ripetere il problema del vecchio test7, la scelta più sicura è conservare **tutta la cartella test12-re-runs**, senza selezionare solo i checkpoint. Non serve scaricare `.spot-runtime/` (ricreabile dal setup), cache, librerie installate o l'intera VM.

Le valutazioni del training salvano statistiche ma non tutte le traiettorie XY: per i consueti grafici start/goal/percorso si eseguirà un test di playback con i checkpoint scelti e si conserveranno anche le relative cartelle `results/`.

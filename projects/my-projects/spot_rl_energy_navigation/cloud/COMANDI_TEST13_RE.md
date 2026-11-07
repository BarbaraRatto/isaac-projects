# Test13 SAC Re — istruzioni riassuntive

ZIP unico completo: `spot-test13-sac-re-cloud.zip`. Include training, mappa grigio/nero, robot, camera, DINO, test di 9 episodi e script dei grafici. Nessun secondo archivio dei terreni. Da zero, 128 Spot, 1.500.000 transizioni, 16 aggiornamenti SAC. Start e goal sempre sul grigio. Nero: attrito statico 0,10, dinamico 0,08, nessuna resistenza viscosa.

## 1. VM e trasferimento

Ubuntu 22.04/24.04 x86_64, GPU RTX con supporto grafico/Vulkan e driver compatibile col runtime del test12. Nessun vincolo alla 5090. Prevedere indicativamente almeno 100 GB di disco e controllare lo spazio residuo. Setup scarica Isaac Sim/Isaac Lab e librerie: serve Internet.

Sostituire `IP_GPU` e `PORTA_SSH` con quelli della NUOVA VM. I comandi assumono utente root. `/root/test13-re-runs` non è garantito persistente dopo l'eliminazione della VM: se il provider offre un volume persistente, sostituire questo percorso ovunque con quello reale. Scaricare comunque il backup prima di rilasciare la macchina.

Da Windows, dopo aver salvato ZIP e checksum in Downloads:

```bat
scp -P PORTA_SSH "C:\Users\babir\Downloads\spot-test13-sac-re-cloud.zip" root@IP_GPU:/root/
scp -P PORTA_SSH "C:\Users\babir\Downloads\spot-test13-sac-re-cloud.zip.sha256" root@IP_GPU:/root/
ssh -p PORTA_SSH root@IP_GPU
```

Alla prima connessione confrontare la chiave SSH con quella del fornitore. Usare `/root/`, non `/~`.

## 2. Estrarre, controllare, installare

Sulla GPU:

```bash
nvidia-smi
df -h /root
cd /root
sha256sum -c spot-test13-sac-re-cloud.zip.sha256
apt-get update
apt-get install -y unzip tmux
unzip spot-test13-sac-re-cloud.zip
cd /root/spot-test13-sac-re-cloud
bash setup_test13_re.sh --check-only
tmux new -s test13-re
```

Checksum deve rispondere OK. La directory di estrazione deve essere nuova; non sovrascrivere un progetto già in uso.

Dentro tmux:

```bash
cd /root/spot-test13-sac-re-cloud
bash setup_test13_re.sh
```

Aspettare il completamento senza errori. Installa runtime, librerie e matplotlib, verifica CUDA/DINO e due robot; non avvia RL e non calibra la PCA nel probe. Non installare a mano driver o CUDA Toolkit. Log in `.spot-runtime/logs/`.

## 3. Avviare il training

Nella stessa sessione tmux:

```bash
bash start_test13_re.sh /root/test13-re-runs \
  --device cuda:0 --agent-device cuda:0 \
  --num-envs 128 --dino-batch-size 128 --gradient-steps 16 \
  --map-backend cuda --rgb-backend cuda --replay-device cuda:0 \
  --timesteps 1500000
```

Prima calibra automaticamente la PCA grigio/nero, poi avvia RL. Durante la calibrazione il monitor può non mostrare ancora progressi. Il controllo RGB sui frame reali deve riuscire. Per staccarsi lasciando il processo acceso: Ctrl+B, rilasciare, D. Per rientrare dopo nuova connessione SSH: `tmux attach -t test13-re`.

## 4. Monitorare

Da un secondo terminale SSH:

```bash
cd /root/spot-test13-sac-re-cloud
watch -n 10 bash monitor_test13_re.sh /root/test13-re-runs
```

Per vedere tutto il report uscire da watch con Ctrl+C e usare:

```bash
bash monitor_test13_re.sh /root/test13-re-runs
```

Valutazioni su 128 episodi ogni 10k transizioni nelle fasi 0–1 e ogni 30k nelle fasi 2–3. Promozione con almeno 80% in due valutazioni consecutive. Durante valutazioni e backup il log può temporaneamente non avanzare.

## 5. Interruzione e ripresa, solo se necessario

Nel terminale del training Ctrl+C UNA volta; aspettare salvataggio completo e chiusura. Non usare kill -9 o lo spegnimento per interromperlo. Recovery completa ogni 30 minuti, iniziale, promozione, interruzione ordinata e finale; due generazioni con checksum e replay abbinato. Un crash può perdere i progressi successivi all'ultima recovery valida.

Per riprendere:

```bash
cd /root/spot-test13-sac-re-cloud
PREVIOUS_RUN=$(cat /root/test13-re-runs/latest_run.txt)
bash start_test13_re.sh /root/test13-re-runs \
  --resume "$PREVIOUS_RUN" --timesteps 1500000
```

Il traguardo include le transizioni già completate; su una run già terminata bisogna aumentarlo per estenderla. Si crea una nuova run, mantenere anche quella precedente. Gli episodi fisici ripartono da reset. Non caricare checkpoint del test12: identità di mappa, materiali, PCA e missioni sono differenti.

## 6. A fine training: verifica, test da 9 episodi, grafici

Aspettare messaggio di salvataggio finale e chiusura del processo, non solo contatore al 100%.

```bash
cd /root/spot-test13-sac-re-cloud
RUN_DIR=$(cat /root/test13-re-runs/latest_run.txt)
cat "$RUN_DIR/summary.json"
cat "$RUN_DIR/recovery_status.json"
```

Il summary deve avere `interrupted: false` e 1.500.000 transizioni. Poi lanciare il test headless sulla GPU:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/test_9.sh \
  --runs-root /root/test13-re-runs --headless
```

Seleziona best dell'ultima run completata. Nove episodi sequenziali: 2 vicini, 2 a circa 6 m, 2 a 9 m, 2 a 12 m, 1 con ghiaccio sulla retta A–B. Tutti grigio→grigio. I percorsi sono decisi dalla policy. Per confrontare il final sugli stessi punti:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/test_9.sh \
  --checkpoint "$RUN_DIR/policy_final.pt" --headless
```

Generare grafici e tabella dell'ULTIMO test completato, senza Isaac Sim o ChatGPT:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/analyze_results.sh
```

Per un test precedente, indicarne la directory:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/analyze_results.sh \
  spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/results/NOME_TEST
```

Output dentro `results/NOME_TEST/analysis/`: mappa_percorsi PNG/PDF, percorsi_per_episodio PNG/PDF, diagnostica_episodi.pdf, riepilogo_episodi.csv, REPORT.md e analisi.json. Non viene usata la policy durante la produzione dei grafici.

## 7. Salvare sul PC prima di eliminare la VM

Nel terminale Windows:

```bat
mkdir "C:\Users\babir\Downloads\backup_test13_re"
scp -r -P PORTA_SSH root@IP_GPU:/root/test13-re-runs "C:\Users\babir\Downloads\backup_test13_re"
scp -r -P PORTA_SSH root@IP_GPU:/root/spot-test13-sac-re-cloud/spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/results "C:\Users\babir\Downloads\backup_test13_re"
scp -P PORTA_SSH root@IP_GPU:/root/spot-test13-sac-re-cloud/spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/artifacts/pca32_lab.npz "C:\Users\babir\Downloads\backup_test13_re"
scp -P PORTA_SSH root@IP_GPU:/root/spot-test13-sac-re-cloud/spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/artifacts/manifest.json "C:\Users\babir\Downloads\backup_test13_re"
```

Conservare anche ZIP e checksum originali. `-r` è indispensabile per le cartelle. Aspettare il completamento e verificare i file locali prima di eliminare la VM; la copia delle run può essere di molte decine di GB.

| Da conservare | Perché |
| --- | --- |
| Tutta `test13-re-runs`, soprattutto `recovery/` e `recovery_status.json` | Ripresa completa con replay, learner, ottimizzatori, scaler, curriculum e RNG |
| `policy_best.pt`, `policy_final.pt`, `policy_best_stage_*.pt`, intermedi | Confronti e test; i checkpoint ordinari non bastano da soli alla ripresa completa |
| PCA, config, curriculum, summary e sources nella run | Calibrazione e riproducibilità |
| CSV e `evaluation_*.json` nella run | Analisi dell'apprendimento |
| `results/` completa | Traiettorie e grafici dei 9 episodi |
| PCA/manifest post-calibrazione scaricati separatamente | Copia della calibrazione del progetto |

Non serve scaricare `.spot-runtime/` o la VM intera. Per trasferire su un'altra macchina installare prima il pacchetto originale; solo dopo recuperare eventuali PCA/manifest modificati dal training. Il checksum del bundle descrive la copia originale pre-calibrazione.

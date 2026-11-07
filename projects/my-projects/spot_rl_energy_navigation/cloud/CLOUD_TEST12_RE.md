# Test12 SAC Re sulla GPU online

Il pacchetto unico è `spot-test12-sac-re-cloud.zip`. Per la sequenza completa, dal trasferimento al backup, vedere [COMANDI_TEST12_RE.md](COMANDI_TEST12_RE.md), incluso anche come `LEGGIMI_TEST12_RE.md` alla radice dello ZIP. I comandi seguenti sono i percorsi interni del progetto.

## Installazione e verifica

```bash
bash spot_rl_energy_navigation/cloud/setup_test12_re.sh
```

Il setup verifica robot, camera, DINO, mappa e missioni, installa il runtime e controlla CUDA/DINO e un ambiente minimo con due robot. Non avvia l'RL né calibra la PCA durante il probe. Al primo avvio del training la PCA viene calibrata automaticamente sui due materiali.

Se il runtime del test9 è ancora presente e funzionante, è possibile riutilizzare solo il software, senza caricare pesi o replay del test9:

```bash
bash spot_rl_energy_navigation/cloud/setup_test12_re.sh \
  --reuse-existing /root/spot-test9-sac-cloud
```

Non eliminare la vecchia directory software se viene riutilizzata. Per controllare solo i file, senza installare:

```bash
bash spot_rl_energy_navigation/cloud/setup_test12_re.sh --check-only
```

## Avvio

Usare una directory su un volume che il fornitore garantisce persistente. `/root` non è necessariamente persistente quando la VM viene eliminata. Sostituire il percorso d'esempio con quello reale del volume.

```bash
tmux new -s test12-re
bash spot_rl_energy_navigation/cloud/start_test12_re.sh \
  /PERCORSO_PERSISTENTE/test12-re-runs
```

Default: 128 robot, 1.500.000 transizioni, 16 aggiornamenti SAC, DINO batch 128, simulazione/learner/replay/mappa/RGB su CUDA:0. Non servono altri parametri per questa configurazione. Il comando crea una nuova run e aggiorna `latest_run.txt`.

Con `Ctrl+B`, poi `D`, si lascia tmux senza fermare il processo; `tmux attach -t test12-re` riapre il terminale. Tmux protegge dalla chiusura SSH, non dallo spegnimento della VM.

In un secondo terminale, dalla radice del progetto:

```bash
watch -n 10 bash spot_rl_energy_navigation/cloud/monitor_test12_re.sh \
  /PERCORSO_PERSISTENTE/test12-re-runs
```

Per leggere il report senza il limite verticale di watch, eseguire una volta lo stesso comando senza `watch -n 10`. Il monitor mostra backend, recovery disponibile, avanzamento, successi, curriculum e prossima valutazione. Per tempi dei componenti aggiungere `--profile-timing` al comando di training; il report viene salvato in `timing.json`.

## Arresto e ripresa

Premere Ctrl+C **una volta** nel terminale del training e aspettare il messaggio di recovery e la chiusura. Verificare `recovery_status.json`. Per riprendere, passare la cartella della run precedente: il resolver trova l'ultimo snapshot completo, verifica hash e replay, e prova la generazione precedente se necessario.

```bash
bash spot_rl_energy_navigation/cloud/start_test12_re.sh \
  /PERCORSO_PERSISTENTE/test12-re-runs \
  --resume /PERCORSO_PERSISTENTE/test12-re-runs/NOME_RUN_PRECEDENTE \
  --timesteps 1500000
```

`--timesteps` è il traguardo **totale**, non il numero di transizioni aggiuntive. La ripresa crea una nuova run; conserva entrambe le directory. Gli episodi fisici vengono resettati. La copia completa è riutilizzabile anche su un'altra GPU compatibile, mantenendo codice, PCA, mappa, configurazione e asset.

Salvare l'intera run, soprattutto `recovery/`: una policy ordinaria non basta per ripristinare tutto il replay. Le due generazioni di recovery occupano diversi GiB, oltre ai checkpoint ordinari, ai log e al runtime. Scegliere un volume con spazio sufficiente; sono presenti controlli preventivi dello spazio e scritture atomiche. Copiare anche fuori dalla VM i salvataggi importanti.

# Setup GPU cloud per Spot

Questo setup prepara un ambiente privato riutilizzabile per il test3 e per gli
esperimenti successivi che usano lo stesso stack. Non modifica il Python della
macchina, non installa driver e non avvia il training completo.

Versioni: Python 3.11, Isaac Sim 5.1.0, Isaac Lab v2.3.0, PyTorch 2.7.0 CUDA 12.8,
torchvision 0.22.0, NumPy 1.26.4. Transformers 5.8.1 e safetensors 0.7.0 sono le
versioni rilevate nell'installazione originale; i pesi DINO sono copiati e
controllati contro gli hash della PCA. La procedura segue
[Isaac Lab 2.3](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html).
Le dipendenze transitive risolte vengono registrate in
`.spot-runtime/requirements-resolved.txt`; non sono tutte bloccate a priori.

## 1. Preparazione sulla macchina originale

Dalla directory `projects/my-projects`:

```bash
python3 spot_rl_energy_navigation/cloud/package.py
```

Crea `spot_rl_energy_navigation/cloud/dist/spot-cloud-training.tar.gz`.
Il pacchetto contiene codice test3, setup, robot/materiali, policy locomotoria,
DINO esatto, mappa, PCA, banca dei punti e camera ZED 5.1 locale. Include anche
i file ignorati da Git; esclude run, risultati, analisi, cache e configurazione
dell'ambiente. Non serve clonare la repo sul server se usi questo pacchetto.

Rigeneralo dopo aver modificato il codice. Per includere anche un nuovo esperimento:

```bash
python3 spot_rl_energy_navigation/cloud/package.py \
  --experiment train_general_big_map_test3 \
  --experiment train_general_big_map_test4
```

Questo aggiunge la cartella indicata con i suoi artefatti; eventuali nuove
dipendenze esterne al folder vanno aggiunte esplicitamente al packager/setup.
Il test3 resta incluso per la verifica dell'ambiente. `--zed /percorso/ZED_X.usdc`
seleziona un percorso diverso della camera. Il packager non genera mappe/PCA e
non sostituisce i pesi con revisioni nuove scaricate da Internet.

## 2. Scelta della macchina Vast.ai

- Linux Ubuntu 22.04/24.04 **x86_64**, con Python 3, root o sudo senza password.
- Una GPU RTX: L40S o RTX 6000 Ada da 48 GB sono il profilo iniziale previsto.
- Driver host compatibile con Isaac Sim 5.1 e CUDA 12.8 (scegliere un host con
  driver NVIDIA recente, ad esempio ramo 580 o successivo).
- Come punto di partenza, 64 GB RAM e almeno 150 GB di disco libero per ambiente,
  download, cache e risultati; riservare ulteriore spazio per training lunghi.
- Il template/container deve esporre anche **graphics**, oltre a compute/utility.
  Impostare `NVIDIA_DRIVER_CAPABILITIES=all` nella configurazione del container
  **prima di avviarlo**. Esportarlo in SSH dopo l'avvio non monta le librerie mancanti.
- Usare un template Ubuntu/CUDA con accesso SSH e librerie grafiche NVIDIA
  esposte, oppure un template Isaac Sim compatibile. Non serve Docker dentro
  Docker. Questo setup gira direttamente nel container/host assegnato.

`nvidia-smi` da solo non verifica Vulkan/RTX. Se il provider non espone le
librerie grafiche del driver, lo script non può correggere il driver dell'host:
la prova di simulazione segnalerà il problema. Non usare WSL, A100/H100 o istanze
solo compute. [Requisiti NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

## 3. Una sola installazione sul server

Carica il pacchetto via SCP/SFTP nella directory persistente scelta (per esempio
`/workspace`), usando porta e host SSH mostrati da Vast.ai:

```bash
scp -P PORTA_SSH spot_rl_energy_navigation/cloud/dist/spot-cloud-training.tar.gz \
  UTENTE@HOST:/workspace/
```

Poi nel terminale SSH del server:

```bash
cd /workspace
tar -xzf spot-cloud-training.tar.gz
cd spot-cloud-training
bash spot_rl_energy_navigation/cloud/setup.sh
```

Scegli una destinazione nuova quando estrai un nuovo pacchetto: non sovrascrivere
una cartella con modifiche locali senza salvarle prima. Lo script:

1. Controlla file, hash, sistema e visibilità della GPU prima dei download grandi.
2. Installa librerie di sistema, uv privato, Python 3.11 e ambiente isolato.
3. Scarica Isaac Sim e Isaac Lab e installa le dipendenze del progetto.
4. Verifica un kernel CUDA, inferenza DINO e compatibilità con la PCA.
5. Esegue `--probe-steps 2` del test3: avvia fisica e camere, senza aggiornare PPO.

Al primo avvio potresti dover leggere e accettare la licenza NVIDIA nel terminale.
Le estensioni e gli shader possono richiedere diversi minuti. Rilanciare lo stesso
comando riutilizza checkout, ambiente e cache; un'installazione interrotta può
essere ripresa. Rilanciandolo si ripetono le verifiche e la breve prova.

Opzioni:

```bash
# Solo file/hash, nessuna installazione né GPU richiesta:
bash spot_rl_energy_navigation/cloud/setup.sh --check-only

# Ambiente in un altro volume persistente:
bash spot_rl_energy_navigation/cloud/setup.sh --runtime-dir /workspace/spot-runtime

# Se le librerie di sistema sono già installate:
bash spot_rl_energy_navigation/cloud/setup.sh --skip-apt

# Installa e verifica CUDA/DINO; non certifica fisica/rendering:
bash spot_rl_energy_navigation/cloud/setup.sh --skip-probe
```

Log, versioni risolte e commit Isaac Lab sono nella directory runtime. Solo una
prova di simulazione completata crea `simulator-probe.ok`. Le verifiche dipendenti
dalla GPU devono essere eseguite sull'istanza: non sono state certificate da un
test sulla macchina di sviluppo.

## 4. Training attuali e successivi

Dalla directory del progetto:

```bash
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test3
```

Funziona anche il comando originale del test3: carica automaticamente `cloud/env.sh`
quando non è stato impostato esplicitamente un altro launcher Isaac Lab.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test3/train.sh
```

Con argomenti aggiuntivi o per riprendere un checkpoint:

```bash
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test3 \
  --resume /percorso/policy_latest.pt --timesteps 400000
```

Per esperimenti futuri con un `train.sh` analogo:

```bash
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test4
```

Per qualsiasi script Isaac Lab:

```bash
bash spot_rl_energy_navigation/cloud/run.sh --python \
  spot_rl_energy_navigation/train_general_big_map_test3/play.py \
  --headless --checkpoint /percorso/policy_final.pt
```

Puoi anche eseguire `source spot_rl_energy_navigation/cloud/env.sh` in un terminale
nuovo. Per nuovi esperimenti, riutilizza la risoluzione di `ZED_USD` presente nel
nuovo `assets.py` del test3 (supporta `SPOT_ZED_USD`) e mantieni il supporto a
`GENERAL_MAP_ISAACLAB` nel launcher. Il setup non riscrive automaticamente tutti
i vecchi percorsi assoluti nei vari esperimenti.

Per mantenere il training attivo dopo la disconnessione SSH:

```bash
tmux new -s spot
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test3
# Ctrl+B, poi D per staccarsi. Rientro: tmux attach -t spot
```

I risultati restano nelle cartelle `runs/` degli esperimenti: copiali fuori
dall'istanza prima di eliminarla. Il setup non sincronizza automaticamente i run
e non allena nuovi algoritmi o dipendenze future senza aggiornare i requisiti.

# Test 9 SAC — pacchetto cloud completo

Questo archivio contiene test 9, mappa, tutte le texture, PCA congelata, banco
missioni, Spot, policy di locomozione, camera ZED e pesi DINO esatti.
**Non servono il vecchio archivio del test 7 né l'archivio separato dei terreni.**
Nessuna policy di navigazione precedente è inclusa: si parte da zero, salvo
richiedere esplicitamente `--resume` con una run completa di test 9.

Il pacchetto non contiene l'installazione di Isaac Sim: lo script la scarica
insieme a Isaac Lab e alle librerie. Serve Internet durante il setup.

## Macchina e spazio

Usare una GPU RTX con accesso grafico/Vulkan oltre a CUDA, Linux Ubuntu
22.04/24.04 x86_64 e root/sudo per le librerie di sistema. È lo stesso profilo
di installazione usato per il test 7 (Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python
3.11, PyTorch 2.7.0 CUDA 12.8). Il setup non installa o modifica driver host.

Scegliere un volume persistente della piattaforma. **Un percorso chiamato
`/workspace` non garantisce da solo la persistenza:** dipende dal contratto
dell'istanza/provider. Copiare le run fuori dall'istanza prima di cancellarla.

Pianificare circa 200 GB di spazio libero complessivo per software, cache e
training: la run da 1,5 milioni conserva anche i checkpoint periodici per i test,
oltre alle due generazioni complete di ripresa. I salvataggi completi da soli
richiedono circa 10–15 GiB di margine. Usare `tmux` per non interrompere il processo
quando si chiude SSH.

## 1. Copia ed estrazione

Caricare **solo `spot-test9-sac-cloud.tar.gz`** nel volume scelto. Il file `.sha256`
è opzionale, utile per controllare il trasferimento. Nei comandi seguenti
`/workspace` è un esempio da sostituire con il proprio volume persistente.

```bash
cd /workspace
tar -xzf spot-test9-sac-cloud.tar.gz
cd spot-test9-sac-cloud
```

Estrarre in una cartella nuova. Non sovrascrivere una precedente installazione
contenente modifiche o risultati.

## 2. Installazione e verifica

```bash
bash spot_rl_energy_navigation/cloud/setup.sh
```

Il setup seleziona automaticamente il test 9 dal manifest del pacchetto:

1. Verifica checksum, mappa, texture, PCA, DINO e file trasferiti.
2. Prepara un ambiente Python privato e installa le versioni del test 7.
3. Controlla CUDA e inferenza DINO/PCA.
4. Avvia una breve prova di Isaac Sim con **due robot e due passi**, senza
   aggiornamenti SAC e senza avviare il training completo.

Al primo avvio leggere e accettare la licenza NVIDIA quando richiesto.
Se il setup viene interrotto, si può rilanciare lo stesso comando. Il successo
del probe a due robot non certifica la memoria richiesta dai 128 robot.

Verifica solo file, senza scaricare/installare/avviare simulatori:

```bash
bash spot_rl_energy_navigation/cloud/setup.sh --check-only
```

`--skip-probe` salta il test del simulatore, `--runtime-dir /volume/runtime`
sceglie un'altra directory per l'installazione. Dettagli in
`spot_rl_energy_navigation/cloud/README.md` (i suoi comandi storici per il test 3
non sono necessari per questo pacchetto).

## 3. Training

Configurazione: **128 robot, 1.500.000 transizioni, 16 aggiornamenti SAC per
passo vettoriale valido**, energy weight 10, penalità temporale 0,10/passo.
Il goal è scalato /10; gli altri scaler sono adattivi, come nel test 7.
Spawn e goal escludono ghiaccio e scale. Mappa e PCA non vengono rigenerate.

Per utilizzare anche learner SAC e gridmap sulla GPU, come supportato dai
launcher del test 7 cloud, rendere esplicite le due opzioni di calcolo:

```bash
tmux new -s test9
bash start_test9.sh /workspace/test9-runs \
  --agent-device cuda:0 --map-backend cuda
```

Per usare esattamente i backend CPU predefiniti del launcher test 9:
`bash start_test9.sh /workspace/test9-runs`. Simulazione e DINO restano su CUDA.
Queste opzioni scelgono il dispositivo/backend di calcolo; non modificano reward,
rete, curriculum, transizioni o numero di aggiornamenti. La memoria e la velocità
effettive vanno verificate sulla macchina noleggiata.

Lo script crea una nuova directory nella destinazione indicata e scrive
`latest_run.txt`. I salvataggi non finiscono quindi in una posizione implicita.
Per staccarsi da tmux: Ctrl+B, poi D. Per rientrare: `tmux attach -t test9`.

## 4. Monitor

Da un secondo terminale nella cartella `spot-test9-sac-cloud`:

```bash
watch -n 10 bash monitor_test9.sh /workspace/test9-runs
```

Mostra transizioni, stima del tempo, risultati e **ultimo salvataggio completo**.
Durante l'inizializzazione mostra che il primo log non è ancora disponibile.

## 5. Ripresa dopo interruzione

Conservare **tutta la cartella della run**, inclusa `recovery/`, non soltanto
i `.pt`. Il sistema salva modello + replay + ottimizzatori + scaler + PCA +
curriculum + RNG periodicamente e verifica che i file siano completi.

```bash
bash start_test9.sh /workspace/test9-runs \
  --resume /workspace/test9-runs/NOME_RUN_INTERROTTA \
  --timesteps 1500000 \
  --agent-device cuda:0 --map-backend cuda
```

Il budget è **totale**: se la copia recuperata è a 700k, ne esegue circa altre
800k. Viene usata l'ultima generazione completa verificata; se corrotta, si
cerca quella precedente. Gli episodi fisici vengono resettati. È possibile
spostarsi su un'altra macchina con lo stesso pacchetto e stack software:
copiare anche l'intera run salvata. Cambiare il dispositivo di calcolo è ammesso;
numero di ambienti e gradient-steps devono corrispondere al checkpoint.

Per un controllo prima di lanciare la simulazione:

```bash
source spot_rl_energy_navigation/cloud/env.sh
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/recovery.py \
  /workspace/test9-runs/NOME_RUN_INTERROTTA
```

I checkpoint periodici `policy_*.pt` e `policy_best*.pt` sono per la valutazione;
per la ripresa usare la directory della run come sopra. Un arresto non gestibile
può perdere le transizioni dall'ultimo salvataggio (normalmente circa 30 minuti,
più eventuale valutazione/scrittura). Un volume eliminato non è recuperabile
senza una copia esterna.

## 6. Risultati da riportare sul PC locale

Per continuare ad allenare: tutta la run, in particolare entrambe le generazioni
in `recovery/`. Per sola valutazione: almeno il checkpoint scelto e i log.
La mappa portabile e l'identità degli asset del test 9 sono le stesse qui e nel
pacchetto cloud, così il checkpoint può essere testato sul PC locale aggiornato.

Per episodi headless sul cloud:

```bash
source spot_rl_energy_navigation/cloud/env.sh
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/test.sh \
  --checkpoint /workspace/test9-runs/NOME_RUN/policy_final.pt \
  --headless --episodes 9
```

Senza `--headless` sul PC con display si può vedere la simulazione. Per i grafici:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/analyze_paths.py \
  /PERCORSO_DELLA_CARTELLA_EVALUATION
```

La configurazione dettagliata e lo schema della mappa sono nel README del test 9.
Il pacchetto viene verificato offline prima della consegna; l'installazione e
il probe sullo specifico host cloud si eseguono con il comando di setup sopra.

# Parametri del training

Il launcher accetta `--num-envs N` e `--gradient-steps N` senza modificare
`config.py`. I valori devono essere positivi e vengono salvati nella firma del
checkpoint e nelle impostazioni del learner. Per riprendere un training usare
gli stessi valori. In assenza dei parametri si usano i valori di `SAC_CFG`.
Gli aggiornamenti SAC non vengono scalati automaticamente quando cambia il
numero di ambienti: il comando seguente mantiene circa il rapporto originale
(1 aggiornamento ogni passo di 32 ambienti).

```bash
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test7_SAC \
  --num-envs 128 --gradient-steps 4 \
  --dino-batch-size 128 --timesteps 400000 \
  --device cuda:0 --agent-device cuda:0 --map-backend cuda
```

Per una prova preliminare aggiungere `--probe-steps 50`. I parametri del training
non cambiano il numero di ambienti della calibrazione PCA separata, che mantiene
le sue impostazioni. Per la valutazione `play.py --num-envs` resta indipendente
dal numero di ambienti usato durante il training.

# Test 7 SAC — goal con scala fissa

Training da zero per **200.000 transizioni di navigazione complessive** su 32 ambienti,
con la stessa mappa congelata real_terrains_v2, PCA, curriculum, reward, architettura,
controllore e iperparametri SAC del test 6.

Unica modifica metodologica: goal XY robocentrico diviso per **10 metri**.
Nessuna sottrazione della media, nessuna varianza adattiva, nessun clipping sul goal.
La stessa trasformazione è usata nell'actor, nei due critic, nel test e nell'actor esportato.
Gli altri scaler continuano ad aggiornarsi come nel test 6; maschere e validità invariati.
Le statistiche dello stato ora contengono solo velocità lineari/angolari e potenza (7 valori).
La firma del checkpoint include la scala del goal; non riprendere checkpoint del test 6.

## Avvio dalla directory principale del progetto

```bash
bash spot_rl_energy_navigation/train_general_big_map_test7_SAC/train.sh
```

Lo script verifica e riusa la copia della PCA già calibrata sui sette terreni del test 6.
La mappa e gli artefatti di missione sono copie identiche; non vengono rigenerati.
Ogni lancio crea una nuova run, salvo uso esplicito di `--resume` con checkpoint del test 7.

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test7_SAC/status.py
```

Al termine, test visibile del modello finale più recente: nove missioni, seed 20000,
fase corrispondente all'ultima valutata durante il training.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test7_SAC/test.sh
```

Per confrontare esplicitamente la fase 1 con gli esperimenti precedenti, aggiungere `--stage 1`.
Non sono imposti nuovi limiti alla rotazione o penalità: il robot mantiene due azioni (vx, wz).

## Verifiche offline

```bash
PYTHONPATH=spot_rl_energy_navigation OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest -q spot_rl_energy_navigation/train_general_big_map_test7_SAC
```

Le verifiche includono regressione rispetto al normalizzatore del test 6 per tutti
gli input diversi dal goal, indipendenza del goal dalla distribuzione osservata,
checkpoint/export e training SAC con ambiente simulato fittizio (senza Isaac Sim).
# Mappe su GPU (opzionale)

`train.py` e `play.py` accettano `--map-backend cuda` (default: `cpu`).
Il backend Torch esegue sulla GPU backprojection RGB-D, esclusione dei punti
del robot, aggiornamento della memoria sparsa, aggregazione nelle celle locali,
pendenza, rugosità e invecchiamento. Le mappe di tutti gli ambienti sono
raggruppate; non vengono copiate in NumPy a ogni osservazione.

Il resize RGB OpenCV e il processor DINO restano quelli originali, per mantenere
il contratto con i pesi/PCA congelati. La calibrazione (`capture=True`) conserva
il percorso CPU. Replay, curriculum e altre parti del simulatore non vengono
spostati da questa opzione. I momenti geometrici usano float64 per conservare
precisione anche negli ambienti traslati. La memoria sparsa cresce con i bin
osservati e viene liberata al reset dell'episodio; non è troncata per velocizzare.

Prima di usare il backend sulla macchina cloud:

```bash
.spot-runtime/venv/bin/python spot_rl_energy_navigation/train_general_big_map_test7_SAC/check_gpu_maps.py --device cuda:0
bash spot_rl_energy_navigation/cloud/run.sh train_general_big_map_test7_SAC \
  --map-backend cuda --agent-device cuda:0 --dino-batch-size 32 --probe-steps 50
```

Il primo comando confronta l'aritmetica CUDA con il riferimento CPU, senza
avviare Isaac Sim. Il secondo verifica l'integrazione col simulatore.
Per il training completo togli `--probe-steps 50`. `--map-backend cpu` ripristina
il backend originale. Configurazione SAC, artefatti e formato dei checkpoint
restano invariati; il backend è registrato negli argomenti del run. Una ripresa
reinizializza gli episodi come prima e non garantisce una traiettoria bit-per-bit
identica (riduzioni parallele in floating point).

Validazione di sviluppo: test numerici Torch su CPU contro NumPy, incluse
maschere, aggiornamenti ripetuti, reset parziali, grandi origini, profondità
non valide e integrazione con estrattore simulato. La macchina di sviluppo
non dispone di CUDA funzionante: prestazioni, picchi VRAM e prova Isaac Sim
con questo backend devono essere misurati sull'istanza GPU prima di un run lungo.


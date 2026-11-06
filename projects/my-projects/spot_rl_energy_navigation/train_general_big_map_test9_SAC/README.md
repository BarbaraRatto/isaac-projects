# Test 9 SAC — nuova mappa e ripresa completa

Training da zero basato sul test 7 SAC. Nessun checkpoint dei test precedenti viene caricato automaticamente.

## Configurazione

| Impostazione | Valore |
|---|---|
| Budget totale | 1.500.000 transizioni di navigazione |
| Ambienti indipendenti | 128 copie della stessa mappa |
| Aggiornamenti SAC per passo vettoriale con campioni validi | 16, come nel test 7 cloud a 128 robot |
| Buffer | 100.000 transizioni, osservazioni grezze float32 |
| Batch / learning rate / gamma | 256 / 0,0003 / 0,999 |
| Actor | stato 15→64→32; mappa 3240→1024→512→256→128; fusione 160→128→64 |
| Azioni | vx 0–1,6 m/s; wz ±1,5 rad/s; vy=0 |
| Scaler | solo goal XY fisso /10; altri scaler adattivi, come test 7 |
| PCA | stessa PCA32 congelata del test 7, calibrata sui sette materiali |
| Curriculum | cinque livelli, promozione con ≥80% in due valutazioni consecutive |
| Valutazione periodica | ogni circa 10.000 transizioni, un episodio per robot sulla fase corrente |
| Spawn e goal | esclusi ghiaccio e scale, con gli stessi controlli locali del test 7 |
| Cadute | inclinazione ≥90° per 1 s oppure contatto del corpo col terreno >1 N per 1 s |

Restano identici allineamento iniziale, memoria della mappa, rete e aggiornamenti SAC,
criteri di terminazione e difficoltà del curriculum. Il budget conta le transizioni
di navigazione; allineamento iniziale e valutazioni richiedono tempo ma non sono
inseriti nel replay. Le copie della mappa restano separate.

## Reward

Per passo di navigazione di 0,2 s:

`r = 30*(d_precedente - d_attuale) - 0.10 - 10*(E_passo / 100 J) + r_orientamento + r_terminale`

- Goal: +1000. Fallimento: −1000.
- Tempo: −0,10 punti/passo = −0,5 punti/s (cinque volte il test 7).
- Energia meccanica: −0,1 punti/J (dieci volte il test 7).
- Orientamento invariato: zero fino a 110°, poi cresce linearmente fino a −2 punti/s a 180°.
- Esempio: 1500 J di navigazione costano 150 punti; risparmiarne 500 vale 50 punti.

L'energia totale riportata negli episodi include anche l'allineamento iniziale;
il replay include soltanto i passi controllati dalla policy. I log distinguono
energia di navigazione e di allineamento.

## Mappa congelata

File: `maps/real_terrains_test9.usd`. Celle 6×3 m; riga 1 in basso, colonne verso +X.
Geometrie, topologia, materiali e numero di celle per materiale sono preservati.
I riferimenti alle texture sono relativi alla radice del progetto, identici sul PC e nel pacchetto cloud.
Ghiaccio separato sia sui lati sia sulle diagonali. Le celle di scale restano
nella mappa e possono essere incontrate; non sono punti di spawn o goal.

| Riga / colonna | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 7 | G | O | A | I | G | R | O |
| 6 | O | G | P | S | R | P | G |
| 5 | R | I | G | A | I | A | A |
| 4 | O | S | G | P | O | G | S |
| 3 | I | R | P | R | S | P | S |
| 2 | P | S | I | A | I | R | O |
| 1 | I | P | O | A | A | R | S |

A asfalto; G ghiaia; R rocce; P rampe; S scale; O ostacoli; I ghiaccio.
`artifacts/map_layout.png` e `.pdf` mostrano la disposizione in scala.

Rispetto al test 7, scambi successivi con coordinate a partire da 1:

1. (2,4) ↔ (1,1).
2. (6,2) ↔ (4,6).
3. (7,6) ↔ (5,5), come richiesto.
4. (4,6) ↔ (3,1), per eliminare anche il contatto diagonale introdotto dallo scambio precedente.

Il nuovo spawn bank contiene 3500 punti locali, 2800 training e 700 valutazione.
Non è stato aggiunto un pianificatore né un filtro di raggiungibilità globale.

La PCA è copiata senza modificarne matrici o metadati. Il manifest registra
esplicitamente il trasferimento dalla mappa originale alla permutazione nuova;
i dati e le metriche di calibrazione continuano a riferirsi alla mappa originale.
I controlli verificano hash delle matrici, identità DINO/preprocessing e identità
esatta della nuova mappa e del nuovo banco missioni. Non vengono simulati nuovi
dati di calibrazione al lancio.

## Avvio

Dalla radice del progetto, con Isaac Lab e gli asset condivisi disponibili:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/train.sh
```

Per il cloud usare un percorso realmente persistente della piattaforma, non
il disco temporaneo dell'istanza. La cartella della singola run deve essere nuova:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/train.sh \
  --run-dir /PERCORSO_VOLUME_PERSISTENTE/test9/run01
```

`GENERAL_MAP_ISAACLAB` può indicare l'installazione di Isaac Lab.
Il launcher legge l'eventuale `../cloud/env.sh`. Backend mappa e learner restano
CPU per default, come nel launcher test 7; simulazione e DINO usano CUDA.
Sono disponibili gli override esistenti `--agent-device`, `--map-backend`,
`--num-envs`, `--gradient-steps`, `--torch-threads`, `--dino-batch-size`.
Non cambiare num-envs o gradient-steps durante la ripresa della stessa run.
Il pacchetto cloud include una guida dedicata; `start_test9.sh` permette di scegliere esplicitamente volume delle run e backend CUDA.

Monitoraggio (aggiungere `--run-dir` se si salva fuori dalla cartella standard):

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/status.py
```

## Salvataggi e ripresa

I modelli periodici `policy_*.pt`, `policy_latest.pt` e `policy_best*.pt` servono
alla valutazione. Non sono sufficienti da soli per riprendere SAC correttamente.

Il nuovo sistema salva una generazione completa:

- all'avvio, prima delle nuove transizioni;
- ogni 30 minuti, al confine di un passo completo (valutazioni/scrittura possono allungare l'intervallo);
- al cambio di fase;
- alla fine e su SIGINT/Ctrl+C o SIGTERM gestibili.

Ogni `recovery/snapshot_*/` contiene `policy.pt`, `replay.npz` e `manifest.json`:
actor, critic e target, ottimizzatori, temperatura SAC e relativo ottimizzatore,
scaler, PCA, replay grezzo e contatori, curriculum, stato casuale, identità degli
asset e configurazione. Il codice usato è anche in `sources/` della run.

La scrittura è sincrona, senza aggiornamenti nel mezzo; vengono eseguiti flush,
fsync e checksum prima del commit atomico della cartella. Si conservano almeno
le ultime due generazioni complete verificate. Le cartelle `.pending_*` lasciate
da un arresto brusco vengono ignorate. Non eliminare la generazione precedente
per liberare spazio prima che la nuova sia stata salvata.

Il monitor mostra l'ultimo salvataggio completo e il percorso. **Trasferire tutta
la cartella della run, inclusa `recovery/`, non solo i file `.pt`.** Sono inoltre
necessari lo stesso progetto, mappa, PCA e asset condivisi sul nuovo computer.

Ripresa automatica dall'ultima generazione verificata; se corrotta, cerca quella
precedente. Il target è il totale, incluse le transizioni già eseguite:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/train.sh \
  --resume /PERCORSO_VOLUME_PERSISTENTE/test9/run01 \
  --run-dir /PERCORSO_VOLUME_PERSISTENTE/test9/run02 \
  --timesteps 1500000
```

Verifica offline, senza simulazione:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/recovery.py \
  /PERCORSO_VOLUME_PERSISTENTE/test9/run01
```

Una GPU diversa è ammessa con software e configurazione compatibili. I robot
ripartono da nuovi reset: non si ripristina il fotogramma fisico dell'arresto.
Un errore CUDA o un aggiornamento interrotto da un'eccezione non viene promosso
a checkpoint valido; si usa la generazione completa precedente. SIGKILL, perdita
di alimentazione e cancellazione dell'istanza non permettono un salvataggio finale.
Un volume persistente/copertura esterna resta necessario per la perdita del disco.

Un replay pieno occupa circa 2,45 GiB più il modello (circa 170 MiB). Per due
generazioni e una in scrittura riservare 10–15 GiB; i checkpoint leggeri conservati
ogni 10k aggiungono circa 25 GiB a 1,5M. Consigliati almeno 50 GiB liberi per la
run completa, oltre a software e asset. Spazio insufficiente interrompe il processo
con un messaggio, preservando i salvataggi già completi.

## Valutazione e grafici

```bash
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/test.sh
```

Seleziona l'ultimo `policy_final.pt` locale e mostra nove episodi con un robot,
seed 20000, nella fase dell'ultima valutazione del training. Per un'altra run:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test9_SAC/test.sh \
  --checkpoint /PERCORSO_DELLA_RUN/policy_final.pt --stage 3 --episodes 9
```

Grafici con start, goal, percorso e tabella dopo la valutazione:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test9_SAC/analyze_paths.py \
  /PERCORSO_DEI_RISULTATI_EVALUATION
```

## Verifiche e dipendenze

`python3 -m pytest -q spot_rl_energy_navigation/train_general_big_map_test9_SAC`
esegue test numerici, missioni, reward, segnali e fault injection dei salvataggi.
Il ciclo di training e la ripresa sono verificati con un ambiente fittizio e
aggiornamenti SAC reali su piccoli dati sintetici, senza avviare Isaac Sim.

`check_setup.py` controlla asset, mappa, missioni e PCA senza simulazione.
`prepare_assets.py` e `prepare_map.py` servono solo alla costruzione offline e
rifiutano di sovrascrivere gli output congelati. Non vanno lanciati prima del training:
i file pronti sono già inclusi in questa cartella.

Asset condivisi: `IsaacRobotics/assets/spot.usd`, policy di locomozione Spot,
camera ZED (`SPOT_ZED_USD` o `.cloud_assets/ZED_X.usdc`), pesi DINO locali e
texture dei terreni. Il test 7 è necessario solo per ricostruire la mappa da zero,
non per usare il test 9 già preparato. Il pacchetto cloud completo è `cloud/dist/spot-test9-sac-cloud.tar.gz`; istruzioni in `cloud/CLOUD_TEST9.md`.
# Aggiornamento GPU compatibile con le run esistenti

Per installare il piccolo aggiornamento cloud, fermare con salvataggio e
riprendere il training esistente: [istruzioni complete](../cloud/TEST9_GPU_UPDATE.md).
Il launcher `resume_gpu.py RUNS_ROOT` verifica prima il salvataggio reale e il
preprocessing sulla GPU; conserva le impostazioni della run originale.
Le nuove opzioni sono `--rgb-backend cuda --replay-device cuda:0 --profile-timing`.
Reward, curriculum e firma dei checkpoint restano quelli del test9.


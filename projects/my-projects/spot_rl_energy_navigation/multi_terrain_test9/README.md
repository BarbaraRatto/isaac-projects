# Multi terrain test9 — SAC, 100.000 transizioni, energia 0,03

Copia del codice di test8, con la seguente configurazione corrente:

| Parametro | Test8 attuale | Test9 |
| --- | ---: | ---: |
| Transizioni totali | 100.000 | 100.000 |
| Penalità energetica per joule | 0,04 | 0,03 |

Il training parte da zero. Nessun checkpoint o replay precedente viene caricato.
100.000 transizioni sono esattamente 3.125 passi vettoriali di 32 ambienti.
Le valutazioni automatiche non contano nelle transizioni di training.

## Configurazione

- 32 Spot: 16 sulla mappa `rocks_gravel__right_rocks`, 16 sulla speculare
  `rocks_gravel__right_gravel`. Mappe 3×3, tasselli da 2 m, partenza (0,0),
  goal (4,4), ghiaccio centrale e corridoi rocce/ghiaia scambiati.
- SAC: buffer 100.000, batch 256, learning rate 0,0003, gamma 0,999,
  tau 0,005, entropia automatica, 4.096 transizioni iniziali casuali,
  un aggiornamento per passo vettoriale. Il buffer resta da 100.000:
  una volta pieno, sostituisce progressivamente le transizioni più vecchie.
- RGB-D e DINOv2-small congelato, 384 componenti senza PCA; 20 posizioni
  frontali con maschera di validità, nessuna memoria visiva tra frame.
- 7.711 input: 11 di stato/goal e 7.700 visivi. Il consumo non viene aggiunto
  agli input: contribuisce alla reward, come in test8.
- Ramo visivo 7700 → 2048 → 1024; ramo stato 11 → 128 → 128;
  fusione 1152 → 512 → 256, ELU. Encoder separati per actor e critic.
- Avanzamento 0–1,6 m/s e rotazione ±1,5 rad/s; nessun comando laterale
  o di retromarcia. Azione di arresto (-1,0).
- Fisica 200 Hz, decisioni ogni 0,2 s, episodi fino a 40 s.

Reward per azione:

```text
3 × (distanza_precedente − distanza_attuale)
− 0,03 × energia_meccanica_J_dello_step
− 0,01
+ 60 all'arrivo
− 25 per caduta o uscita dai limiti
```

Il timeout non aggiunge una penalità terminale. L'energia integra
`sum(abs(coppia_misurata × velocità_giunto))`; non misura la batteria.
80 J risparmiati corrispondono a 2,4 punti di reward non scontata.
La precedente run test9 usava 500.000 transizioni e coefficiente 0,275.
I suoi risultati e checkpoint restano conservati; i nuovi training usano
la configurazione corrente, senza cambiare gli altri termini della reward.

## Avvio da my-projects

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test9/train.py \
  --headless --timesteps 100000
```

Il simulatore usa le impostazioni standard di Isaac Lab; SAC usa CPU con
quattro thread per default, come test8. Non viene avviato dalla preparazione
dei file. Non serve cambiare config.py per usare questo budget.

Checkpoint e valutazione deterministica ogni 51.200 transizioni; valutazione
anche alla fine, con 16 episodi per mappa. Ogni valutazione resetta gli episodi
di training ancora aperti e non aggiunge dati al replay.

Output in `runs/test9_sac_32env_<timestamp>/`: configurazione, sorgenti,
hash dei terreni, metriche, episodi, valutazioni, `policy_best.zip` e
`policy_final.zip`. La selezione del migliore privilegia il successo su
entrambe le mappe, poi l'energia degli arrivi. Ctrl+C durante l'apprendimento
salva una policy interrotta; il replay non viene salvato e non è implementata
la ripresa del training.

## Play dopo il training

Il launcher `play_test9.sh` usa la policy finale della run completata
`test9_sac_32env_20261021_152607_024312`: **100.000 transizioni e peso
energetico 0,03**.

```bash
bash spot_rl_energy_navigation/multi_terrain_test9/play_test9.sh
```

Esegue 5 episodi per ciascuna delle due mappe in sequenza, con finestra
grafica. `--dry-run` mostra i comandi senza avviare Isaac Sim; `--episodes N`
cambia il numero per mappa. Per scegliere un altro checkpoint usare
`--checkpoint /percorso/al/checkpoint.zip`, oppure il comando seguente.

Per ripetere soltanto la mappa speculare (5 episodi), mantenendo i risultati
precedenti nella loro cartella:

```bash
bash spot_rl_energy_navigation/multi_terrain_test9/play_test9.sh \
  --layout rocks_gravel__right_gravel
```

Il play crea esplicitamente la camera della finestra grafica prima che
Isaac Lab la posizioni, per gestire l'errore di avvio `Accessed invalid null
prim` su `/OmniverseKit_Persp`. La camera RGB-D del robot resta quella della
configurazione di training.

Sostituire `NOME_DELLA_RUN` con la cartella prodotta dal training:

```bash
python3 spot_rl_energy_navigation/multi_terrain_test9/test_known_maps.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test9/runs/NOME_DELLA_RUN/policy_final.zip \
  --episodes 5
```

Esegue 5 episodi per ciascuna mappa, con finestra grafica. Accetta anche
`--headless` e `--dry-run`. Si può scegliere `policy_best.zip` nello stesso
comando. I risultati vengono salvati sotto `results/` di test9.

## Isolamento e verifiche

Tutto il codice dell'esperimento è locale: nessun import da test8 o altri
esperimenti, nessun vecchio log/checkpoint copiato. Come test8, legge gli
asset esistenti del robot e della locomozione in `IsaacRobotics/`, le due
mappe in `terrain_generator/multi_terrain_binary_test2/`, le ottiche ZED
installate e i pesi DINO dalla cache. Questi asset non vengono modificati.
Restano necessarie le librerie già installate per test8.

Controlli CPU senza simulazione o caricamento DINO:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  spot_rl_energy_navigation/multi_terrain_test9/test_offline.py
```

I controlli includono un piccolo aggiornamento SAC su dati sintetici,
osservazioni, reward, campionamento, isolamento e salvataggio checkpoint.
Non misurano il comportamento del robot.

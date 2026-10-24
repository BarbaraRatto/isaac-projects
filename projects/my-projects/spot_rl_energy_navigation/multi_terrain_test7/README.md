# Multi terrain test7 — peso energetico 1,1

Esperimento indipendente con la stessa configurazione di training di test6,
salvo due cambiamenti richiesti:

| Impostazione | Test6 | Test7 |
| --- | ---: | ---: |
| Penalità per joule | 0,02 | **1,1** |
| Transizioni totali predefinite | 409.600 | **100.000** |

Il training parte **da zero**, come test6: non carica policy o replay degli
esperimenti precedenti. Il nuovo coefficiente è 55 volte quello precedente;
un risparmio di 80 J corrisponde a 88 punti di reward in più.

## Configurazione mantenuta

- SAC con entropia automatica, buffer da 100.000 transizioni (circa 5,75 GiB
  di RAM), batch 256, learning rate 0,0003, gamma 0,999 e tau 0,005.
- 4.096 transizioni iniziali casuali; un aggiornamento SAC per passo
  vettoriale di 32 robot; SAC su CPU con quattro thread per default.
- Una sola policy condivisa da 32 Spot: 16 per ciascuna delle due mappe
  `rocks_gravel__right_rocks` e `rocks_gravel__right_gravel`.
- Mappe 3×3 con celle da 2×2 m, corridoi rocce/ghiaia scambiati, asfalto
  a partenza e goal, ghiaccio centrale. Start `(0,0)`, goal `(4,4)` m,
  yaw iniziale 45°, jitter di reset ±0,05 m.
- RGB-D e DINOv2-small congelato, feature originali a 384 componenti,
  senza PCA o atlante. Stessi 20 campioni frontali, massimo 18 validi;
  nessuna memoria tra frame e nessuna supervisione energetica ausiliaria.
- Input totale di 7.711 valori: 11 di stato/goal e 7.700 visivi.
- Ramo visivo `7700 → 2048 → 1024`, ramo stato `11 → 128 → 128`, fusione
  `1152 → 512 → 256`, attivazioni ELU. Actor e critic hanno encoder
  separati; SAC usa due Q-head e un critic target.
- Solo avanzamento `[0,1,6] m/s` e rotazione `±1,5 rad/s`; comando laterale
  sempre zero, nessun comando di retromarcia. Le azioni esposte a SAC sono
  entrambe in `[-1,1]`: `vx=(a0+1)/2*1,6`, `wz=a1*1,5`.
  **L'arresto corrisponde a `(-1,0)`; `(0,0)` avanza a 0,8 m/s.**
- Fisica a 200 Hz, decisioni ogni 0,2 s, massimo 40 s per episodio e
  successo entro 0,55 m dal goal. Assestamento iniziale di 2 s escluso
  dall'energia dell'episodio. Controllore locomotorio congelato.

## Reward

```text
3 × (distanza_precedente − distanza_attuale)
− 1,1 × energia_meccanica_J_dello_step
− 0,01
+ 60 all'arrivo
− 25 per caduta o uscita dai limiti
```

Il timeout non aggiunge una penalità terminale. L'energia è l'integrale della
somma di `abs(coppia_misurata × velocità_giunto)` sui passi fisici, calcolato
con il metodo dei trapezi. Non è una misura del consumo elettrico e non
entra nell'osservazione della policy.

Il cambio del coefficiente è separato da modifiche di rete, algoritmo,
percezione o geometria. La durata è però più breve di test6: per confrontare
i risultati, considerare anche il checkpoint test6 a un budget simile.

## Isolamento

Ambiente, configurazioni, estrattore DINO, locomozione, scena, rete, training,
valutazione e controlli sono implementati in questa cartella. Non vengono
importati moduli dagli esperimenti precedenti. Non sono copiati vecchi
checkpoint, log o risultati.

Restano dipendenze esterne in sola lettura:

- librerie Isaac Lab, Isaac Sim, Stable Baselines3, PyTorch e Transformers;
- robot USD e pesi locomotori sotto `IsaacRobotics/`;
- ottiche ZED X installate;
- i due USD originali sotto `terrain_generator/multi_terrain_binary_test2/`,
  elencati in `layouts.py`, con hash registrati per ogni run;
- pesi DINO nella normale cache Hugging Face.

I file degli altri esperimenti e gli asset originali non vengono modificati.

## Comandi da my-projects

Il simulatore e il training vengono lanciati dall'utente.

Probe tecnico con finestra grafica, senza addestramento o checkpoint:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test7/train.py --probe-steps 8
```

Il probe inizia e termina con comando di arresto; nelle azioni intermedie
avanza lentamente con rotazioni alternate. Aggiungere `--headless` per
eseguirlo senza finestra.

Training, con le 100.000 transizioni predefinite:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test7/train.py --headless --timesteps 100000
```

100.000 è esattamente un multiplo di 32: sono 3.125 passi vettoriali, cioè
3.125 transizioni per robot. Eventuali altri budget devono essere multipli
positivi di 32. Le valutazioni non contano nel budget del training.

Opzioni aggiuntive: `--seed`, `--dino-batch-size`, `--agent-device`,
`--torch-threads`, `--buffer-size`, `--gradient-steps`, `--eval-every` e
`--checkpoint-every`. `--device` controlla il simulatore, `--agent-device`
controlla SAC. `--eval-every 0` disabilita le valutazioni automatiche.

## Valutazioni e output

Valutazione deterministica a 51.200 transizioni e a fine training, un
episodio per robot: 16 per mappa. Riutilizza il simulatore tra due raccolte,
senza aggiungere le transizioni di valutazione al replay. Il seed di
valutazione è `seed_training + 10000`.

Gli episodi di training ancora aperti vengono resettati dopo la valutazione;
soltanto gli episodi completati entrano nel CSV degli episodi. La selezione
del checkpoint migliore è la stessa di test6: successo sulla mappa peggiore,
poi successo medio, poi energia dei successi. Con zero arrivi usa la
distanza finale come spareggio. Si salvano anche la policy finale e tutti
i risultati di valutazione, indipendentemente dalla selezione del migliore.

Ogni training crea `runs/test7_sac_32env_<timestamp>/` con configurazione,
manifest/hash dei terreni, copia dei sorgenti locali, architettura,
`episodes.csv`, `timing.csv`, metriche SAC in `sb3/progress.csv`,
`evaluations.csv`, `best_evaluation.json`, checkpoint e `summary.json`.
Checkpoint periodico a 51.200 transizioni; alla fine `policy_final.zip`.
Ctrl+C durante l'apprendimento salva una policy interrotta. Il replay
rimane soltanto in RAM; lo script non implementa la ripresa del training.

Per testare un checkpoint test7 sulle due mappe note, 20 episodi per mappa:

```bash
TEST7_CHECKPOINT="spot_rl_energy_navigation/multi_terrain_test7/runs/NOME_DELLA_RUN/policy_best.zip"
python3 spot_rl_energy_navigation/multi_terrain_test7/test_known_maps.py --checkpoint "$TEST7_CHECKPOINT"
```

Il launcher esegue due play in sequenza con finestra grafica e seed 20000.
Accetta anche `--headless`, `--episodes N`, `--seed N` e `--dry-run`.
Usare `policy_final.zip` nello stesso comando per valutare la policy finale.

Per una sola mappa:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test7/play.py --checkpoint "$TEST7_CHECKPOINT" --layout rocks_gravel__right_rocks --episodes 20
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test7/play.py --checkpoint "$TEST7_CHECKPOINT" --layout rocks_gravel__right_gravel --episodes 20
```

I risultati e le traiettorie finiscono in `results/` locale; un eventuale
`--output` deve restare in questa cartella. Il play verifica firma del
checkpoint e hash dei terreni: i checkpoint test6 sono incompatibili con
questa nuova configurazione di reward.

## Controlli offline

```bash
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/multi_terrain_test7/test_offline.py
```

Verificano feature raw, campionamento, comandi, reward con peso 1,1,
budget di 100.000 transizioni, isolamento degli import, due rami, gradienti,
aggiornamenti SAC con dati sintetici, replay, valutazioni e checkpoint.
Non avviano il simulatore e non misurano le prestazioni di navigazione.

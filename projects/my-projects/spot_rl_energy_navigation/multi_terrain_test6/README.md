# Multi terrain test6 — SAC con rete a due rami

Nuova policy di navigazione allenata da zero con **SAC**, senza caricare
checkpoint dei test precedenti. L'esperimento usa ancora i **20 campioni
frontali**, non la mappa completa. Non introduce memoria tra frame o una
supervisione esplicita del costo energetico per cella.

Tutto il codice specifico è locale: ambiente, configurazioni, scena, geometria,
proiezione RGB-D, estrattore DINO, controllore locomotorio, rete, training,
valutazione e test. Non importa codice da test2/3/4/5 o dai vecchi esperimenti.
Le librerie Isaac Lab/SB3/PyTorch e gli asset esistenti restano dipendenze
esterne in lettura: non sono copiati o modificati.

## Mappe e ambiente

| Impostazione | Valore |
| --- | --- |
| Robot / policy | 32 Spot, una sola policy SAC condivisa |
| Layout 1 | 16 robot, `rocks_gravel__right_rocks` |
| Layout 2 | 16 robot, `rocks_gravel__right_gravel` (speculare) |
| Terreni | Due corridoi rocce/ghiaia, asfalto ai due estremi e ghiaccio centrale |
| Griglia fisica | 3×3 celle da 2×2 m; due USD esistenti di test4/test5 |
| Partenza → goal | `(0,0)` → `(4,4)` m; jitter iniziale ±0,05 m |
| Yaw iniziale | 45° verso il goal |
| Fisica / azioni RL | 200 Hz / ogni 0,2 s |
| Episodio / successo | Massimo 40 s; distanza dal goal ≤0,55 m |
| Assestamento | 2 s iniziali, esclusi dall'energia dell'episodio |
| Percezione | RGB-D 640×360, ottiche ZED X CameraRight, inclinazione 15° |
| DINO | `facebook/dinov2-small` congelato, immagini live, input 518×518, microbatch 4 |
| Feature | 384 componenti originali, mediate per cella; nessuna PCA o atlante |

La griglia delle feature è 12×12 con risoluzione 0,5 m. Viene ricostruita a
ogni osservazione; non accumula le zone viste in precedenza. I campioni sono
avanti `(0,5; 1; 1,5; 2)` m e lateralmente `(-1; -0,5; 0; 0,5; 1)` m. Il
filtro ±50° esclude due posizioni; al massimo 18 campioni sono validi.

Ogni campione contiene 384 feature più un flag. L'input mantiene l'ordine:
**11 valori di stato/goal + 7.700 valori visivi = 7.711**. I valori di stato
sono goal relativo (2), velocità lineare nel corpo (2), velocità angolare (1),
gravità nel corpo (3), comando precedente normalizzato (3). Il comando
laterale precedente è sempre zero. Le feature raw non hanno normalizzazione
PCA, clipping o normalizzazione statistica aggiuntiva.

Reward e misura energetica restano quelle di test5:

```text
3 × progresso_m − 0,02 × energia_J − 0,01
+ 60 all'arrivo − 25 per caduta o uscita dai limiti
```

La potenza è `sum(abs(coppia_misurata × velocità_giunto))`, integrata con il
metodo dei trapezi. L'energia non è un input della rete.

## Rete

```text
visione: 7700 → 2048 → 1024 ──┐
                             ├─ concatenazione 1152 → 512 → 256 → actor
stato:      11 → 128 → 128 ───┘
```

Attivazioni **ELU**. L'actor produce media e log-deviazione delle due azioni,
poi SAC applica la trasformazione tanh. Il critic ha un proprio encoder a
due rami, distinto dall'actor; i suoi due Q-head ricevono anche le due azioni:
`1154 → 512 → 256 → 1`, ciascuno con pesi propri. Esiste anche un critic
target separato, aggiornato gradualmente da SAC. DINO resta congelato;
encoder e head della policy sono appresi dal ritorno RL.

## Azioni

Entrambe le azioni esposte a SAC sono in `[-1,1]`:

```text
vx = (azione_0 + 1) / 2 × 1,6 m/s
vy = 0
wz = azione_1 × 1,5 rad/s
```

**Attenzione alla convenzione:** `(-1,0)` significa arresto; `(0,0)` significa
avanzamento a 0,8 m/s. La prima azione negativa non comanda retromarcia.
Sono possibili rotazione sul posto e avanzamento in curva. Il vincolo
riguarda i comandi; il robot può comunque scivolare fisicamente.

## SAC e valutazione

| Parametro | Default |
| --- | --- |
| Transizioni di training | 409.600 totali sui 32 robot |
| Replay buffer | 100.000 transizioni, circa **5,75 GiB** di RAM |
| Raccolta iniziale casuale | 4.096 transizioni |
| Batch / learning rate | 256 / 0,0003 |
| Gamma / tau | 0,999 / 0,005 |
| Entropia | Coefficiente automatico; target automatico |
| Aggiornamenti | 1 minibatch SAC dopo ogni passo vettoriale di 32 robot |
| Device SAC | CPU con 4 thread, separato dal device del simulatore |
| Checkpoint | Ogni 51.200 transizioni |
| Valutazione | Ogni 51.200 transizioni e dopo l'ultimo aggiornamento |

`--gradient-steps` conta aggiornamenti **per passo vettoriale**, non per
singolo robot. L'impostazione iniziale evita di moltiplicare automaticamente
per 32 il costo degli aggiornamenti della rete grande. Non è una ricerca
degli iperparametri né una garanzia di migliori risultati rispetto a PPO.

La valutazione periodica è deterministica, con un episodio per robot:
16 episodi per mappa. Riutilizza il simulatore tra due raccolte di training,
senza creare un secondo Isaac Sim. Le azioni di valutazione non entrano nel
replay e non contano nelle transizioni di training. Dopo la valutazione i
robot vengono resettati e l'osservazione interna di SAC viene sincronizzata;
gli episodi di training allora ancora aperti vengono interrotti e non
appaiono tra gli episodi completati di `episodes.csv`.

La selezione di `policy_best.zip` privilegia nell'ordine: successo sulla
mappa peggiore, successo medio sulle due mappe, energia dei successi. Se non
ci sono successi usa la distanza finale come spareggio. Le mappe di questa
valutazione sono viste durante il training: non è una prova di generalizzazione.

Il replay rimane in RAM e non viene salvato nei checkpoint. I checkpoint
servono alla valutazione; lo script non implementa la ripresa del training.

## Comandi da my-projects

Il simulatore e il training vengono lanciati dall'utente. Probe con finestra
grafica: crea tutti i robot e verifica scena, osservazioni e comandi, senza
allenare o salvare una policy.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test6/train.py --probe-steps 8
```

Il probe inizia e termina fermo; nei passi intermedi avanza a 0,32 m/s con
rotazioni alternate. Aggiungere `--headless` per il probe senza finestra.

Training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test6/train.py --headless --timesteps 409600
```

Opzioni disponibili: `--buffer-size` (multiplo di 32, almeno 256),
`--gradient-steps`, `--eval-every` (0 disabilita le valutazioni),
`--checkpoint-every`, `--torch-threads`, `--agent-device`,
`--dino-batch-size` e `--seed`. `--device` controlla il simulatore;
`--agent-device` controlla SAC. I timesteps devono essere multipli di 32.
Cambiare device SAC richiede memoria sufficiente insieme a rendering e DINO.

Per vedere la policy migliore, sostituire `NOME_DELLA_RUN` con il nome reale:

```bash
TEST6_CHECKPOINT="spot_rl_energy_navigation/multi_terrain_test6/runs/NOME_DELLA_RUN/policy_best.zip"

/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test6/play.py --checkpoint "$TEST6_CHECKPOINT" --layout rocks_gravel__right_rocks --episodes 5

/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test6/play.py --checkpoint "$TEST6_CHECKPOINT" --layout rocks_gravel__right_gravel --episodes 5
```

`policy_final.zip` è utilizzabile nello stesso modo. Il play verifica firma
dell'esperimento e hash dei terreni, e mostra un solo Spot.

### Test delle due mappe note: 20 episodi per mappa

Per la run completata `test6_sac_32env_20261019_235423_917318`, il launcher
seguente usa direttamente `policy_best.zip` ed esegue i due play in sequenza:

```bash
python3 spot_rl_energy_navigation/multi_terrain_test6/test_known_maps.py
```

Sono **40 episodi complessivi**, 20 per ciascuna mappa, con finestra grafica
e seed 20000 (diverso dal seed 10000 delle valutazioni automatiche durante
il training). La prima simulazione termina e poi viene aperta la seconda.
Non avvia addestramento. Tutti i CSV, le traiettorie, la configurazione del
test e un riepilogo JSON finiscono in una nuova sottocartella
`results/known_maps_<timestamp>/`; al termine stampa anche una tabella.

Opzioni: `--headless`, `--checkpoint PERCORSO`, `--episodes N` per mappa,
`--seed N`, `--agent-device DEVICE`, `--torch-threads N`. Con `--dry-run`
stampa i due comandi senza avviare Isaac Sim o creare risultati.

### Due mappe nuove: lanci separati

Due configurazioni mai inserite in `training_conditions()`, con rocce e
ghiaia mescolate sui due corridoi. `A`=asfalto, `R`=rocce, `G`=ghiaia,
`I`=ghiaccio. Le righe seguenti corrispondono nell'ordine a y=0, 2, 4 m:

```text
Mappa nuova 1        Mappa nuova 2
A R G                A G R
G I G                R I R
R R A                G G A
```

Partenza `(0,0)` e goal `(4,4)` invariati. I due USD sono in `maps/heldout/`.
Sono composti traslando le nove celle già presenti nel USD di training:
stessi mesh, materiali, texture, attriti e scala. Ciascun donor viene usato
una sola volta per mappa; la seconda disposizione traspone la prima.
Non viene presupposto quale dei due corridoi abbia meno consumo totale.

**Mappa nuova 1**, 20 episodi con `policy_best.zip` della run completata:

```bash
python3 spot_rl_energy_navigation/multi_terrain_test6/test_new_map_1.py
```

**Mappa nuova 2**, altri 20 episodi:

```bash
python3 spot_rl_energy_navigation/multi_terrain_test6/test_new_map_2.py
```

Entrambi aprono la finestra grafica, usano seed 20000 e non allenano il
modello. Accettano le stesse opzioni del launcher delle mappe note, incluso
`--dry-run`. Gli output sono separati sotto `results/heldout_<layout>_<timestamp>/`:
episodi, traiettoria, griglia, metadati/hash della mappa e riepilogo JSON.

`heldout_layouts.py` definisce soltanto condizioni di valutazione;
`layouts.py`, `training_conditions()`, gli hash delle mappe originali e la
firma del checkpoint non vengono modificati. `play.py` continua a verificare
il checkpoint rispetto alla sua configurazione di training, poi valida
separatamente la mappa di valutazione.

`build_heldout_maps.py` genera e valida i nuovi USD usando solo pxr, senza
avviare il simulatore. Se un USD manca viene creato al primo play; i file
originali non vengono sovrascritti. È eseguibile anche separatamente:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh spot_rl_energy_navigation/multi_terrain_test6/build_heldout_maps.py
```

## Output e dipendenze

I training scrivono solo in `runs/test6_sac_32env_<timestamp>/`: configurazione,
manifest, hash dei terreni, copia dei sorgenti locali, architettura,
`episodes.csv`, `timing.csv`, metriche SAC in `sb3/progress.csv`,
`evaluations.csv`, `best_evaluation.json`, checkpoint e `summary.json`.
Ctrl+C durante l'apprendimento salva un checkpoint interrotto.

Il play salva risultati, traiettorie con i comandi e griglia sotto `results/`.
Un eventuale `--output` deve rimanere dentro questa cartella.

Gli asset esterni in sola lettura sono elencati in `assets.py` e `layouts.py`:
robot e pesi locomotori sotto `IsaacRobotics/`, ottiche ZED X installate,
e le due mappe USD sotto `terrain_generator/multi_terrain_binary_test2/`.
I pesi DINO usano la normale cache Hugging Face, come nei test precedenti.
Non serve un atlante PCA o un checkpoint precedente.

## Controlli offline

```bash
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/multi_terrain_test6/test_offline.py
```

I test non avviano Isaac Sim e non caricano DINO: verificano feature, azioni,
isolamento degli import, reward, architettura, gradienti su entrambi i rami,
due aggiornamenti SAC con dati sintetici, replay, valutazione e roundtrip
del checkpoint. Non misurano la capacità di navigazione della policy.

# Multi terrain test5

Esperimento isolato di navigazione energetica: mantiene le due mappe e la
fisica di `multi_terrain_binary_test4`, usa feature DINOv2 originali da 384
componenti e permette soltanto avanzamento e rotazione. La policy di
navigazione parte da zero; la policy locomotoria IsaacRobotics resta congelata.

## Configurazione

| Parametro | Test5 |
| --- | --- |
| Ambienti | 32 Spot, una sola policy PPO condivisa |
| Mappe | 16 `rocks_gravel__right_rocks`, 16 `rocks_gravel__right_gravel` |
| Partenza / goal | `(0, 0)` → `(4, 4)` m; jitter iniziale ±0,05 m |
| Orientamento iniziale | 45° verso il goal |
| Camera | RGB-D 640×360, ottiche ZED X CameraRight, inclinazione 15° |
| Percezione | DINOv2-small congelato, RGB corrente, input 518×518, microbatch 4 |
| Griglia intermedia | 12×12 celle da 0,5 m, ricostruita dal frame corrente |
| Campioni della policy | Stesse 20 posizioni frontali di test4 |
| Feature | Media delle feature DINO delle patch nella cella; 384 componenti originali |
| PCA / atlante | Non utilizzati né richiesti |
| Osservazione | 7.711 valori: 11 di stato + 20×(384 feature + validità) |
| Azioni PPO | Due: frazione di avanzamento `[0,1]`, frazione di rotazione `[-1,1]` |
| Comandi locomotori | `vx ∈ [0,1,6] m/s`, `vy = 0`, `wz ∈ [-1,5,+1,5] rad/s` |
| Actor | `7711 → 512 → 256 → 128 → 2`, Tanh negli strati nascosti |
| Critic | `7711 → 512 → 256 → 128 → 1`, Tanh negli strati nascosti |
| Fisica / decisione RL | 200 Hz / ogni 0,2 s (40 passi fisici) |
| Episodio | Massimo 40 s; arrivo entro 0,55 m |
| Assestamento iniziale | 2 s, esclusi dall'energia dell'episodio |
| Training predefinito | 409.600 transizioni totali sui 32 robot |
| PPO | Rollout 32 passi/robot, batch 256, 5 epoche, LR 0,0003, gamma 0,999; CPU |

La reward è invariata rispetto a test4:

```text
3 × progresso_m − 0,02 × energia_J − 0,01
+ 60 all'arrivo − 25 per caduta o uscita dai limiti
```

L'energia è lavoro meccanico: somma di `abs(coppia × velocità del giunto)`,
integrata con il metodo dei trapezi sui passi fisici. Non entra nelle
osservazioni e non viene assegnato un costo supervisionato alle celle.

Gli 11 valori di stato restano quelli di test4: goal relativo (2), velocità
lineare misurata avanti/laterale (2), velocità angolare (1), gravità nel corpo
(3), comando precedente normalizzato (3). Il comando laterale precedente è
sempre zero; la velocità laterale misurata rimane osservabile.

Le posizioni campionate sono avanti `(0,5; 1; 1,5; 2)` m e lateralmente
`(-1; -0,5; 0; 0,5; 1)` m. Il filtro angolare ±50° esclude sempre due
posizioni: al massimo 18 dei 20 campioni sono validi. I campioni invalidi
hanno feature e flag pari a zero. Non si accumula memoria fra frame.

Non vengono applicati PCA, normalizzazione PCA o clipping delle feature:
restano le 384 componenti prodotte da DINO, mediate per cella. La normale
preelaborazione dell'immagine prevista da DINO resta attiva.

Il comando `(0,0)` ferma la navigazione; `(0,wz)` permette la rotazione sul
posto. Non sono consentiti comandi di retromarcia o traslazione laterale.
Il vincolo riguarda il comando: scivolamenti fisici rimangono possibili.

## Isolamento e file

Tutto il codice specifico, le configurazioni e gli output dell'esperimento
sono in questa cartella. I file esistenti non vengono modificati. Si
riutilizzano in lettura il controllore locomotorio congelato, alcuni tipi e
parametri condivisi e gli asset già presenti nel progetto:

- `IsaacRobotics/assets/spot.usd` e pesi della locomozione;
- i due USD sotto `terrain_generator/multi_terrain_binary_test2/`;
- le ottiche ZED X e l'estrattore DINO già usati dal progetto.

Non serve `runs/ice_choice_atlas.npz`. Non si possono caricare checkpoint
test2/3/4: osservazioni, azioni e rete sono differenti.

| File | Ruolo |
| --- | --- |
| `config.py` | Configurazione locale, dimensioni, rete e firma dell'esperimento |
| `layouts.py` | Mappe e assegnazione 16/16 con coordinate centrate |
| `scene.py` | Robot, terreni, sensori, fisica e spazi Gymnasium |
| `env.py` | Energia, reward, azioni, osservazioni, reset e fine episodio |
| `actions.py` | Conversione delle due azioni nei tre comandi del controllore |
| `vision.py`, `visual_features.py` | DINO online, proiezione depth e campioni raw |
| `train.py` | Probe tecnico, PPO da zero, checkpoint e log |
| `play.py` | Valutazione deterministica di un checkpoint test5 |
| `test_offline.py` | Controlli CPU senza simulatore né caricamento dei pesi DINO |

## Comandi

Eseguire dalla directory `my-projects`. Il simulatore e il training vengono
lanciati dall'utente. La prima esecuzione può usare la cache standard del
modello DINO, come gli esperimenti precedenti.

Probe tecnico: crea i 32 ambienti, controlla osservazioni e comandi, fa otto
azioni senza allenare o salvare policy. Inizia e termina con comando zero;
nelle azioni intermedie avanza lentamente con rotazioni alternate.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test5/train.py --headless --probe-steps 8
```

Togliere `--headless` per vedere la scena.

Training completo:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test5/train.py --headless --timesteps 409600
```

`--timesteps` deve essere un multiplo positivo di 1.024. È possibile usare
`--dino-batch-size 1` per ridurre il microbatch DINO, senza ridurre i 32 robot.

Il training scrive una directory `runs/test5_raw384_32env_<timestamp>/` con:

- `config.json`, manifest delle mappe, hash dei USD e firma dell'esperimento;
- copia dei sorgenti locali in `sources/` e `policy_architecture.txt`;
- `episodes.csv` e `timing.csv`;
- checkpoint ogni 10.240 transizioni e `policy_final.zip`;
- `summary.json` con tempi, successi e statistiche per condizione.

Ctrl+C durante l'apprendimento salva `policy_interrupted_<N>_steps.zip`.

Per la valutazione, impostare il percorso stampato alla fine del training:

```bash
TEST5_CHECKPOINT="spot_rl_energy_navigation/multi_terrain_test5/runs/NOME_DELLA_RUN/policy_final.zip"

/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test5/play.py --checkpoint "$TEST5_CHECKPOINT" --layout rocks_gravel__right_rocks --episodes 5

/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_test5/play.py --checkpoint "$TEST5_CHECKPOINT" --layout rocks_gravel__right_gravel --episodes 5
```

Ogni play mostra un solo Spot, verifica configurazione e hash dei terreni,
e salva CSV di episodi, traiettoria e griglia in `results/`. La traiettoria
include anche i comandi `vx`, `vy`, `wz`. Un eventuale `--output` deve indicare
un percorso dentro questa cartella.

## Verifica offline

```bash
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/multi_terrain_test5/test_offline.py
```

Questi controlli verificano limiti delle azioni, feature raw senza PCA,
campionamento e validità, assenza di memoria tra frame, equivalenza delle
mappe/reward con test4, microbatch DINO con interfacce simulate, e dimensioni
e salvataggio/caricamento della MLP. Non verificano il rendering, i contatti
fisici o l'effettivo apprendimento: per quelli serve il probe in Isaac Sim
e successivamente il training.

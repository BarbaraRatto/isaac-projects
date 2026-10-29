# Navigazione su mappa grande — test 6 SAC

Training da zero di **100.000 transizioni di navigazione**, 32 ambienti separati. Cartella autonoma: `train_general_big_map_test6_SAC`. Mappa `real_terrains_v2.usd`, banca missioni, PCA sui sette materiali, DINO, controllore locomotorio, input, allineamento, reward e curriculum provengono dal test 5.

## Comandi

Dalla cartella principale del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test6_SAC/train.sh
```

Il launcher controlla e riusa la PCA copiata, quindi avvia SAC headless, simulazione/DINO su `cuda:0` e agente su CPU (4 thread). Il controllo PCA con `--reuse` non avvia il simulatore se l'artefatto è già valido. Supportati `GENERAL_MAP_ISAACLAB` e `../cloud/env.sh`. Un percorso PCA alternativo viene inoltrato alla preparazione e al training.

Monitoraggio da un altro terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test6_SAC/status.py
```

Test visibile dell'ultimo modello finale:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test6_SAC/test.sh
```

Per un checkpoint specifico, aggiungere `--checkpoint /percorso/policy_....pt`. Il test usa un robot, nove missioni, seed 20000 e per default la fase dell'ultima valutazione del checkpoint. `--stage 1` impone la fase 1 per confrontarla con i test precedenti. Salva `episodes.csv`, `trajectories.csv`, `fall_diagnostics.csv`, `summary.json` in una nuova cartella `results/evaluation_*`. In valutazione le azioni sono deterministiche e nessun campione aggiorna replay, normalizzazione o rete.

## Algoritmo e rete

SAC implementato in PyTorch, senza dipendenza da Stable Baselines3. Actor asimmetrico con distribuzione gaussiana dipendente dallo stato, azioni limitate tramite tanh. Due Q indipendenti stimano la stessa reward totale; le corrispondenti copie target vengono aggiornate gradualmente. I due Q non separano navigazione ed energia.

- Actor: stato 15 → 64 → 32; gridmap 3240 → 1024 → 512 → 256 → 128; concatenazione 160 → 128 → 64 → 4 (due medie e due log-deviazioni). Output operativo: due comandi normalizzati, convertiti in vx ∈ [0,1.6] m/s e wz ∈ [-1.5,1.5] rad/s.
- Ogni Q: ramo stato con 15 input actor + 24 privilegiati + 2 azioni → 64 → 32; ramo mappa identico all'actor; concatenazione → 128 → 64 → 1. I Q non condividono parametri con l'actor né tra loro. Attivazioni ELU come test 5.
- Learning rate: 3e-4 per actor, critic e temperatura. Gamma 0.999; tau 0.005; batch 256.
- Alpha automatico, iniziale 1.0; target entropy -2 per le due azioni normalizzate. Entropia misurata sulla distribuzione dopo tanh.
- Warmup: azioni casuali uniformi fino a 4096 transizioni di navigazione. Poi **1 aggiornamento SAC per passo parallelo con almeno un campione di navigazione accettato**. Un aggiornamento comprende critic, actor, alpha e aggiornamento delle reti target; non è un aggiornamento per ciascuno dei 32 robot. Gli ultimi passi possono includere meno di 32 campioni.
- Nessun rollout/GAE/clipping PPO. Normalizzazione delle osservazioni come nel test 5: replay conserva dati raw; osservazioni attuali e successive vengono normalizzate al momento del campionamento. Statistiche aggiornate una volta sui soli nuovi campioni di training, mai sui batch ricampionati o in valutazione.

## Replay e conteggio

Capacità di 100.000 transizioni individuali in float32 su RAM CPU, circa **2,44 GiB** (esclusi modelli, ottimizzatori, DINO e simulazione). Ogni transizione include osservazioni attuali e successive, privilegi, azione effettivamente applicata, reward e terminale. La componente actor non è duplicata nello storage critic. I campioni vecchi restano nel buffer dopo i cambi di fase; vengono ricampionati uniformemente.

Solo i passi comandati dalla rete entrano nel replay e nel budget; l'allineamento autonomo resta escluso. Goal, caduta, uscita mappa, timeout e fallimento dell'allineamento mantengono la semantica terminale del test 5: il target non usa il valore del nuovo episodio dopo un reset automatico. Le ultime righe eccedenti sono scartate per raggiungere esattamente 100.000 transizioni, senza obbligo di multipli di 32.

Il replay viene salvato su disco una volta a fine training o dopo Ctrl+C (fino a circa 2,44 GiB aggiuntivi). I checkpoint periodici contengono reti, target, ottimizzatori, alpha, normalizzazione, PCA, RNG e stato curriculum; il replay abbinato è disponibile solo per `policy_final.pt` o `policy_interrupted.pt`.

Per continuare in futuro da uno di questi due checkpoint e dal relativo `replay_<transizioni>.npz`, senza cambiare la configurazione SAC:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test6_SAC/train.sh \
  --resume /percorso/policy_final.pt --timesteps 200000
```

`--timesteps` indica il totale cumulativo. I checkpoint PPO non sono compatibili. Il simulatore riparte con nuovi episodi: non si ripristina la posa fisica esatta. La cartella del nuovo training è distinta da quella di origine.

## Task, reward e curriculum

Per ogni passo di navigazione:

`r = 30*(d_precedente-d_attuale) - 0.02 - penalità_orientamento + 1000*successo - 1000*fallimento - energia_J/100`

Peso energetico 1. La penalità di orientamento è zero fino a 110° e cresce linearmente fino a 2 punti/s a 180°; si moltiplica per dt=0.2 s. Reward e costi mantengono la gestione dell'allineamento del test 5. Energia meccanica integrata sui 40 passi fisici da 0.005 s; totali di missione includono anche l'allineamento.

Mappa congelata del test 5, rocce della funzione originale. Griglia robocentrica persistente 9×9, PCA32 + geometria + validità/età; stato actor di 15 valori. Critic con privilegi dei quattro piedi.

Caduta: inclinazione ≥90° oppure contatto corpo/terreno >1 N, ciascuno per almeno 1 s consecutivo, controllo ogni 0.005 s. Altezza minima e altezza non valida solo diagnostiche. Limiti di velocità, deadline, allineamento e reset invariati.

Cinque fasi, numerate 0–4: asfalto 1.5–3 m; endpoint asfalto/ghiaia/rampe/rocce 3–6 m; tutti gli endpoint ammessi 3–6 m; 6–12 m; 12–20 m. Esclusi ghiaccio dallo spawn e ghiaccio/scale dal goal; i percorsi non sono vincolati a questi endpoint. Replay di missioni più facili 25%, come test 5.

Promozione ≥80% di successi in due valutazioni consecutive su 32 episodi, seed fisso 710031 per fase. Nessuna retrocessione. Valutazione e checkpoint ogni circa 10.000 transizioni più valutazione finale. `policy_best_stage_*.pt` conserva il migliore di ogni fase; `policy_final.pt` conserva l'ultimo.

## Verifica

```bash
python3 -m pytest -q spot_rl_energy_navigation/train_general_big_map_test6_SAC
```

Test offline per SAC, target terminali/entropia, aggiornamento target, asimmetria, replay/reset/allineamento, salvataggio/ripristino, export, geometria, energia, cadute, curriculum e launcher. Questi controlli non sostituiscono l'avvio reale in Isaac Sim; durante la preparazione non viene avviata alcuna simulazione.

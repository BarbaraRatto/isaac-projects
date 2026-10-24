# Navigazione su mappa grande — test 2

Esperimento diagnostico da zero, con **100.000 transizioni aggregate**, 32 ambienti separati e missioni A–B indipendenti. La mappa è la copia congelata di `real_terrains_v2.usd` già usata nel test 1: nessuna rigenerazione. Sono inclusi mappa, PCA e banca di punti; robot, controllore locomotorio, camera e DINO usano gli asset locali del progetto.

## Avvio

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test2/train.sh
```

Monitoraggio in un altro terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test2/status.py
```

Il launcher usa Isaac Lab 2.3.0, simulazione/DINO su CUDA, PPO su CPU, modalità headless. Non riprende automaticamente checkpoint di altri esperimenti. Ogni avvio crea una nuova cartella `runs/general_map_test2_ppo_*`.

## Reward e architettura

Per ogni azione di navigazione (0,2 s):

`r = 10*(d_precedente - d_attuale) - 0.02 + 150*successo - 300*fallimento - 1*(energia_step_J/100)`.

Il progresso è espresso in metri. Il coefficiente energetico **1** moltiplica l'energia normalizzata per **100 J**, non i joule grezzi. Il costo integra la somma dei valori assoluti delle potenze meccaniche articolari. Fallimento include caduta, uscita dalla mappa e timeout; successo e fallimento sono mutuamente esclusivi. Raggio goal 0,55 m. Deadline invariata: `clip(15+4*d_iniziale, 30, 90)` secondi, arrotondata al passo di navigazione.

PPO asimmetrico con due teste critic (ritorno navigazione e costo energetico); advantage combinato con peso energetico 1. Actor e critic MLP 512→256→128→64. Actor 3.255 input: stato e 81 celle robot-centriche con PCA32, geometria, validità e anzianità. Critic 3.279 input, con informazioni privilegiate sui contatti. Comandi `vx∈[0,1.6] m/s`, `wz∈[-1.5,1.5] rad/s`. Percezione, memoria della gridmap, normalizzazione, controllore e iperparametri PPO ereditati dal test 1.

## Curriculum

| Fase | Distanza A–B | Errore di orientamento iniziale | Terreni degli endpoint |
|---|---|---|---|
| 0 | 1,5–3 m | ±15° | stessa cella di asfalto |
| 1 | 1,5–3 m | ±45° | stessa cella di asfalto |
| 2 | 1,5–3 m | ±90° | stessa cella di asfalto |
| 3 | 1,5–3 m | ±180° | stessa cella di asfalto |
| 4 | 3–6 m | ±180° | asfalto, ghiaia, rampe, rocce |
| 5 | 3–6 m | ±180° | A: tutto tranne ghiaccio; B: tutto tranne ghiaccio e scale |
| 6 | 6–12 m | ±180° | come fase 5 |
| 7 | 12–20 m | ±180° | come fase 5 |

Lo yaw è campionato rispetto alla direzione A→B, non rispetto agli assi globali. A e B cambiano al reset dell'episodio, indipendentemente per ogni robot. Nelle prime quattro fasi il segmento diretto resta nella stessa cella di asfalto; nelle successive si vincolano gli endpoint, senza imporre un percorso o filtrare la raggiungibilità globale. Rimangono i controlli locali di appoggio della banca originale.

Una fase avanza solo dopo **due valutazioni consecutive con almeno l'80% di successi** (almeno 26/32). Niente avanzamento automatico al trascorrere delle transizioni. Dal livello 1, il 75% delle nuove missioni usa la fase corrente e il 25% una fase precedente scelta uniformemente.

Valutazioni deterministiche ogni 10.000 transizioni, al termine del rollout: prima a 12.288, poi 20.480, 32.768, ecc., più quella finale a 100.000. Ogni valutazione contiene 32 episodi, uno per robot, **solo della fase corrente**; punti riservati alla valutazione e missioni riproducibili per fase. Non entrano nel buffer PPO né nelle statistiche di normalizzazione. Quando cambia la fase, cambia la difficoltà: non confrontare le percentuali come se fossero lo stesso test.

Con rollout 32×128=4.096, il budget comprende 24 rollout completi e uno da 53 passi vettoriali. Le transizioni di valutazione sono aggiuntive e richiedono tempo. **100.000 transizioni servono a verificare l'apprendimento iniziale**, non a completare tutte le otto fasi: con questa frequenza e due valutazioni per promozione non si possono raggiungere tutte entro il budget. Nessuna garanzia di policy generalistica a fine run.

## Risultati e valutazione manuale

`status.py` mostra successi, esiti, ETA, fase e valutazioni. `episodes.csv` contiene A/B, yaw iniziale, distanza, esito, energia e ritorni; `training.csv` le metriche PPO; `evaluation_*.json` i risultati deterministici. Durante le valutazioni si stampa la missione nel terminale.

Checkpoint periodici ogni circa 10.000 transizioni, più `policy_latest.pt`, `policy_final.pt` e `policy_best_stage_N.pt` per confrontare policy sulla stessa difficoltà. `policy_best.pt` privilegia la fase valutata più alta, poi successo ed energia: **non è un confronto su una distribuzione comune fra tutte le fasi**. Curriculum, ultima fase valutata e punteggi per fase sono salvati per la ripresa.

Per vedere una policy, sostituire il percorso del checkpoint:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/train_general_big_map_test2/play.py \
  --checkpoint /percorso/al/run/policy_final.pt \
  --device cuda:0 --episodes 9
```

Di default usa l'ultima fase effettivamente valutata in training. `--stage 0` ... `--stage 7` seleziona esplicitamente un'altra fase. Le valutazioni manuali usano per default un seed diverso da quello delle valutazioni periodiche. I risultati e le traiettorie sono in `results/`.

## Verifiche eseguite

Controllo fisico indipendente prima della preparazione: `../navigation_coherence/results_20261024/report.json`. Sedici casi di trasformazione del goal; rotazione con segno positivo/negativo; avanzamento con quattro orientamenti; raggiungimento di goal davanti, a sinistra, a destra e dietro. Tutti superati sul terreno di asfalto della mappa congelata, con il controllore effettivo. Non è una valutazione della policy RL, della percezione o della percorribilità degli altri terreni.

23 test offline superati: matematica PPO e reward, budget esatto, memoria, campionamento/distanza/angolo/materiali, separazione train/eval, promozioni e ripristino, reset parziali e indipendenza delle valutazioni. Il training del test 2 non è stato avviato.

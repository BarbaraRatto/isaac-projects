# Navigazione su mappa grande — test 3

Training da zero con **300.000 transizioni RL**, 32 scene separate e la mappa congelata `maps/real_terrains_v2.usd`. Include mappa, PCA32, banca dei punti e relativo manifest. Asset Spot, camera, controllore locomotorio e DINO rimangono quelli locali del progetto.

## Avvio e monitoraggio

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test3/train.sh
```

In un secondo terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test3/status.py
```

Il launcher usa Isaac Lab 2.3.0, CUDA per simulazione/DINO e CPU per PPO. Ogni avvio crea un run distinto `runs/general_map_test3_ppo_*`. Il training non è stato lanciato durante la preparazione. Durante l'inizializzazione non è ancora disponibile l'ETA. Non ci sono dipendenze Python dai folder test1/test2; restano gli asset locali condivisi del progetto.

Per continuare un run del test 3:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test3/train.sh \
  --resume /percorso/al/run/policy_final.pt --timesteps 400000
```

`--timesteps` indica il totale desiderato, non le transizioni da aggiungere. Non occorre modificare config.py per estendere il budget. I checkpoint dei test precedenti sono incompatibili con la nuova architettura. Riprendere ripristina reti, entrambi gli ottimizzatori, normalizzazione, RNG e curriculum, ma ricrea gli episodi fisici.

## Allineamento iniziale fisico

A e B vengono campionati al reset, indipendentemente per ogni robot. Lo yaw iniziale è casuale a 360° anche nella prima fase. Il robot parte in assetto eretto; un controllore automatico comanda **vx=0** e **wz=clip(1.5*errore_angolare, -0.8, +0.8) rad/s**.

La policy assume il controllo dal passo successivo a **tre misure consecutive entro 10°** dal goal (tre campioni distanti 0,2 s). Scadenza dell'allineamento: **12 s**, oltre la quale la missione fallisce con esito `alignment_timeout`. Cadute e uscite dalla mappa restano fallimenti. Non ci sono riallineamenti automatici durante la navigazione. Non si teletrasporta l'orientamento verso B.

Il procedimento avviene anche durante le valutazioni/play. Camera, DINO e memoria della gridmap continuano ad aggiornarsi durante la rotazione. Il robot può dunque usare solo osservazioni effettivamente raccolte, mantenute fino al termine della missione.

Ogni ambiente può allinearsi o navigare indipendentemente dagli altri. I reset parziali non fanno avanzare di nascosto la simulazione degli altri robot.

## Conteggio, reward e separazione delle fasi

La misura meccanica integra `sum(abs(tau*qdot))` su tutti i 40 passi fisici da 0,005 s per ciascun comando da 0,2 s. Energia, tempo e percorso dell'allineamento sono inclusi nei **totali della missione**, e registrati anche separatamente dalla navigazione.

Per ogni passo fisico di navigazione/allineamento da 0,2 s il resoconto della missione usa:

```text
r = 10*(d_precedente - d_attuale) - 0.02
    + 150*successo - 300*fallimento - energia_step_J/100
```

Il critic/actor PPO si allena **solo sui passi controllati dalla policy**. Azioni, reward e costi dell'allineamento non sono inseriti nel buffer o nelle ricorsioni GAE: il controllore di partenza non è una decisione appresa dell'actor. Anche i fallimenti prima dell'attivazione dell'actor vengono registrati e contano nei tassi di successo e nella promozione del curriculum.

Il tempo limite della missione è invariato: `clip(15 + 4*d_iniziale, 30, 90)` secondi, **incluso l'allineamento**. Raggio di successo 0,55 m. La policy riceve il tempo residuo effettivo. Un allineamento che consuma più tempo lascia meno tempo alla navigazione.

Il budget è esattamente **300.000 campioni RL utilizzati nel training**; allineamento e valutazioni sono aggiuntivi. Si raccolgono circa 4.096 campioni RL per rollout (fino a 31 in più per terminare un passo parallelo completo). Solo al limite finale del budget si scartano, se necessario, al massimo 31 campioni RL simultanei in eccesso, registrati in `discarded_final_samples`. Non si scartano campioni a ogni rollout. I totali fisici di missione includono tutti i passi eseguiti. Con raccolta asincrona il budget può essere qualsiasi intero positivo.

Ogni transizione conserva identità del robot, terminale e valore del proprio stato successivo. Le GAE non mischiano robot diversi, reset, o azioni del controllore automatico. I terminali interrompono entrambe le teste. Un watchdog interrompe con errore se non viene raccolta alcuna transizione di navigazione per 60 s simulati consecutivi: indica di controllare l'allineamento, invece di continuare indefinitamente.

## Reti

Actor, senza accesso privilegiato:

```text
Stato + goal: 15 → 64 → 32 ──────────────────┐
                                           ├→ 160 → 128 → 64 → 2
Gridmap: 3240 → 1024 → 512 → 256 → 128 ───────┘
```

4.039.906 parametri del solo actor, oltre ai due parametri log-standard-deviation della distribuzione. Attivazioni ELU, uscite Gaussiane con tanh. L'azione normalizzata è convertita in `vx∈[0,1.6] m/s`, `wz∈[-1.5,1.5] rad/s`; zero normalizzato equivale ancora a 0,8 m/s. L'allineamento usa -1 come azione longitudinale per ottenere esattamente zero.

Il critic ha pesi completamente distinti: ramo stato **39→64→32** (15 stato/goal + 24 privilegiati), stesso tipo di ramo mappa e fusione a due uscite: ritorno di navigazione e costo energetico. Le informazioni privilegiate non entrano nell'actor. Normalizzazione con maschere, feature DINO/PCA e memoria spaziale sono conservate. Non c'è LSTM/GRU.

## PPO separato e diagnostica

- Due ottimizzatori Adam separati, learning rate 3e-4, epsilon 1e-5.
- Clipping della norma del gradiente separato per actor e critic, massimo 0,5.
- Stop KL 0,02 **solo per l'actor**; critic fino alle 10 epoche previste sui target fissi del rollout.
- Batch 256, gamma 0,999, GAE lambda 0,95, clip PPO 0,2, entropia 0,005, coefficiente loss 0,5 per ciascuna testa critic.
- Advantage combinato `A_nav - 1*A_energy`, normalizzato dopo la combinazione. Nessuna normalizzazione indipendente che alteri il peso dell'energia.
- Statistiche di normalizzazione ferme durante raccolta e aggiornamenti; si aggiornano dopo il rollout con le osservazioni RL. Solo all'inizio si usa un primo campione fisico per inizializzarle.

`training.csv` registra aggiornamenti effettivi per rete, norme dei gradienti prima del clipping, frequenza di clipping, KL che attiva lo stop, KL finale sull'intero batch, MSE ed explained variance per entrambe le teste prima/dopo gli aggiornamenti. Explained variance non definita per target costanti rimane vuota. Queste misure descrivono il fitting dei target del rollout; non sono da sole una misura della qualità di navigazione. `total_updates`/`minibatch_updates` contano gli aggiornamenti del critic; quelli dell'actor hanno campi distinti.

## Curriculum

| Fase | Distanza | Endpoint ammessi |
|---|---|---|
| 0 | 1,5–3 m | A e B nella stessa cella di asfalto |
| 1 | 3–6 m | Asfalto, ghiaia, rampe, rocce |
| 2 | 3–6 m | A: tutto tranne ghiaccio; B: tutto tranne ghiaccio e scale |
| 3 | 6–12 m | Come fase 2 |
| 4 | 12–20 m | Come fase 2 |

Il controllore allinea fisicamente il robot prima dell'RL in tutte le fasi. Dal livello 1, 75% nuove missioni della fase corrente e 25% di una fase precedente scelta uniformemente. Nessun controllo globale di raggiungibilità; rimangono i controlli locali di appoggio nella banca congelata.

Promozione dopo **due valutazioni consecutive con almeno 80% di successi**. Valutazioni ogni circa 10.000 transizioni RL, al confine del rollout, più quella finale: 32 episodi, uno per robot, su punti riservati alla valutazione. Seed fisso per fase. La percentuale comprende gli eventuali fallimenti dell'allineamento. Una soglia superata non garantisce il completamento dell'intero curriculum entro il budget.

## Checkpoint e test visibile

Sono salvati checkpoint periodici, `policy_latest.pt`, `policy_final.pt`, e `policy_best_stage_N.pt`. `policy_best.pt` privilegia la fase valutata più alta, poi successo ed energia totale: non è un confronto uniforme tra fasi diverse.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/train_general_big_map_test3/play.py \
  --checkpoint /percorso/al/run/policy_final.pt \
  --device cuda:0 --episodes 9 --seed 20000
```

Di default valuta l'ultima fase effettivamente valutata durante il training; `--stage 0` ... `--stage 4` seleziona una fase. Non vengono sovrascritti risultati precedenti. `results/` contiene episodi, summary e traiettorie con start/goal, orientamento, fase, comandi realmente applicati e energia. La posa terminale esatta è nei campi `final_x/final_y` del record episodio; i campioni della traiettoria sono pre-step. `actor.pt` esporta soltanto rete e normalizzazione: un'integrazione esterna deve mantenere anche la macchina di allineamento iniziale, oltre alla percezione e al controllore locomotorio.

## Verifiche

Test offline per PPO/checkpoint/export, isolamento delle informazioni privilegiate, geometria/PCA/memoria, sampling e curriculum, reset parziali, energia su tutti i passi fisici, allineamento/timeout, raccolta asincrona e GAE, budget finale e aggiornamenti critic dopo stop actor. Verificati asset e hash. Nessun training o simulazione del test 3 avviato durante la preparazione: il comportamento fisico completo va verificato al primo avvio.

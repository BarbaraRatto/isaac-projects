# Navigazione su mappa grande — test 4

Training da zero con **300.000 transizioni RL**, 32 scene separate e la mappa congelata `maps/real_terrains_v2.usd`. Include mappa, banca dei punti e relativo manifest. La nuova PCA32 viene calibrata automaticamente al primo avvio sulla mappa attuale. Asset Spot, camera, controllore locomotorio e DINO rimangono quelli locali del progetto.

## Avvio e monitoraggio

Dalla radice del progetto, **un solo comando** per calibrazione e training in sequenza:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test4/train.sh
```

In un secondo terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test4/status.py
```

Al primo avvio il terminale mostra `[PCA]` con avanzamento delle viste, campioni per materiale e stima del tempo residuo di acquisizione. Il normale monitor RL diventa informativo dopo la calibrazione e il primo rollout. La calibrazione viene chiusa prima di aprire la simulazione del training: non restano due istanze attive contemporaneamente.

Il launcher usa Isaac Lab 2.3.0, CUDA per simulazione/DINO e CPU per PPO. Ogni avvio crea un run distinto `runs/general_map_test4_ppo_*`. Il training non è stato lanciato durante la preparazione. Durante l'inizializzazione non è ancora disponibile l'ETA. Non ci sono dipendenze Python dai folder test1/test2/test3; restano gli asset locali condivisi del progetto.

Per continuare un run del test 4:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test4/train.sh \
  --resume /percorso/al/run/policy_final.pt --timesteps 400000
```

`--timesteps` indica il totale desiderato, non le transizioni da aggiungere. Non occorre modificare config.py per estendere il budget. Questo test parte da zero. Il ripristino rifiuta i checkpoint dei test precedenti e della precedente configurazione test4 con PCA vecchia perché la firma di esperimento è diversa; l’architettura è uguale al test 3. Riprendere ripristina reti, entrambi gli ottimizzatori, normalizzazione, RNG e curriculum, ma ricrea gli episodi fisici.

## Differenze rispetto al test 3

Reward: progresso 30 punti/m, successo +1000, fallimento −1000 e penalità di orientamento oltre 110°. Inoltre la PCA32 viene ricalibrata sui sette terreni della mappa corrente. Energia −E/100, tempo −0,02/passo, mappa, curriculum, architettura, PPO e budget di 300.000 transizioni restano invariati. I fallimenti comprendono caduta, uscita dalla mappa, timeout e timeout di allineamento.

## Allineamento iniziale fisico

A e B vengono campionati al reset, indipendentemente per ogni robot. Lo yaw iniziale è casuale a 360° anche nella prima fase. Il robot parte in assetto eretto; un controllore automatico comanda **vx=0** e **wz=clip(1.5*errore_angolare, -0.8, +0.8) rad/s**.

La policy assume il controllo dal passo successivo a **tre misure consecutive entro 10°** dal goal (tre campioni distanti 0,2 s). Scadenza dell'allineamento: **12 s**, oltre la quale la missione fallisce con esito `alignment_timeout`. Cadute e uscite dalla mappa restano fallimenti. Non ci sono riallineamenti automatici durante la navigazione. Non si teletrasporta l'orientamento verso B.

Il procedimento avviene anche durante le valutazioni/play. Camera, DINO e memoria della gridmap continuano ad aggiornarsi durante la rotazione. Il robot può dunque usare solo osservazioni effettivamente raccolte, mantenute fino al termine della missione.

Ogni ambiente può allinearsi o navigare indipendentemente dagli altri. I reset parziali non fanno avanzare di nascosto la simulazione degli altri robot.

## Conteggio, reward e separazione delle fasi

La misura meccanica integra `sum(abs(tau*qdot))` su tutti i 40 passi fisici da 0,005 s per ciascun comando da 0,2 s. Energia, tempo e percorso dell'allineamento sono inclusi nei **totali della missione**, e registrati anche separatamente dalla navigazione.

Per ogni passo fisico di navigazione/allineamento da 0,2 s il resoconto della missione usa:

```text
r = 30*(d_precedente - d_attuale) - 0.02
    + 1000*successo - 1000*fallimento - energia_step_J/100
    + r_orientamento
r_orientamento = -1.0 * dt * clip((abs(errore_goal_gradi)-110)/70, 0, 1)
```

La penalità di orientamento vale solo nei passi controllati dalla policy, usando la posa dopo il passo. È zero fino a 110° inclusi e cresce linearmente fino a −1 punto/s a 180° (massimo −0,2 per passo da 0,2 s); è simmetrica destra/sinistra. Durante l’allineamento automatico è zero, compreso il passo che lo conclude. Entra nella testa navigazione del critic; il costo energetico resta invariato. `extras["heading_reward"]` espone il contributo istantaneo.

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

Per testare automaticamente il `policy_final.pt` più recente (per data di modifica), con un robot visibile e nove episodi:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test4/test.sh
```

Il launcher stampa il checkpoint selezionato e usa la PCA incorporata nel checkpoint. Se manca un modello finale termina con un messaggio; si può scegliere un altro checkpoint con `--checkpoint /percorso/policy_latest.pt`. Accetta anche le opzioni di `play.py`, ad esempio `--episodes 20` o `--stage 0`. Non avvia training o calibrazione.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/train_general_big_map_test4/play.py \
  --checkpoint /percorso/al/run/policy_final.pt \
  --device cuda:0 --episodes 9 --seed 20000
```

Di default valuta l'ultima fase effettivamente valutata durante il training; `--stage 0` ... `--stage 4` seleziona una fase. Non vengono sovrascritti risultati precedenti. `results/` contiene episodi, summary e traiettorie con start/goal, orientamento, fase, comandi realmente applicati e energia. La posa terminale esatta è nei campi `final_x/final_y` del record episodio; i campioni della traiettoria sono pre-step. `actor.pt` esporta soltanto rete e normalizzazione: un'integrazione esterna deve mantenere anche la macchina di allineamento iniziale, oltre alla percezione e al controllore locomotorio.

## Verifiche

Test offline per PPO/checkpoint/export, isolamento delle informazioni privilegiate, geometria/PCA/memoria, sampling e curriculum, reset parziali, energia su tutti i passi fisici, allineamento/timeout, raccolta asincrona e GAE, budget finale e aggiornamenti critic dopo stop actor. Verificati asset e hash. Nessun training o simulazione del test 4 avviato durante la preparazione: il comportamento fisico completo va verificato al primo avvio.

## Nuova PCA automatica sulla mappa corrente

`train.sh` esegue prima `prepare_pca.py --reuse`, e soltanto se questo termina correttamente avvia `train.py`. La nuova PCA è `artifacts/pca32_big_map.npz`. La vecchia `artifacts/pca32.npz` rimane conservata come riferimento storico, ma non è il default e viene rifiutata dal training aggiornato.

La raccolta usa **4 copie** della stessa mappa congelata `maps/real_terrains_v2.usd`, la stessa camera, DINO e preprocessing del training. Acquisisce **96 viste di calibrazione + 24 di verifica per terreno**, per un totale di **840 viste**, con posizioni e yaw diversi. Sono acquisizioni statiche da pose del robot eretto: non occorre che una policy sappia già navigare e non si percorrono missioni RL. Il ghiaccio viene osservato da punti vicini non ghiacciati. Per gli altri terreni si usano i punti di appoggio già controllati nella banca. Le regole di spawn/goal del training non cambiano.

I punti camera derivano solo dalla porzione training della banca: i punti riservati alla valutazione delle missioni non sono selezionati. Le posizioni di acquisizione fit e verifica PCA sono disgiunte. Questo è un controllo su viste separate della stessa mappa, non un test di generalizzazione su mappe sconosciute: i campi visivi possono sovrapporsi.

Le patch DINO384 vengono associate al materiale tramite depth e coordinate della mappa soltanto per bilanciare la calibrazione, escludendo i centri entro 12 cm dai bordi delle celle. Non è una garanzia di purezza dell'intera patch RGB. L'actor non riceve etichette o conoscenza aggiuntiva della mappa.

Reservoir fino a **10.000 patch per materiale**, quindi fit bilanciato fino a 70.000 patch; il numero effettivo è quello raggiunto dal materiale meno campionato. Si richiedono almeno **1.000 patch e 8 viste per ogni terreno** nel fit e **250 patch e 8 viste per terreno** nella verifica. Se manca copertura, la sequenza si interrompe senza avviare il training. PCA centrata 384→32, senza whitening, poi completamente congelata durante l'RL. La verifica misura la varianza conservata, non accuratezza di classificazione o successo di navigazione; non seleziona automaticamente una dimensione diversa da 32.

Output:

- `artifacts/pca32_big_map.npz`: nuova PCA, metadata, identità mappa/DINO.
- `artifacts/pca32_big_map.json`: copertura e varianza conservata, anche per materiale nelle viste di verifica.
- `artifacts/pca32_big_map.samples.npz`: feature grezze di fit/verifica separate e codici dei materiali.
- `artifacts/pca32_big_map.collection.json`: punti/yaw delle viste, split e quantità osservate.

La PCA viene pubblicata con rinomina atomica solo dopo fit e controlli riusciti; un lock impedisce due calibrazioni contemporanee sullo stesso file. Un avvio successivo riusa la PCA se identità mappa/DINO, protocollo e copertura sono compatibili, senza aprire la simulazione di raccolta. Se la raccolta fallisce prima del salvataggio della PCA, il comando standard la ripete da zero. Con `--resume`, il launcher salta la raccolta e il training usa la PCA incorporata nel checkpoint, verificandone la compatibilità.

Le 300.000 transizioni contano soltanto l'RL: calibrazione, allineamento iniziale e valutazioni aggiungono tempo. Nessun training o nuova acquisizione è stato avviato durante la preparazione; verificati 39 test offline e la sequenza del launcher con un eseguibile simulato.

Per una calibrazione con più viste, se quella standard segnala copertura insufficiente, si può usare un nuovo file:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/train_general_big_map_test4/prepare_pca.py \
  --headless --device cuda:0 --views-per-terrain 192 --validation-views 48 \
  --output spot_rl_energy_navigation/train_general_big_map_test4/artifacts/pca32_big_map_more_views.npz

bash spot_rl_energy_navigation/train_general_big_map_test4/train.sh \
  --pca spot_rl_energy_navigation/train_general_big_map_test4/artifacts/pca32_big_map_more_views.npz
```

Il report `analysis/pca_audit_20261027/REPORT.md` riguarda la PCA **storica**. La decisione successiva è di sostituirla prima del nuovo training, senza alterare i vecchi checkpoint.

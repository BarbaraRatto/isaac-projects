# Test 13: PPO con memoria RGB-D sulle mappe asfalto/rocce

Da `my-projects`:

```bash
bash spot_rl_energy_navigation/multi_terrain_test13/train_test13.sh
```

100.000 transizioni esatte, 32 ambienti (16 per mappa), 24 rollout da
4096 transizioni e uno da 1696 (53 passi vettoriali). GUI disabilitata,
simulazione e DINO su cuda:0, PPO su CPU. Nessuna rigenerazione dei terreni,
nessun fitting PCA e nessun riutilizzo dei pesi di policy precedenti.
Il primo log arriva dopo il primo rollout; inizializzazione e caricamento
delle mesh precedono la raccolta delle transizioni.

Monitor in un secondo terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/multi_terrain_test13/status.py
```

I risultati sono in `runs/test13_ppo_<timestamp>/`. Checkpoint circa ogni
50.000 transizioni (al confine del rollout), valutazione deterministica ogni circa 100.000 transizioni e alla fine
su tutte le 32 istanze, `policy_final.pt`, `policy_best.pt`, `actor.pt`.
Le valutazioni sono escluse dal budget di apprendimento e dalle statistiche
di normalizzazione. È possibile anticiparle con `--eval-every 25000`.

Valutazione grafica del modello finale (sostituire il percorso):

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test13/play.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test13/runs/<run>/policy_final.pt \
  --episodes 5 --device cuda:0 --agent-device cpu
```

`play.py` valuta entrambe le mappe (due ambienti); la viewport mostra la prima.
Per vedere una mappa alla volta, in sequenza automatica:

```bash
bash spot_rl_energy_navigation/multi_terrain_test13/play_sequential.sh \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test13/runs/<run>/policy_final.pt \
  --episodes 5
```

Prima 5 episodi sulla mappa con rocce a destra, poi 5 sulla speculare. La prima
finestra si chiude automaticamente e si apre la seconda, con un nuovo processo
Isaac Sim. Attendere il completamento senza chiudere manualmente la finestra.
Ogni mappa produce una directory risultati distinta; lo stesso seed iniziale
permette di abbinare le condizioni di reset nei due test a singolo ambiente.

CSV di episodi e traiettorie e riepilogo JSON in `results/`. Le metriche per
mappa includono successo, energia e lato percorso. `route` è una diagnostica
del percorso della base: `right`, `left`, `mixed` (>25 cm su entrambi i rami)
o `none` (<10 cm sui rami). Non rappresenta i contatti reali delle zampe né
un'etichetta assegnata all'actor. Il tasso di percorso asfaltato non implica
automaticamente un minor consumo: confrontare i joule degli episodi riusciti.

## Asset congelati

`maps/rocks_right.usd` e `maps/rocks_left.usd` sono copie immutate della
coppia originale `terrain_generator/manual_asphalt_rocks/`, realizzata dal
terreno originale `real_terrains.usd` prima della variante con sassi arrotondati.
Si riutilizzano gli USD esistenti: il generatore, modificato successivamente,
non viene eseguito. Asfalto a partenza/goal, ghiaccio centrale, rami asfalto e
rocce speculari. Le texture restano gli asset condivisi del progetto.
Non modificarle durante l'esperimento.

`artifacts/pca32.npz` è la PCA di test10, copiata senza modificarne componenti
o metadati di calibrazione. Si verifica l'identità DINO/preprocessing, senza
richiedere che le mappe di calibrazione siano quelle di training. Gli hash
dei nuovi USD vengono invece verificati integralmente per resume e play.
`artifacts/manifest.json` conserva hash e provenienza degli asset. Ogni run
salva una copia del codice, della PCA, della configurazione e degli hash.

## Osservazione e memoria

Actor: **3255 → 512 → 256 → 128 → 64 → 2**, ELU, Gaussian tanh.
Critic indipendente: **3279 → 512 → 256 → 128 → 64 → 2** (valore navigazione,
costo energetico positivo). Iperparametri PPO, azioni e controller locomotorio invariati;
reward aggiornata come sotto. Comandi vx in [0,1.6] m/s, vy=0,
wz in [-1.5,1.5] rad/s. Nav 5 Hz, locomozione 20 Hz, fisica 200 Hz.

15 valori di stato: goal XY nel frame yaw, velocità lineare e angolare della
base (3+3), gravità proiettata (3), azione precedente (2), potenza meccanica
filtrata (1), tempo residuo (1). Le 81 celle sono un quadrato 9×9 con passo
0,5 m, centri da -2 a +2 m su entrambi gli assi, bordi esterni ±2,25 m.
X avanti, Y sinistra; ordine X esterno/Y interno. Il robot è nella cella 40.

40 valori per cella: PCA32, altezza media relativa alla base, pendenza lungo
X/Y (angoli), rugosità (RMS residuo ortogonale al piano), validità visiva,
validità geometrica, età visiva, età geometrica. Le età sono medie pesate
sui campioni conservati, divise per 40 s e limitate a [0,1]. Valori invalidi
azzerati anche dopo normalizzazione; maschere binarie non normalizzate.
Le età invalide sono zero, con la rispettiva maschera zero a distinguerle.

La PCA si applica prima dell'inserimento nella memoria. Ogni ambiente ha
una mappa sparsa di bin XY da 10 cm in un riferimento fisso dell'episodio,
ancorato alla prima posa osservata. Per ogni bin, la nuova osservazione
sostituisce le statistiche della stessa modalità: media PCA e posizione
dei patch, oppure numero/media/covarianza dei punti depth. Non si accumulano
indefinitamente conteggi delle rivisite. Le covarianze conservano la rugosità
interna ai bin; la geometria non viene ridotta a una singola quota per bin.

Alla lettura si assegnano i centroidi dei bin alle celle robot-centriche;
si aggregano feature e momenti, si ruotano le pendenze e si sottrae la quota
attuale della base. È un'approssimazione spaziale a 10 cm: un bin che incrocia
il bordo di una cella viene assegnato interamente in base al centroide.
Nessuna interpolazione ripetuta della mappa e nessun riempimento dei vuoti.
Validità geometrica: almeno 10 punti e supporto XY bidimensionale sufficiente.

Le misure fuori vista restano disponibili fino a fine episodio. Una finestra
vuota della camera non cancella le osservazioni pregresse. Un reset elimina
soltanto la memoria dell'ambiente interessato, anche fra valutazione e training.
Checkpoint/resume ricominciano da nuove scene/episodi con memoria vuota;
`actor.pt` richiede che il chiamante costruisca le stesse osservazioni, non
contiene il mapper o DINO. Non vengono usati terreno USD, etichette o attriti
per riempire la mappa dell'actor. La registrazione usa la posa simulata esatta:
non simula deriva odometrica o SLAM. Lo scenario è statico e non si applica TTL.

Critic: input actor più 24 privilegi (attriti statici/dinamici delle quattro
zampe, contatti, forze normali, velocità tangenziali), come test10.
Reward: 3×progresso in metri −0,02 per passo +150 goal −300 fallimento;
costo meccanico separato E_abs/100, peso 1 (penalità effettiva −0,01/J). Deadline 40 s, goal entro 0,55 m.
GAE delle due teste combinati per l'actor; non è un algoritmo di risoluzione
esplicita dei conflitti tra gradienti. La memoria permette di usare misure
passate ma non garantisce che la policy impari a scegliere il percorso migliore.

Il peso energetico resta fisso a 1, senza curriculum. Dopo questa revisione
avviare una nuova run senza `--resume`: i checkpoint precedenti hanno una
firma diversa. Le run esistenti conservano configurazione e sorgenti originali.

## Verifiche senza simulazione

```bash
python3 -m pytest spot_rl_energy_navigation/multi_terrain_test13/test_offline.py \
  spot_rl_energy_navigation/multi_terrain_test13/test_memory.py -q
```

Coprono traslazione/rotazione, altezza relativa, rugosità, dati mancanti,
età separate, reset parziali, budget esatto, normalizzazione, GAE, PPO sintetico,
checkpoint e export. Gli USD vengono controllati con il Python USD senza
SimulationApp. Nessun training o test fisico è stato avviato dall'assistente.

Per un controllo tecnico nella simulazione, eseguibile dall'utente:

```bash
bash spot_rl_energy_navigation/multi_terrain_test13/train_test13.sh --probe-steps 10
```

Non aggiorna PPO; crea i 32 ambienti, acquisisce osservazioni e stampa forme
3255/3279, validità ed energia. Tempi e uso GPU richiedono verifica nella simulazione effettiva. Il monitor stima l'ETA dai rollout reali.

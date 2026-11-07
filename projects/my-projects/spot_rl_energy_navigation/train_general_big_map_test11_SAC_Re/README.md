# Test11 SAC Re — training locale, 280.000 transizioni

Training da zero con la configurazione locale del test10 SAC Re ripristinata
dopo la prova delle accelerazioni del test9. Cambiano soltanto il layout
della mappa e il budget: **280.000 transizioni complessive** con 32 robot.
Il numero è anche multiplo di 32 (8750×32), sebbene questo collector SAC
accetti un budget qualsiasi e conti esattamente i campioni richiesti.

## Avvio e monitoraggio

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/train.sh
```

Prima controlla la mappa già generata, calibra automaticamente una nuova
PCA384→32 sui due materiali della nuova mappa e poi avvia il training.
Le viste di calibrazione sono distinte dai candidati di valutazione.
La PCA viene congelata durante RL e riutilizzata ai successivi lanci se
compatibile. Non vengono importati checkpoint o replay del test10.

In un altro terminale:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/status.py
```

Simulazione, locomozione, DINO e proiezione PCA girano sulla GPU; agente SAC,
replay, preprocessing RGB, gridmap e geometria restano sulla CPU, come nella
versione locale precedente. Nessuna accelerazione del test9 viene abilitata.
Il replay resta da 100.000 campioni, il batch da 256 e gli aggiornamenti da
uno per vector step con campioni di navigazione: aumentare il budget non
modifica questi parametri. Il target è la somma dei campioni dei 32 robot,
escludendo allineamento, calibrazione e valutazione.

## Nuova mappa

`maps/lab_15x15.usda`: 15×15 m, celle 3×3 m, 8 verdi e 17 grigie.
La prima riga dello schema è quella in alto, a Y maggiore:

```text
v g g v g
g v g g g
g v g v g
g g g v v
v g g g g
```

`v` = verde, `g` = grigio. Il layout esplicito è salvato in
`layout_spec.json`; `maps/layout.png` e `maps/layout.pdf` ne mostrano lo
schema. Non viene randomizzato tra gli episodi. Le 32 scene hanno copie
separate della stessa mappa.

| Parametro | Grigio | Verde |
|---|---:|---:|
| Attrito statico / dinamico | 1,0 / 0,8 | 1,0 / 0,8 |
| Contatto | Rigido | Rigido |
| Superficie | Piana | Piana |
| Resistenza orizzontale | Nessuna | 90 N·s/m |

Come nel test precedente, F_xy = −90 v_xy viene applicata alla base quando
la sua proiezione XY è dentro una cella verde. Si applica anche durante
l'allineamento. La forza è aggiornata ogni 0,005 s e non introduce coppie o
forze verticali. È un modello sintetico di resistenza all'avanzamento,
non un pavimento reale calibrato. L'energia è sempre misurata ai giunti,
senza aggiungere Joule artificiali. Il solo USD non applica la resistenza:
serve l'ambiente di questo esperimento.

## Missioni, rete e reward

Start e goal sono ricampionati indipendentemente per ogni robot a fine
episodio, con margine di 0,65 m dai bordi delle celle. Curriculum invariato:

| Fase | Distanza A–B | Endpoint |
|---|---|---|
| 0 | 1,5–3 m | Grigio → grigio |
| 1 | 3–6 m | Grigio e verde |
| 2 | 6–10 m | Grigio e verde |
| 3 | 10–14 m | Grigio e verde |

Valutazione ogni 10.000 transizioni e alla fine, 32 episodi deterministici;
promozione dopo due valutazioni consecutive con almeno l'80% di successi.
Il 25% delle missioni nelle fasi successive riprende una fase più semplice.
Allineamento fisico iniziale al goal invariato. Il goal ha raggio 0,55 m;
timeout clip(15+4×distanza, 30, 90) secondi.

SAC asimmetrico con due Q indipendenti. Actor: 15 stati e 81×40 valori della
gridmap robocentrica, 3255 input. Stato 15→64→32; mappa
3240→1024→512→256→128; fusione 160→128→64; due azioni vx e wz.
Goal relativo diviso per 10, senza statistiche dinamiche; altri scaler
dinamici con validità, congelati durante i test. Il critic ha gli stessi
privilegi del test10: attriti, contatti, forze e slip delle quattro zampe.

Reward invariata per azione di 0,2 s:

`r = 30(d_pre−d_post) − 0,02 − E_step/100 + 1000 I_goal − 1000 I_failure + r_heading`

La penalità di orientamento è nulla fino a 110°, poi cresce linearmente
fino a −2 punti/s a 180°. Caduta: inclinazione ≥90° oppure contatto del
torso persistenti per 1 s; l'altezza della base è soltanto diagnostica.
Learning rate 3e−4, gamma 0,999, tau 0,005, warmup 4096, alpha iniziale 1.

## Arresto e salvataggi

Ctrl+C una volta: compare `[SAC STOP]`, si conclude il passo corrente e si
salvano `policy_interrupted.pt` e replay. Attendere il ritorno del prompt.
Sono mantenuti il gestore dei segnali e la diagnostica degli errori.
Conservare l'intera run, non soltanto il checkpoint. Le policy periodiche
senza replay corrispondente non bastano per una ripresa completa.

```bash
bash spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/train.sh \
  --resume /PERCORSO_DELLA_RUN/policy_interrupted.pt
```

Il target resta 280.000 totali; la ripresa ricrea gli episodi fisici.

## Valutazione dopo il training

Per il test richiesto di nove episodi visibili, uno alla volta:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/test_9.sh
```

Seleziona `policy_best.pt` della run completata più recente. Esegue due
missioni grigio→grigio vicine, due a circa 6 m, due a circa 9 m, due a circa
12 m e una grigio→verde. Sono le nove nuove missioni predisposte per il
test12, compatibili con la stessa nuova mappa e lo stesso banco di endpoint;
non riutilizzano i punti del vecchio test10. Piano e coordinate sono in
`artifacts/missions_9.md` e `artifacts/missions_9.json`.
Salva percorsi, start, goal, energia ed esiti in una nuova cartella `results/`.
Non avvia apprendimento, aggiornamento scaler o calibrazione PCA.
Se la run è una ripresa che non ha migliorato la best precedente, la selezione
risale alla run originaria e usa quel checkpoint, controllando configurazione
e asset. Le nuove run interrotte e incomplete vengono ignorate; nessun modello
viene sovrascritto o sostituito con quello avviato per errore.

`test.sh` seleziona la **policy best della run completata più recente**:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/test.sh
```

Per un altro checkpoint usare `--checkpoint /percorso/policy_final.pt`.
Salva percorsi, start, goal, energia e diagnostica in `results/`.
Il generatore `analyze_paths.py` crea mappa dei percorsi, zoom e tabella:

```bash
python3 spot_rl_energy_navigation/train_general_big_map_test11_SAC_Re/analyze_paths.py \
  /PERCORSO_DEI_RISULTATI
```

Il pacchetto per la GPU online non viene creato in questo passaggio:
sarà preparato dopo averne definito la configurazione separata.

# Test12 SAC Re — GPU online

Training da zero, stessa mappa e task del test11 SAC Re. Nessun checkpoint o replay precedente viene caricato automaticamente. Pacchetto portatile unico: `spot-test12-sac-re-cloud.zip`; istruzioni complete in `cloud/COMANDI_TEST12_RE.md`.

| Impostazione | Valore |
| --- | --- |
| Robot | 128 scene indipendenti, stessa mappa |
| Budget | 1.500.000 transizioni di navigazione complessive |
| SAC | 16 aggiornamenti per passo vettoriale contenente campioni di navigazione, dopo warmup |
| Replay | 100.000 campioni, batch 256, circa 2,44 GiB su GPU |
| Learning rate / gamma / tau | 0,0003 / 0,999 / 0,005 |
| Valutazioni | Ogni 10k transizioni nelle fasi 0–1; ogni 30k nelle fasi 2–3; anche alla fine |
| Avanzamento | Almeno 80% di successi in due valutazioni consecutive, 128 episodi ciascuna |
| Recovery | Ogni 30 minuti, iniziale, promozione, finale e interruzione; conserva 2 generazioni verificate |
| Checkpoint ordinari | Ogni 10k transizioni |

## Mappa e missioni

Mappa piana 15×15 m, 25 celle da 3×3 m: 17 grigie, 8 verdi. Schema visto dall'alto, prima riga in alto (Y maggiore):

```text
v g g v g
g v g g g
g v g v g
g g g v v
v g g g g
```

La mappa `maps/lab_15x15.usda` è identica byte per byte a quella del test11. Entrambi i materiali hanno attrito statico 1 e dinamico 0,8. Sul verde si applica alla base una resistenza orizzontale `Fxy = -90 * velocità_xy`, in N con velocità in m/s, senza forza verticale o coppia. L'energia è il lavoro meccanico misurato dalle articolazioni, senza aggiunte artificiali. La resistenza si applica anche durante l'allineamento iniziale.

Start e goal casuali, con margine dai bordi delle celle e candidati di valutazione separati da quelli di training. Le fasi sono:

| Fase | Distanza euclidea A–B | Terreni per start e goal |
| --- | --- | --- |
| 0 | 1,5–3 m | Grigio |
| 1 | 3–6 m | Grigio e verde |
| 2 | 6–10 m | Grigio e verde |
| 3 | 10–14 m | Grigio e verde |

Dal livello 1, il 25% delle missioni di training proviene dalle fasi precedenti. I percorsi intermedi possono attraversare qualsiasi cella, anche nella fase 0. Yaw iniziale casuale; allineamento algoritmico al goal prima della navigazione. Nessun pianificatore impone il percorso. Timeout `clip(15 + 4*distanza, 30, 90)` secondi; goal entro 0,55 m. Caduta: inclinazione ≥90° oppure contatto del tronco oltre 1 N, persistenti per 1 secondo; l'altezza è diagnostica.

## Policy e reward

Actor con due rami: stato `15→64→32`, mappa `3240→1024→512→256→128`; concatenazione `160→128→64`, distribuzione gaussiana per 2 azioni: vx 0–1,6 m/s e wz ±1,5 rad/s. Critic SAC asimmetrici gemelli con 24 input privilegiati aggiuntivi. Griglia locale robocentrica 9×9, da −2 a +2 m in ciascuna direzione; 32 feature PCA più geometria, validità e anzianità delle misure. Memoria spaziale delle celle già osservate.

Reward per passo di navigazione:

```text
30 * (distanza_precedente - distanza_attuale)
- 0,02
- energia_meccanica_del_passo_J / 100
+ 1000 se raggiunge il goal
- 1000 in caso di fallimento
- penalità orientamento
```

La penalità di orientamento è nulla fino a 110°, cresce linearmente fino a 180°, con massimo 2 punti/s (0,4 punti per azione da 0,2 s). Il peso energetico resta 1. Nessun peso viene aumentato rispetto al test11.

PCA 32 congelata dopo calibrazione automatica su viste dei due materiali, prima dell'RL. Il launcher riutilizza la calibrazione solo se compatibile. Goal relativo diviso per 10 senza centratura; gli altri scaler continuano ad aggiornarsi durante il training, come nel test7/test11. In valutazione sono congelati. Non si usa lo schema con tutti gli scaler congelati del vecchio test10 non-Re.

## Accelerazioni e portabilità

Fisica, locomozione, rendering, DINO/PCA, RGB, griglia geometrica/visiva, SAC e replay usano CUDA. Reset raggruppati, metriche SAC raccolte al logging, resistenza del verde calcolata in batch. Il preprocessore RGB verifica l'equivalenza esatta sui primi frame reali e interrompe in caso di differenze. Campionamento missioni, gestione curriculum, CSV e salvataggi restano sulla CPU/disco.

Non esiste un vincolo alla RTX 5090: il profilo di installazione supporta GPU RTX, incluse RTX 6000, con memoria sufficiente. Il driver deve supportare le versioni del runtime; il valore CUDA riportato da nvidia-smi indica la compatibilità del driver, non obbliga a installare quella stessa versione nelle librerie Python. Il setup conserva il runtime cu128 già usato dai training online precedenti, senza modificare i driver.

I backend e il dispositivo sono configurabili. Cambiare robot o aggiornamenti SAC cambia l'esperimento: per una ripresa fedele mantenere 128 e 16. Non è stato eseguito un benchmark su GPU online per questo test.

## Avvio, monitor e ripresa

Vedere [la guida cloud](../cloud/CLOUD_TEST12_RE.md). Il launcher `train.sh` prepara automaticamente mappa/PCA e poi avvia il training. Non è stato lanciato durante la preparazione dei file.

Conservare **tutta la cartella della run**, inclusa `recovery/`, e il codice/asset dell'esperimento. Un `policy_*.pt` da solo non conserva il replay necessario alla ripresa completa. Ogni run contiene anche una copia di sorgenti, mappa e manifest, oltre alla PCA. I salvataggi completi includono learner, ottimizzatori, scaler, curriculum, RNG e replay. Non conservano lo stato fisico preciso della simulazione: al resume gli episodi ripartono da nuovi reset.

Ctrl+C una sola volta richiede arresto e salvataggio al prossimo punto sicuro. Un kill forzato, crash GPU o macchina spenta può impedire il salvataggio corrente: si riparte dall'ultima generazione completa e verificata, perdendo le transizioni successive. Il recovery evita che una scrittura incompleta sostituisca l'ultima copia valida; non può impedire la perdita dell'intero disco della VM.

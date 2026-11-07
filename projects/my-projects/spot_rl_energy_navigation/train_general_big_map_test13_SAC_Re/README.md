# Test13 SAC Re — grigio e ghiaccio nero

Training nuovo da zero. Base: test12 SAC Re; stessi algoritmi, rete, budget, reward, scaler, accelerazioni e recovery. Differenze autorizzate: sostituzione del verde resistente con nero scivoloso, rimozione della resistenza viscosa e esclusione del nero da start/goal in tutte le fasi. Non carica pesi o replay di altri training.

## Configurazione

| Voce | Valore |
| --- | --- |
| SAC | Asimmetrico con due Q critic |
| Transizioni totali | 1.500.000, aggregate sui robot |
| Robot | 128 scene separate, stessa mappa |
| Aggiornamenti | 16 per passo vettoriale contenente campioni di navigazione, dopo warmup |
| Replay / batch / warmup | 100.000 / 256 / 4096 |
| Learning rate / gamma / tau | 0,0003 / 0,999 / 0,005 |
| Valutazioni | 128 episodi, ogni 10k transizioni nelle fasi 0–1, 30k nelle fasi 2–3 e alla fine |
| Promozione | Almeno 80% in due valutazioni consecutive |
| Recovery completa | Ogni 30 minuti; anche iniziale, promozione, interruzione ordinata e finale; 2 generazioni verificate |
| Checkpoint ordinari | Ogni 10k transizioni |

## Mappa e materiali

15×15 m; 5×5 celle piane rigide da 3×3 m. Disposizione del test12, con 17 celle grigie e 8 nere. Prima riga in alto, Y maggiore:

```text
n g g n g
g n g g g
g n g n g
g g g n n
n g g g g
```

| Proprietà | Grigio | Nero |
| --- | --- | --- |
| Attrito statico | 1,0 | 0,10 |
| Attrito dinamico | 0,8 | 0,08 |
| Restituzione | 0 | 0 |
| Cedevolezza | Nessuna | Nessuna |
| Resistenza viscosa aggiuntiva | Nessuna | Nessuna |
| Aspetto | Grigio chiaro | Nero uniforme |

La fisica del nero riprende `t2_slippery` in `terrain_generator/create_real_terrains.py`. Combine mode del materiale: average; l'attrito effettivo al contatto dipende anche dal materiale del piede e dalla precedenza PhysX, come nei test precedenti. Nessuna penalità diretta per colore, materiale, visita o contatto con il nero. L'energia rimane il lavoro meccanico delle articolazioni, senza incrementi artificiali.

Il codice comune del wrench resta presente per compatibilità del sistema ma tutti i coefficienti sono zero: nessuna forza resistente aggiuntiva viene applicata sul nero o sul grigio.

## Missioni e curriculum

Start e goal casuali, **sempre sul grigio**, con margine di 0,65 m dai bordi delle celle. Il nero rimane fisicamente percorribile e può trovarsi lungo il tragitto; nessun planner impone una deviazione. I campioni di start/goal per valutazione sono separati da quelli di training. Verificata la connessione delle 17 celle grigie per adiacenza sui lati; ciò non garantisce che una policy impari a navigare.

| Fase | Distanza euclidea A–B |
| --- | --- |
| 0 | 1,5–3 m |
| 1 | 3–6 m |
| 2 | 6–10 m |
| 3 | 10–14 m |

Dal livello 1 si campiona il 25% delle missioni dalle fasi precedenti. Le tratte non sono bilanciate esplicitamente per presenza di ghiaccio. Yaw casuale, poi allineamento algoritmico al goal. Goal entro 0,55 m. Deadline `clip(15+4*distanza,30,90)` secondi. Caduta con tilt ≥90° o contatto del tronco oltre 1 N, persistenti per 1 secondo; altezza solo diagnostica.

## Rete, percezione e reward

Identiche al test12: stato `15→64→32`, mappa `3240→1024→512→256→128`, fusione `160→128→64`, distribuzione gaussiana per 2 azioni: vx 0–1,6 m/s e wz ±1,5 rad/s. Actor 3255 input, critic con altri 24 input privilegiati. Gridmap robocentrica 9×9 da −2 a +2 m con feature PCA32, geometria, validità e anzianità; memoria delle misure già viste.

PCA nuova calibrata automaticamente sui due colori prima dell'RL, poi congelata; include viste sul nero anche se nero è escluso dalle missioni. Nessun trasferimento della PCA del test12. Goal /10 fisso senza centratura, altri scaler dinamici durante il training e congelati nella valutazione.

Reward per passo: `30 * progresso_m - 0,02 - energia_J/100`, con `+1000` al goal, `-1000` al fallimento e stessa penalità di orientamento: nulla fino a 110°, lineare fino a 180°, massimo 2 punti/s. Nessuna nuova componente di reward.

## GPU e ripresa

Simulazione, locomozione, rendering, DINO/PCA, RGB, mappe/geometria, SAC e replay su CUDA. Mantiene reset raggruppati, raccolta differita delle metriche e verifica dell'equivalenza RGB sui frame reali. Campionamento, curriculum e file su CPU/disco. Configurabile per GPU RTX compatibili: nessun vincolo al modello 5090. Nessuna simulazione o installazione online è stata avviata durante la preparazione.

Salvare tutta la run, in particolare `recovery/`, e il pacchetto originale. Le recovery verificano hash e abbinamento policy/replay con fallback alla copia precedente. Un checkpoint ordinario da solo non ripristina il replay. Ctrl+C una volta richiede un arresto al prossimo punto sicuro. Il resume ripristina learner, ottimizzatori, scaler, curriculum, RNG e replay, ma resetta gli episodi fisici. Una perdita della VM senza disco persistente richiede un backup esterno.

## Test e grafici

`test_9.sh`: 2 missioni vicine, 2 a circa 6 m, 2 a 9 m, 2 a 12 m, 1 con ghiaccio sul segmento diretto A–B. Tutti i 18 endpoint sono diversi, appartengono allo split di valutazione e sono sul grigio. Seleziona `policy_best.pt` dell'ultima run completata; checkpoint esplicito per confrontare final o best_stage. Salva traiettorie, esiti, energia e diagnostica.

`analyze_results.sh`: genera automaticamente panoramica, zoom, tabella e report dall'ultimo test completato, senza simulazione o chat. Gestisce anche zero successi. `maps/layout.png` e PDF mostrano la disposizione; `artifacts/missions_9.md` descrive le missioni.

Istruzioni riassuntive complete nel file `cloud/COMANDI_TEST13_RE.md`, incluso alla radice dello ZIP come `LEGGIMI_TEST13_RE.md`. Pacchetto: `spot-test13-sac-re-cloud.zip`.

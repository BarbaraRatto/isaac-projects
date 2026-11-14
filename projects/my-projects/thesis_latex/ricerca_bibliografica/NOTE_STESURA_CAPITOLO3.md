# Stesura progressiva del capitolo 3

## Prima sezione — 10 novembre 2026

Scritta la sezione 3.1, **Formulazione del problema**, con tre sottosezioni: obiettivo e costo, informazione disponibile, assunzioni e vincoli. La stesura segue la richiesta di un linguaggio semplice e non contiene risultati sperimentali.

Per lo scenario esteso il riferimento indicato dall'autrice è `train_general_big_map_test7_SAC`. Sono stati consultati direttamente `config.py`, `missions.py`, `train.py`, `actions.py` ed `env.py`, oltre al README. I rimandi a versioni precedenti nel README non sono ripresi nel testo. Per la formulazione sono stati verificati: distanza XY, raggiungimento del goal, segnali dell'actor, comandi longitudinali e angolari, costo meccanico, memoria e termini della ricompensa.

La struttura è aggiornata a otto sezioni. Le parti successive non ancora scritte sono conservate come scaletta dentro un blocco LaTeX `\iffalse ... \fi`: verranno abilitate una alla volta durante la stesura. Il PDF mostra quindi soltanto l'introduzione del capitolo e la sezione effettivamente scritta. Non sono state modificate le impostazioni tipografiche.

Il curriculum avrà spazio nella sezione 3.7: descrivere funzionamento e motivazioni delle cinque fasi del campionamento, criteri di promozione e mantenimento delle missioni più facili. I vincoli di campionamento non equivalgono a divieti di attraversamento delle superfici intermedie. Il raggiungimento effettivo delle fasi andrà verificato dai log quando saranno discussi i risultati.

La compilazione del manoscritto è completata senza errori, rimandi irrisolti o avvisi BibTeX. Non risultano debordamenti nella parte scritta del capitolo 3. La bozza, incluso il titolo del capitolo, occupa le pagine numerate 25–27; l'inizio del capitolo successivo non è indicativo della lunghezza futura della metodologia.

## Sezione 3.2 e supporto grafico alla 3.1 — 10 novembre 2026

Scritta la sezione **Piattaforma e architettura del sistema** con le cinque sottosezioni previste: robot e controllore; sensori; simulazione e terreni; flusso delle informazioni; frequenze e sincronizzazione. È stata abilitata nel PDF, mentre la scaletta delle sezioni 3.3–3.8 resta nel blocco `\iffalse`.

La sezione 3.1 è stata riletta e integrata con un placeholder per una figura concettuale: percorsi su superfici di costo diverso, goal noto e visibilità limitata. Non è una traiettoria sperimentale. Non è stata aggiunta una tabella, perché la prima sezione introduce il compito e le tabelle dei dettagli appartengono alle sezioni successive. La 3.2 contiene due placeholder di figure (camera e architettura) e uno di tabella (terreni).

### Riscontri nel codice e negli asset

- `scene.py`: modello Spot, attuatori impliciti, rigidezza 60 N m/rad, smorzamento 1,5 N m s/rad; limiti di coppia 45 N m alle anche e 115 N m alle ginocchia; RGB e depth a 640×360; ottica dal modello ZED X CameraRight.
- `locomotion.py`: policy congelata, dodici riferimenti articolari, scala 0,2 rispetto alla postura nominale; aggiornamento ogni dieci passi fisici.
- `env.py`: traslazione della camera (0,42, 0, 0,07) m e rotazione di 15° verso il basso; posa aggiornata dal corpo; 40 passi fisici per decisione; integrazione della potenza a ogni passo; acquisizione RGB-D dopo sincronizzazione della posa; metriche raccolte prima del reset.
- `config.py`: passo fisico 0,005 s. Ne derivano 200 Hz di fisica, 20 Hz di locomozione e 5 Hz di navigazione, nel tempo simulato.
- `vision.py` e `privileged.py`: DINOv2/PCA congelati, memoria delle osservazioni, separazione degli ingressi dell'actor dalle informazioni privilegiate dei critic.
- Controllati anche `scene.py`, `locomotion.py`, `config.py` ed `env.py` conservati nella run locale `general_map_test7_sac_20261029_163507_402204`. I primi tre coincidono con le sorgenti correnti. Le differenze in `env.py` riguardano la selezione del backend della mappa e non i parametri o la sequenza descritti nella sezione.
- Letti direttamente gli asset USD senza avviare SimulationApp o un training: verificati i dodici giunti del robot, l'ottica della camera e i materiali della mappa congelata. La mappa non contiene corpi rigidi dinamici né attributi deformabili. Non viene quindi descritta come una simulazione di terreno granulare mobile o cedevole.
- Versioni riscontrate nei file locali e nel setup del progetto: Isaac Lab 2.3.0 e Isaac Sim 5.1.0; il file VERSION del simulatore locale riporta il dettaglio di build `5.1.0-rc.19+release.26219.9c81211b.gl`.

Per compilare in seguito il placeholder della tabella, i coefficienti delle superfici sono stati letti dalla copia USD congelata. Valori arrotondati rispetto alla rappresentazione floating point:

| Terreno | Attrito statico | Attrito dinamico | Restituzione |
| --- | --- | --- | --- |
| Asfalto | 1,0 | 0,8 | 0 |
| Superficie scivolosa | 0,1 | 0,08 | 0 |
| Rampa | 1,0 | 0,8 | 0 |
| Ghiaia fine | 0,4 | 0,3 | 0 |
| Rocce | 0,5 | 0,4 | 0 |
| Ostacoli | 1,0 | 0,8 | 0 |
| Scale | 0,9 | 0,7 | 0 |

Questi valori sono quelli dei materiali del terreno, non quelli effettivi al contatto con il piede. L'aspetto visivo non viene identificato con una misura della risposta fisica. Non sono state modificate sorgenti, asset o configurazioni del progetto robotico.

### Fonti e controllo editoriale

Sono richiamati i riferimenti già presenti `Mittal2025IsaacLab` e `Oquab2023DINOv2`. Il riuso del modello e della policy locomotoria è attribuito a [IsaacRobotics](https://github.com/mschweig/IsaacRobotics) in una nota a piè di pagina. Per il ruolo del simulatore e il significato dei dati della camera sono state consultate la [documentazione di Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html), la [documentazione della camera di Isaac Lab 2.3](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html) e il relativo codice locale. La versione di libreria è quella adottata dal progetto, senza deduzioni sulle versioni correnti.

Verificate compilazione, risoluzione delle citazioni e dei rimandi, oltre all'impaginazione di pagine rappresentative del PDF. La sezione 3.2 si estende sulle pagine numerate 27–31; il testo complessivamente scritto del capitolo 3 occupa le pagine 25–31. Sono presenti tre placeholder di figure e uno di tabella. Le sezioni ancora da scrivere non sono presentate come testo completato.

## Sezione 3.3 — 11 novembre 2026

Scritta **Calcolo del costo energetico** con le tre sottosezioni previste: potenza meccanica articolare; integrazione temporale; impiego del segnale e limiti della misura. Aggiunte cinque equazioni per potenza, integrazione ai trapezi, lavoro cumulato, filtro e penalità energetica. Un placeholder di figura distingue i tre percorsi del segnale: osservazione, costo nella ricompensa e totale dell'episodio. Non sono stati introdotti grafici con dati sperimentali o risultati di training.

### Verifiche delle sorgenti

Sono stati letti `_measured_power`, `step`, `_get_rewards`, `_episode_records` e `_reset_idx` in `multi_terrain_test13/env.py` e `train_general_big_map_test7_SAC/env.py`. Lo stesso calcolo e i valori della configurazione sono stati riscontrati in `train_general_big_map_test12_SAC_Re` e `train_general_big_map_test13_SAC_Re`, indicati dall'utente per i modelli destinati al robot reale.

La lettura delle coppie usa `root_physx_view.get_dof_projected_joint_forces()`, che restituisce le componenti delle forze articolari proiettate lungo i gradi di libertà. La semantica è stata verificata nel codice locale di `omni.physics.tensors`, versione 107.3, e nella [documentazione primaria NVIDIA](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_dof_projected_joint_forces). La fonte è richiamata in una nota a piè di pagina. Il testo distingue questo segnale del solver dai riferimenti di posizione del controllore e dal consumo elettrico.

L'integrazione comprende quaranta intervalli fisici di 0,005 s per ciascuna azione. Le letture precedenti e successive a ogni passo fisico alimentano i trapezi; il risultato dell'ultimo intervallo è raccolto prima del reset. Il costo della ricompensa e il cumulato usano il lavoro non filtrato. L'osservazione usa la potenza media dell'intervallo, filtrata con `beta = exp(-0.2/1.0)` e azzerata al reset. Il costo è normalizzato con 100 J e peso 1.

Verificati anche la preparazione della postura prima degli episodi e, per gli ambienti SAC, i conteggi di allineamento e navigazione. Il totale include l'allineamento iniziale, se previsto; la memoria delle transizioni per l'apprendimento conserva le sole azioni controllate dalla policy, come riscontrato in `replay.py`. La distinzione è riportata nel testo senza attribuire all'actor i comandi dell'allineamento.

Il testo spiega con un esempio statico perché il lavoro meccanico non comprende tutte le perdite elettriche. L'esempio è concettuale, senza valori o risultati inventati. L'acquisizione energetica effettiva sul robot reale resta da descrivere nella sezione di integrazione.

La sezione è abilitata nel PDF; il blocco di scaletta nascosto inizia ora dalla 3.4. Sono stati aggiornati i riferimenti di implementazione nei commenti del sorgente e negli appunti condivisi, registrando l'addestramento online in corso dei modelli Re senza presumere la presenza locale dei risultati.

La compilazione finale termina senza errori, citazioni o rimandi irrisolti e senza avvisi BibTeX. Non risultano debordamenti nella parte scritta del capitolo 3. Controllate visivamente equazioni, nota alla documentazione del solver e placeholder. La sezione 3.3 inizia alla pagina numerata 31 e termina con il placeholder alla pagina 34; il testo scritto del capitolo occupa complessivamente le pagine 25–34. Sono presenti quattro placeholder di figure e uno di tabella nell'intero capitolo finora scritto. La posizione del placeholder finale sarà riconsiderata con la continuazione della sezione 3.4.

## Sezione 3.4 — 11 novembre 2026

Scritta **Rappresentazione percettiva e memoria dell'ambiente**, con le cinque sottosezioni previste. La descrizione segue il percorso dei dati senza presentare analisi sperimentali: preparazione RGB-D, feature, PCA, proiezione, geometria, finestra locale e memoria. Inseriti due placeholder di figure (associazione patch–punto e memoria fissa/finestra mobile) e uno di tabella (quaranta componenti della cella). Sono presenti sette equazioni numerate e una relazione di campionamento non numerata.

### Sorgenti e parametri verificati

Letti `vision.py`, `dino_feature_extractor.py`, `perception.py`, `memory.py`, `gpu_memory.py`, `config.py` e la procedura `prepare_pca.py` dello scenario esteso. Consultati anche i file di configurazione locali del modello DINOv2 e del processore. Confrontate le funzioni geometriche dei quattro riferimenti: `backproject`, `world_to_local`, `cell_indices` e `local_cells` risultano identiche, a livello di AST, nei riferimenti speculare, esteso e Re.

Parametri riportati nel testo:

- RGB-D nativo 640×360; RGB ridimensionato a 518×518 con interpolazione bilineare, senza crop centrale. Normalizzazione del processore con i valori presenti nel suo file di configurazione. La depth e gli intrinseci della proiezione restano alla risoluzione originale.
- DINOv2-small congelato: patch da 14 pixel, 37×37 descrittori da 384 componenti; token globale escluso. PCA centrata 384→32 senza whitening, applicata prima di memorizzare le feature.
- Campione depth ai centri delle patch riportati sull'immagine originale e arrotondati; geometria da un campione ogni quattro pixel. Profondità finita con 0,2 < z < 15 m; esclusione approssimata dei punti entro 8 cm dai corpi del robot.
- Riferimento ottico ROS e trasformazione completa camera→mondo; riferimento locale della lettura ottenuto con sola rotazione yaw e traslazione della base, mantenendo verticale l'asse Z.
- Altezza relativa, angoli di pendenza e RMS dei residui ortogonali al piano ai minimi quadrati. Validità geometrica: almeno dieci campioni e minimo autovalore della covarianza XY non inferiore a 10⁻⁴ m².
- Finestra 9×9 con passo 0,5 m: centri ±2 m, bordi ±2,25 m e lato totale 4,5 m. Quaranta valori per cella, 3240 complessivi; indice lineare 9 i_x + i_y, centro numero 40 con numerazione da zero.
- Memoria sparsa con bin XY da 0,10 m, ancorata alla prima posizione osservata e con assi paralleli al mondo. Statistiche visive e geometriche separate; le nuove misure sostituiscono quelle del medesimo bin e della medesima modalità. L'aggregazione pesa con il numero di campioni e preserva le covarianze interne ai bin.
- Età normalizzate per bin prima della media pesata: scala 40 s per lo scenario speculare, 90 s per gli altri riferimenti. Nessuna scadenza automatica; reset separato per ambiente. Le scale temporali sono state verificate nei quattro `config.py`.

Confrontata anche la memoria nella copia dei sorgenti della run locale dello scenario esteso: la gestione corrente degli array ottimizza la struttura di memorizzazione ma conserva le statistiche e le operazioni descritte. Per i riferimenti Re è stata controllata anche la pipeline RGB, che può usare un backend CUDA progettato per mantenere ridimensionamento e normalizzazione del riferimento OpenCV. Non sono attribuiti risultati di equivalenza o prestazioni a run online non disponibili.

La preparazione del dataset PCA sarà trattata nella sezione 3.7: non si assume un unico artefatto numerico di PCA condiviso da tutti gli scenari. La sezione attuale descrive la trasformazione e la sua applicazione, senza riportare percentuali di varianza o risultati di separabilità. Sono richiamate le fonti bibliografiche già presenti `Oquab2023DINOv2` e `Jolliffe2016PCA`.

La scaletta nascosta inizia ora dalla sezione 3.5. Non sono state modificate le impostazioni tipografiche o le sorgenti del progetto robotico. La compilazione e la lettura di pagine rappresentative del PDF verificano formule, placeholder e rimandi.

La verifica finale è stata eseguita con una compilazione separata in una copia pulita, evitando interferenze sui file ausiliari del progetto. Il PDF verificato è stato aggiornato nella directory del manoscritto. La sezione 3.4 occupa le pagine numerate **34–40**, circa sette pagine; il capitolo finora scritto si estende da 25 a 40. Non risultano errori di compilazione, citazioni o rimandi irrisolti, avvisi BibTeX o debordamenti nel capitolo 3. Controllate visivamente proiezione, formule geometriche, tabella e figura della memoria; la formula dell'età è mantenuta insieme alla sua spiegazione. Nell'intero capitolo sono ora presenti sei placeholder di figure e due di tabelle.

## Sezione 3.5 — 11 novembre 2026

Scritta **Definizione del compito di reinforcement learning**, con le quattro sottosezioni previste: osservazioni, azioni, ricompensa e gestione degli episodi. Inseriti sei equazioni, due placeholder di tabelle (osservazione e reward) e uno di figura (ciclo della missione). I nomi delle versioni restano fuori dal manoscritto; le differenze necessarie sono descritte per scenario.

### Controllo delle implementazioni

Letti `config.py`, `env.py` e `actions.py` dei quattro riferimenti concordati. Per l'allineamento e la terminazione sono stati consultati anche `alignment.py`, `rewards.py`, `falls.py`, `missions.py` e la raccolta delle transizioni in `replay.py`. Il trattamento degli stati terminali è stato riscontrato in `sac.py` e, per lo scenario speculare, in `ppo.py` e `train.py`.

L'actor riceve 15 valori compatti e 3240 valori della mappa, per un totale di 3255. Goal e griglia sono nel riferimento allineato allo yaw; velocità e gravità sono nel riferimento solidale al corpo. Il tempo residuo include l'allineamento, dove previsto. Le quantità sono descritte prima della normalizzazione, riservata alla sezione 3.6.

Le azioni normalizzate sono convertite in vx 0–1,6 m/s, vy nullo e wz ±1,5 rad/s. È esplicitata la differenza fra azione numerica zero e arresto: per vx l'arresto corrisponde a −1, mentre zero richiede 0,8 m/s.

I parametri sono stati verificati anche leggendo gli AST delle dataclass, senza importare o avviare il simulatore:

| Parametro | Speculare | Esteso e ambienti Re |
| --- | --- | --- |
| Progresso, punti/m | 3 | 30 |
| Penalità temporale per intervallo | 0,02 | 0,02 |
| Energia di riferimento, J | 100 | 100 |
| Peso energetico | 1 | 1 |
| Bonus di successo | 150 | 1000 |
| Penalità di fallimento | 300 | 1000 |
| Orientamento | Assente | Zero fino a 110°, lineare fino a −2 punti/s a 180° |

Il termine di orientamento riguarda il movimento controllato dall'actor. L'allineamento iniziale dei riferimenti SAC comanda vx=0 e wz proporzionale all'errore, con guadagno 1,5 s⁻¹, limite 0,8 rad/s, tolleranza 10° per tre controlli consecutivi e durata massima 12 s. Queste transizioni non vengono attribuite all'actor; il loro tempo e lavoro restano nel bilancio dell'episodio.

Il successo usa un raggio di 0,55 m e non impone yaw finale. Caduta e uscita dai confini hanno precedenza sul successo. La mappa speculare ha una scadenza fissa di 40 s; gli altri riferimenti usano clip(15 + 4 d₀, 30, 90) secondi, convertiti per eccesso in intervalli interi di 0,2 s. La scadenza è un terminale del compito, come successo e altri fallimenti, e interrompe il valore futuro nel target di apprendimento.

È stata mantenuta una distinzione importante fra i criteri di caduta: nel riferimento speculare sono altezza rispetto al terreno <0,25 m, inclinazione >60° o quota non valida entro i confini, valutate al passo di navigazione. Nei riferimenti esteso e Re si usa la persistenza per un secondo di inclinazione ≥90° oppure contatto tronco–terreno >1 N, con campionamento fisico e segnalazione mantenuta fino al reset. In questi ultimi l'altezza è soltanto diagnostica. Non viene descritta una regola unica diversa da quelle implementate.

Per coerenza sono state precisate due frasi delle sezioni precedenti: il termine di orientamento della 3.1 è condizionato allo scenario; il sensore di contatto del tronco della 3.2 è riferito agli ambienti che usano tale criterio. Non sono state modificate altre parti del metodo o le configurazioni robotiche.

### Verifica editoriale

La sezione è compilata in una copia separata e il PDF verificato viene aggiornato nel progetto. Il titolo lungo è spezzato su due righe mantenendo invariata la voce di indice e i parametri tipografici. Verificati formule, pagine rappresentative e rimandi. Non risultano errori, citazioni o rimandi irrisolti, avvisi BibTeX o debordamenti nel capitolo 3.

La sezione 3.5 occupa le pagine numerate **41–46**, incluso il placeholder finale; il capitolo finora scritto si estende da 25 a 46. I placeholder dell'intero capitolo sono sette figure e quattro tabelle. La scaletta nascosta inizia dalla 3.6. L'impaginazione delle ultime figure potrà assestarsi con la stesura delle sezioni successive.

## Sezione 3.6 — 11 novembre 2026

Scritta **Architettura delle reti e fusione delle informazioni**, nelle tre sottosezioni concordate: organizzazione e normalizzazione degli ingressi; rete della policy e fusione; critic e informazioni privilegiate. Aggiunti quattro equazioni, un placeholder di figura per le architetture e uno di tabella per le dimensioni delle reti. La scaletta nascosta inizia ora dalla 3.7. Le differenze vengono presentate per scenario, senza nomi delle versioni nel manoscritto.

### Riscontri nelle implementazioni

Letti `networks.py` e `config.py` dei quattro riferimenti. Gli AST dei file `networks.py`, escludendo gli import che cambiano namespace, coincidono fra scenario esteso, entrambi i riferimenti Re e copia dei sorgenti della run estesa `general_map_test7_sac_20261029_163507_402204`. Consultati anche `privileged.py`, `ppo.py`, `sac.py`, `replay.py` e i passaggi pertinenti di `train.py`.

- La trasformazione adattiva usa media e varianza progressive, epsilon 10⁻⁶ e clipping a ±10. Nello scenario speculare comprende goal XY, velocità lineari/angolari e potenza (9 canali); nei riferimenti SAC soltanto velocità e potenza (7 canali), mentre il goal viene diviso per 10 m senza centratura o clipping adattivo.
- Le statistiche visive (32 canali) e geometriche (4) sono condivise fra le posizioni delle celle e aggiornate solo con misure valide. Le maschere vengono riapplicate dopo la normalizzazione; le età sono limitate a [0,1] e mascherate. Gravità proiettata, azione precedente, tempo residuo e indicatori di validità mantengono la loro scala.
- PPO mantiene le statistiche fisse durante raccolta e ottimizzazione del rollout e le aggiorna dopo. SAC conserva osservazioni grezze nel replay, aggiorna le statistiche con i nuovi campioni accettati e applica la trasformazione corrente durante l'ottimizzazione. L'esecuzione e la valutazione usano le statistiche salvate senza aggiornarle.
- Actor PPO: 3255→512→256→128→64→2, fusione prima del primo strato; due log-deviazioni standard globali apprese, inizializzate a −0,5 e limitate a [−5,2]. Critic PPO separato: 3279→512→256→128→64→2, con valore di navigazione e valore del costo energetico positivo. I vantaggi vengono combinati come A_nav − w_E A_energia, poi normalizzati.
- Actor SAC: stato 15→64→32, mappa 3240→1024→512→256→128, fusione 160→128→64→4. Produce due medie e due log-deviazioni standard dipendenti dall'osservazione, limitate a [−20,2]. Strati nascosti ELU in entrambe le architetture. La gaussiana latente è trasformata con tanh; la versione deterministica applica tanh alla sua media.
- Ogni critic SAC ha il ramo stato 41→64→32 (15 valori compatti + 24 privilegiati + azione corrente di dimensione 2), lo stesso dimensionamento del ramo mappa dell'actor e fusione 160→128→64→1. I due critic hanno pesi indipendenti ed entrambi stimano lo stesso obiettivo complessivo regolarizzato dall'entropia; non corrispondono alle due componenti del critic PPO. Sono presenti due copie obiettivo per l'addestramento.
- Il vettore privilegiato comprende coefficienti statici/dinamici effettivi dei contatti (8), indicatori di contatto (4), intensità delle forze (4) e componenti XY delle velocità tangenziali ai piedi (8). Le statistiche dei 20 canali continui escludono i piedi senza contatto; gli indicatori restano espliciti. La parte comune al critic e all'actor usa la stessa normalizzazione, mentre i pesi delle reti sono separati.

Il testo distingue la memoria percettiva esterna dalle reti feedforward e dai loro pesi, fissi durante la missione. DINOv2 e PCA restano a monte dell'actor; il modello esportato richiede le osservazioni già costruite. Citato `Pinto2018Asymmetric`, già presente in bibliografia, per l'impostazione asimmetrica. Non sono stati aggiunti risultati o affermazioni di trasferimento sul robot reale.

### Verifica editoriale

Compilazione riuscita in una copia pulita separata, con successivo aggiornamento del PDF nel progetto. Non risultano citazioni o rimandi irrisolti, avvisi BibTeX, oppure debordamenti e avvisi di impaginazione nel capitolo 3. Controllate visivamente quattro pagine rappresentative e ricontrollati titolo su due righe e citazione dopo la revisione finale. Restano gli avvisi del template estranei alla sezione.

La sezione 3.6 si estende sulle pagine numerate **46–50**, circa cinque pagine; la 46 contiene anche il placeholder finale della 3.5. Il capitolo finora scritto va da 25 a 50 e comprende otto placeholder di figure e cinque di tabelle. Non sono stati modificati sorgenti robotici, impostazioni tipografiche o sezioni successive.

## Sezione 3.7 — 11 novembre 2026

Scritta **Procedura di addestramento**, nelle quattro sottosezioni previste: calibrazione e preparazione; aggiornamento delle reti; generazione delle missioni e curriculum; iperparametri e selezione del checkpoint. Aggiunti un placeholder di figura per il curriculum e uno di tabella per i parametri. Nessuna nuova equazione: si richiamano quelle della ricompensa, del vantaggio e della scadenza, lasciando i principi generali di PPO/SAC al capitolo 2. Citazioni già disponibili: `Schulman2017PPO` e `Haarnoja2018SAC`.

### Fonti e riscontri

Consultati `config.py`, launcher di training, `train.py`, `missions.py`, `prepare_map.py`, `evaluation.py`, `calibration.py`, `prepare_pca.py`, `ppo.py`, `sac.py`, `runtime_config.py`, `artifacts.py`, `layouts.py` e i metadati degli artefatti PCA. Per la mappa estesa gli AST di `missions.py`, `calibration.py`, `prepare_pca.py`, `evaluation.py`, `sac.py` e `prepare_map.py` coincidono con i sorgenti archiviati nella run `general_map_test7_sac_20261029_163507_402204`.

- **PCA speculare:** letti i metadati di `multi_terrain_test13/artifacts/pca32.npz`: 40000 campioni bilanciati, 10000 ciascuno per i codici 0, 1, 3, 4 (asfalto, scivoloso, ghiaia fine, rocce), seed 123, dataset `scripted_small_mirrored_boards`. È una calibrazione già pronta, proveniente da una raccolta preparatoria, non una nuova PCA sui sette materiali della mappa estesa. Il testo distingue questo insieme dalle missioni PPO e non introduce la cronologia degli artefatti.
- **PCA estesa:** 96 viste di calibrazione e 24 di controllo per materiale; posizioni della base divise in due sottoinsiemi di candidati di training, esclusi quelli di valutazione RL. Patch etichettate per bilanciare la raccolta mediante proiezione sulla mappa, scartando i bordi. Artefatto locale: 70000 descrittori di fit (10000 per materiale) e 2469 campioni di controllo per materiale. Nel testo non si riportano percentuali di varianza conservata o risultati di generalizzazione. I due ambienti Re prevedono la procedura analoga sulle proprie due superfici.
- **PPO:** 32 ambienti, 128 decisioni per rollout ordinario, 4096 transizioni, minibatch 256, massimo 10 epoche, Adam 3e-4, gamma 0,999, lambda GAE 0,95, clipping 0,2, soglia KL 0,02, entropia 0,005, pesi delle due perdite di valore 0,5 ciascuno, norma del gradiente limitata a 0,5. La raccolta finale viene accorciata per rispettare il budget. Questi valori e il budget di 100000 sono confermati da configurazione e argomenti salvati nella run `test13_ppo_20261023_122929_726370`.
- **SAC esteso:** warmup di almeno 4096 transizioni con azioni uniformi, replay 100000, minibatch 256, Adam 3e-4, gamma 0,999, tau 0,005, alpha iniziale 1 e target entropy −2. Un aggiornamento per passo parallelo utile. Configurazione e argomenti della run locale confermano 32 ambienti e budget 200000. Il budget descrive il riferimento metodologico, non tutte le eventuali continuazioni del training.
- **SAC Re:** i default e i launcher prevedono 128 ambienti, 16 aggiornamenti per passo parallelo utile e budget 1500000. Sono presentati come configurazione prevista, senza attribuire questi valori a run online non disponibili. Il codice corrente offre anche override e modalità di distribuzione delle missioni; il manoscritto descrive il curriculum configurato come riferimento, senza una rassegna delle opzioni. Gli argomenti effettivi delle run online andranno verificati quando disponibili.

### Curriculum e valutazioni

La mappa speculare usa due disposizioni contemporaneamente, bilanciate fra gli ambienti, senza curriculum. Per la mappa estesa sono descritte le cinque fasi 1,5–3, 3–6, 3–6, 6–12 e 12–20 m, distinguendo l'ampliamento dei punti ammessi dall'aumento della distanza. Prima fase nella stessa cella di asfalto; seconda con esclusione di scale e ostacoli dagli estremi; esclusioni generali permanenti: scivoloso alle partenze e ai goal, scale ai goal. I controlli locali dell'archivio dei candidati sono margine 0,65 m, quadrato di appoggio con semilato 0,55 m, escursione delle quote ≤0,16 m. Circa un quinto dei punti di ogni cella è riservato alla valutazione. Il campionamento non pianifica il percorso né applica un filtro di raggiungibilità globale.

I riferimenti Re usano invece quattro fasi, 1,5–3, 3–6, 6–10, 10–14 m. Per la resistenza al moto si parte con estremi sul grigio e poi si include il verde; per la superficie scivolosa gli estremi restano sul grigio. Non impongono la stessa cella nella prima fase. Nei tre riferimenti SAC con curriculum: promozione con almeno 80% di successo per due valutazioni consecutive, azzeramento della sequenza sotto soglia, nessuna retrocessione; dalla seconda fase 25% di missioni provenienti dalle fasi precedenti, scelte uniformemente. Il testo separa fasi previste e fasi raggiunte dalla singola esecuzione.

Valutazioni deterministiche con un episodio per ambiente: ogni 100000 transizioni nel riferimento speculare, ogni 10000 in quello esteso; per i Re l'intervallo configurato è 10000 nelle prime due fasi e 30000 nelle ultime due. Dati esclusi da training e aggiornamento delle normalizzazioni. Selezione per successo, poi minore lavoro medio sui successi; nei curriculum, salvataggio dei migliori per fase e priorità alla fase più avanzata per il migliore complessivo. Per SAC la ripresa richiede il replay corrispondente; la simulazione viene reinizializzata. Il protocollo sperimentale finale resta al capitolo 4.

### Verifica editoriale

Compilazione riuscita in una copia pulita separata. Controllati impaginazione di pagine rappresentative, elenco delle fasi, placeholder e rimandi. Il titolo della 3.7.1 è disposto su due righe senza spezzare parole. Non risultano citazioni o rimandi irrisolti, avvisi BibTeX o avvisi di debordamento/impaginazione nel capitolo 3. Restano gli avvisi preesistenti del template.

La sezione si estende sulle pagine numerate **50–56**, circa sei pagine e mezzo: la 50 contiene ancora la conclusione della 3.6. Il capitolo finora scritto va da 25 a 56, con nove placeholder di figure e sei di tabelle. La 3.8 rimane una scaletta esclusa dal PDF. Nessuna modifica alle sorgenti robotiche o alle impostazioni tipografiche. Il PDF verificato viene aggiornato nel progetto dopo la compilazione separata.

## Sezione 3.8 — 11 novembre 2026

Scritta **Esecuzione della policy e integrazione sul robot**, con le due sottosezioni concordate: pipeline di inferenza e gestione dei comandi; integrazione con la piattaforma reale. Il testo distingue l'esecuzione implementata in simulazione dall'interfaccia reale ancora da definire. Aggiunto un placeholder di tabella per sensori, computer, sorgenti di stato e misura energetica, interfaccia dei comandi, frequenze e sincronizzazione. Non sono state aggiunte equazioni o figure che ripetano le architetture già illustrate.

### Chiarimento dell'autrice

Alla domanda sulla pipeline reale l'autrice ha risposto esplicitamente: **«È ancora da definire: lascia un placeholder per i dettagli.»** La sottosezione 3.8.2 resta quindi una breve descrizione delle interfacce richieste, seguita dal placeholder concordato. Non attribuire al progetto una camera fisica specifica, un computer, un sistema di localizzazione, un driver o controllore reale, una misura articolare disponibile o frequenze reali già validate. L'integrazione effettiva e le prove andranno descritte quando saranno definite; questo punto non richiede ulteriori chiarimenti finché non arrivano nuove informazioni.

### Sorgenti e contenuti verificati

Consultati `play.py`, `networks.py`, `env.py`, `evaluation.py`, `actions.py`, `alignment.py`, `locomotion.py` e l'esportazione in `train.py` dei riferimenti di simulazione; lette anche le descrizioni degli ambienti Re. Cercate interfacce reali nel workspace. Il nodo ROS 2 `ros2_ws/src/spot_rl_controller/spot_rl_controller/rl_controller_node.py` usa SB3 PPO, osservazioni e azioni differenti; non è assunto come pipeline adottata dagli actor della tesi. Non viene ricostruita nel manoscritto una cronologia di queste implementazioni.

- I programmi di riproduzione caricano il checkpoint completo, ripristinano policy e normalizzazioni, recuperano la PCA e verificano compatibilità di configurazione, modello visivo e asset.
- `ExportedActor` contiene normalizzazione e actor, riceve 3255 valori grezzi e restituisce due azioni normalizzate deterministiche tramite tanh della media latente. DINOv2, PCA e memoria restano esterni. La normalizzazione non deve essere ripetuta dal chiamante.
- L'esportazione automatica a fine training deriva dall'actor corrente, senza caricare automaticamente il modello best. Il testo richiede di associare l'artefatto al checkpoint di origine e rinvia al capitolo 4 per quello utilizzato negli esperimenti.
- Il ciclo continua ad aggiornare memoria, validità, età, stato e potenza filtrata, mantenendo fissi i pesi. Per scegliere l'azione non sono richiesti critic o dati privilegiati, anche quando il programma di valutazione ricostruisce l'oggetto completo.
- Il modulo di allineamento può sostituire il comando; nell'osservazione successiva compare l'azione effettivamente applicata. Conversione in velocità, locomozione e gestione della missione restano esterne all'actor.
- Acquisizione dopo sincronizzazione della posa della camera, intervallo di 0,2 s simulati, registrazione delle metriche prima del reset. Il testo distingue la frequenza del ciclo simulato da quella da misurare sul sistema reale.

### Verifica editoriale

Compilazione riuscita in una copia pulita separata; verificato il PDF su tutte le pagine della sezione. Il placeholder ha margini interni adeguati ed è mantenuto insieme alla sottosezione reale. Non risultano citazioni o rimandi irrisolti, avvisi BibTeX o avvisi di debordamento/impaginazione nel capitolo 3. Restano gli avvisi del template preesistenti.

La sezione 3.8 occupa parte della pagina **56** e le pagine **57–58**, incluso il placeholder. Il capitolo 3 si estende ora da 25 a 58, circa **34 pagine**: tutte le otto sezioni sono visibili, con nove placeholder di figure, sette di tabelle e ventiquattro equazioni numerate. Rimangono da completare i contenuti segnaposto, in particolare l'integrazione reale. Verificato che le sezioni 3.1–3.7 sono identiche alla copia precedente a questo intervento. Nessuna modifica alle sorgenti robotiche, alle impostazioni tipografiche o agli altri capitoli. Il PDF verificato viene aggiornato nella directory del manoscritto.

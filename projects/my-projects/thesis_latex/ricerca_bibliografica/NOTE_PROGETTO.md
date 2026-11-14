# Appunti condivisi per la stesura

Stato al 6 novembre 2026. Questo file registra il perimetro concordato e le osservazioni preliminari sul codice. Non è un capitolo della tesi e non stabilisce quale checkpoint verrà presentato.

## Obiettivo e percorso di lavoro

Tesi magistrale in ingegneria robotica, versione italiana, oltre 80 pagine, usando il template in `thesis_latex/template_tesi_RoboticsLaTeX2/RoboticsLaTeX2`. Mantenere come base i cinque capitoli esistenti: introduzione, stato dell’arte, metodologia, risultati, conclusioni. La struttura interna è ancora modificabile.

L’obiettivo dichiarato è apprendere una policy di navigazione per Spot che raggiunga un goal assegnato scegliendo tragitto e velocità in funzione del costo energetico. Le informazioni visive devono permettere di anticipare le conseguenze dell’interazione con il terreno, senza richiedere una classificazione esplicita dei materiali. L’informazione disponibile è locale e non consente di assumere la conoscenza dell’intero percorso fino al goal.

Ordine concordato:

1. Individuare circa 40 riferimenti e predisporre la bibliografia: prima selezione consegnata in questa cartella.
2. Rivedere la struttura dei capitoli alla luce dei riferimenti.
3. Scrivere una prima versione italiana dei capitoli 1 e 2.
4. Scrivere il capitolo 3 sulla versione del metodo che sarà effettivamente adottata.
5. Definire il capitolo 4, gli esperimenti e i dati da registrare; successivamente discutere risultati e conclusioni.

La prima selezione non equivale a una revisione critica integrale dei 40 lavori. Prima di scrivere affermazioni specifiche occorre consultare le sezioni pertinenti delle fonti primarie. Il numero di riferimenti è un’indicazione di ampiezza, non un obiettivo che impone citazioni superflue.

## Stato del progetto dichiarato dall’autore

- Esiste una policy per lo scenario small specular map; considerare la configurazione più recente, senza raccontare la successione delle prove preliminari.
- Esiste una policy per big general map; la scelta finale fra le configurazioni 7, 8, 9 e 10 resta aperta.
- La policy per le prove su Spot reale deve ancora essere addestrata. Non presentare risultati reali come già ottenuti.
- Nel testo descrivere il sistema finale. I confronti sperimentali potranno includere baseline o ablation deliberate, motivate da una domanda scientifica; non trasformare la cronologia del codice in capitoli della tesi.

## Riscontri preliminari nell’implementazione

Sono stati esaminati soprattutto `multi_terrain_test13` come candidato più recente per le mappe speculari e i README/configurazioni `train_general_big_map_test7_SAC`–`test10_SAC`. Il README generale include anche prove iniziali e non va assunto come descrizione del sistema finale. Le sorgenti attuali possono differire da quelle conservate nel checkpoint: il capitolo 3 dovrà riferirsi ai sorgenti e alla configurazione della run selezionata.

| Aspetto | Riscontro | Conseguenza per la scrittura |
|---|---|---|
| Livello di controllo | La policy di navigazione produce comandi di velocità; un controllore locomotorio preaddestrato gestisce i giunti. | Definire il contributo come navigazione energetica condizionata dal controllore adottato. |
| Azioni | Nelle configurazioni esaminate si comandano velocità longitudinale e angolare; la velocità laterale è fissata a zero. | Evitare di descrivere un controllo diretto delle coppie o una scelta esplicita dell’andatura. |
| Percezione | RGB-D, DINOv2 congelato, PCA e geometria locale alimentano una gridmap per l’actor. | Spiegare feature, proiezione, riduzione dimensionale e geometria; non parlare di classificatore dei sette terreni. |
| Propriocezione nell’actor | Velocità del corpo, gravità proiettata e potenza meccanica filtrata, oltre a goal, azione precedente e tempo residuo. | Non sostenere che ogni posizione e velocità articolare sia fornita direttamente alla rete di navigazione: i giunti entrano nel calcolo energetico. |
| Memoria della missione | La mappa conserva osservazioni visive e statistiche geometriche, con validità ed età; si azzera al reset. | Non chiamarla mappa esplicita di energia per terreno o memoria permanente fra missioni. |
| Apprendimento | L’associazione fra rappresentazione del terreno e decisione energetica viene ottimizzata nel training RL. | Una policy congelata non diventa un algoritmo di apprendimento online solo perché riceve propriocezione e una mappa persistente nell’episodio. |
| Informazione privilegiata | Il critic riceve ulteriori segnali di contatto/attrito; l’actor non riceve le stesse variabili. | Distinguere training asimmetrico e osservazioni effettivamente utilizzabili sul robot. |
| Posa per la mappa | La registrazione usa la posa simulata. | Il realismo della localizzazione e la robustezza a deriva odometrica non sono già dimostrati. |
| Scenari | Il generatore e le configurazioni possono usare identità dei materiali per calibrazione, missioni o diagnostica. | “Senza classificazione esplicita” si riferisce agli input/decisioni dell’actor; non implica assenza assoluta di etichette in ogni fase del progetto. |

Fonti locali: [potenza e osservazioni nello scenario speculare](../../spot_rl_energy_navigation/multi_terrain_test13/env.py), [memoria della mappa](../../spot_rl_energy_navigation/multi_terrain_test13/memory.py), [configurazione dello scenario speculare](../../spot_rl_energy_navigation/multi_terrain_test13/config.py), [controllore locomotorio](../../spot_rl_energy_navigation/locomotion.py), [sistema SAC su mappa estesa](../../spot_rl_energy_navigation/train_general_big_map_test10_SAC/README.md). Sono evidenze di implementazione, non risultati sperimentali verificati in questa analisi.

## Grandezza energetica e formulazione del problema

Nel codice esaminato:

\[
P_{\mathrm{abs}}(t)=\sum_j |\tau_j(t)\dot q_j(t)|,
\qquad
E_{\mathrm{abs}}=\int P_{\mathrm{abs}}(t)\,dt.
\]

Le coppie sono ottenute dalle forze articolari proiettate del solver PhysX; l’integrazione avviene con il metodo dei trapezi ai passi della fisica. La grandezza è una stima di **lavoro meccanico assoluto ai giunti**: comprende il valore assoluto anche del lavoro negativo. È diversa da lavoro meccanico netto, solo lavoro positivo e consumo elettrico della batteria. Non include automaticamente rendimento degli attuatori, dissipazioni, elettronica di bordo o rigenerazione. Per il robot reale occorre ancora definire la misura effettivamente disponibile e il suo rapporto con questa metrica.

La reward contiene anche progresso, tempo, successo, fallimento ed eventualmente orientamento. Perciò il problema effettivamente ottimizzato è un compromesso fra raggiungimento del goal, affidabilità, durata e costo energetico. Non è sufficiente la presenza della penalità energetica per dimostrare una minimizzazione pura dei Joule o una traiettoria ottima.

Anche “scegliere il miglior percorso visibile” è un obiettivo da valutare, non una garanzia dell’algoritmo. Una formulazione prudente per la domanda di ricerca è:

> In quale misura una policy di navigazione che combina rappresentazioni visive locali e feedback propriocettivo riesce a ridurre il lavoro meccanico necessario per raggiungere un obiettivo, mantenendo un adeguato tasso di successo, rispetto a strategie di riferimento?

Questa domanda non richiede una classificazione manuale dei terreni e ammette la disponibilità di informazione parziale. I riferimenti selezionati mostrano precedenti su tutti questi ingredienti: il contributo preciso andrà formulato sull’architettura effettiva e sulle verifiche svolte, senza dichiarazioni di primato non dimostrate.

## Elementi da fissare nelle prossime fasi

La scelta bibliografica può essere discussa già ora. Per metodologia ed esperimenti serviranno poi: checkpoint e configurazione congelati per ciascuno scenario; modello esatto della camera e sensori del robot reale; definizione operativa della misura energetica reale; comportamento della memoria fra missioni; baseline e condizioni di valutazione. Le sette tipologie di terreno del simulatore non dimostrano da sole generalizzazione a nuovi terreni.

Per il confronto futuro distinguere mappe note con nuove coppie start/goal, nuove disposizioni dei materiali, materiali non visti e trasferimento sul robot reale. Sono verifiche diverse. L’assenza di etichette nell’actor non dimostra automaticamente generalizzazione.

I dettagli del protocollo e del logging del capitolo 4 restano alla fase concordata, dopo una prima versione dei capitoli 1–3.

## Aggiornamento per il capitolo 3 — 10 novembre 2026

L'autrice ha indicato `train_general_big_map_test7_SAC` come configurazione di riferimento per descrivere lo scenario esteso nel capitolo 3. Questa indicazione sostituisce, per la stesura metodologica, la precedente scelta aperta fra più configurazioni. Nel manoscritto si descrive il sistema adottato senza usare i nomi delle versioni o confrontarlo con tentativi precedenti.

La stesura procede una sezione alla volta. La proposta di struttura contiene otto sezioni, unendo piattaforma e architettura del sistema, con una lunghezza indicativa iniziale di circa 32 pagine. È richiesta una spiegazione approfondita del curriculum: motivazioni, generazione delle missioni, fasi, criteri di avanzamento e riproposizione delle missioni più semplici. La sequenza delle fasi dell'addestramento non va confusa con la cronologia delle versioni del software.

Il codice di riferimento prevede cinque fasi del curriculum, con distanze 1,5–3, 3–6, 3–6, 6–12 e 12–20 m. Le prime tre modificano anche le condizioni di campionamento dei punti di partenza e arrivo. Le esclusioni riguardano gli endpoint e non prescrivono il percorso; le esclusioni generali restano attive anche nelle fasi avanzate. Si avanza con almeno l'80% di successo in due valutazioni consecutive; dalla seconda fase il 25% delle missioni è campionato da fasi precedenti. Questi sono criteri implementati, non evidenze che una specifica run abbia completato tutte le fasi.

Per i parametri numerici di una run, prima della relativa sezione verificare la configurazione salvata e gli argomenti del launcher: il codice permette di sovrascrivere budget, parallelismo e numero di aggiornamenti senza modificare i default di `config.py`. La camera e l'integrazione effettiva sulla piattaforma reale restano da precisare nella sezione pertinente.

## Riferimenti confermati per i tre scenari — 11 novembre 2026

L'utente ha confermato i seguenti riferimenti di implementazione:

| Scenario | Directory di riferimento |
| --- | --- |
| Mappa piccola speculare | `spot_rl_energy_navigation/multi_terrain_test13/` |
| Mappa estesa | `spot_rl_energy_navigation/train_general_big_map_test7_SAC/` |
| Modelli destinati alle prove sul robot reale | `spot_rl_energy_navigation/train_general_big_map_test12_SAC_Re/` e `train_general_big_map_test13_SAC_Re/` |

I modelli con suffisso Re sono in addestramento su una GPU online alla data della comunicazione. I risultati e gli artefatti aggiornati delle run potrebbero non essere ancora disponibili localmente. La presenza del codice non dimostra il completamento dell'addestramento o di esperimenti sul robot reale.

Questi riferimenti guidano la verifica del metodo nel capitolo 3; i nomi delle directory non vengono riportati nel testo del manoscritto. Il capitolo continua a essere scritto una sezione alla volta. La specifica della misura sul robot reale resta distinta dal calcolo del lavoro meccanico in simulazione.

Per la sezione 3.3 sono stati letti `env.py` e `config.py` nei quattro riferimenti. Coincidono calcolo di potenza assoluta, integrazione ai trapezi, passo fisico (0,005 s), intervallo di navigazione (0,2 s), filtro (1 s), riferimento energetico (100 J) e peso energetico (1). Negli ambienti con allineamento il suo lavoro è incluso nel totale della missione e registrato separatamente; la raccolta dei dati per l'actor esclude le transizioni controllate dall'allineamento. La preparazione iniziale della postura è esterna al conteggio degli episodi.

Le differenze fisiche degli ambienti Re dovranno essere descritte nella parte pertinente: nel modello con resistenza al moto si applica una forza esterna al corpo, senza aggiungere un costo energetico artificiale; il modello con superficie scivolosa impiega invece coefficienti di attrito differenti. Per il calcolo della sezione 3.3 si usano in entrambi i casi i segnali articolari effettivamente prodotti dalla simulazione.

## Precisazioni durante la stesura della sezione 3.5 — 11 novembre 2026

Le configurazioni dei diversi scenari non hanno tutti i parametri del compito in comune. Per la ricompensa, il riferimento speculare usa progresso 3 punti/m, bonus 150, fallimento −300 e nessuna penalità di orientamento; gli altri tre riferimenti usano progresso 30, bonus 1000, fallimento −1000 e la penalità di orientamento documentata nel testo. Restano comuni spazio dell'actor, comandi, passo temporale e normalizzazione energetica. Non trasformare queste differenze per scenario in una cronologia delle versioni.

Anche i criteri di caduta sono distinti: altezza/inclinazione/quota nel riferimento speculare, persistenza di inclinazione o contatto del tronco nei riferimenti esteso e Re. Per i dettagli e la provenienza vedere `NOTE_STESURA_CAPITOLO3.md`, sezione 3.5. Il trasferimento effettivo sul robot reale e i suoi criteri operativi andranno descritti nella sezione 3.8 sulla base del sistema utilizzato, senza dedurli dalla sola presenza degli ambienti di training Re.

## Architetture verificate per la sezione 3.6 — 11 novembre 2026

La sezione 3.6 è scritta e compilata, pagine 46–50, con tre sottosezioni, quattro equazioni e due placeholder (figura delle architetture e tabella delle dimensioni). I dettagli e le sorgenti sono annotati in `NOTE_STESURA_CAPITOLO3.md`. La scaletta nascosta prosegue dalla 3.7.

Lo scenario speculare usa PPO con fusione degli ingressi prima della MLP e un critic a due uscite per navigazione e costo energetico. Lo scenario esteso e gli ambienti destinati alle prove reali usano SAC con rami separati per stato e mappa e due critic indipendenti sullo stesso obiettivo complessivo. Queste due strutture dei critic non vanno confuse. I file delle reti dei tre riferimenti SAC e della copia salvata nella run estesa coincidono a livello di AST una volta esclusi gli import.

Resta una differenza di normalizzazione del goal: media/varianza adattive nello scenario speculare, scala fissa di 10 m nei riferimenti SAC. Sono comuni la normalizzazione delle misure valide per canale, la riapplicazione delle maschere e la disponibilità di 24 valori privilegiati soltanto ai critic. Per l'esecuzione servono actor e statistiche salvate; memoria percettiva ed estrazione delle feature rimangono nella pipeline esterna. La successiva stesura della 3.7 dovrà rispettare queste scelte senza ripetere la trattazione delle reti.

## Procedura di addestramento, sezione 3.7 — 11 novembre 2026

La sezione 3.7 è scritta e compilata, pagine 50–56, con le quattro sottosezioni previste e due placeholder (curriculum e iperparametri). La 3.8 resta esclusa dal PDF. Fonti, parametri e verifiche sono documentati in `NOTE_STESURA_CAPITOLO3.md`.

Confermati dai file delle run locali i riferimenti di 32 ambienti e budget di 100000 transizioni per la mappa speculare e 200000 per quella estesa. I launcher Re prevedono 128 ambienti, 16 aggiornamenti per passo parallelo utile e 1500000 transizioni: nel manoscritto sono parametri di riferimento dell'addestramento predisposto, da confrontare con gli argomenti delle run online quando disponibili.

I curriculum non sono identici: cinque fasi nella mappa estesa (1,5–3, 3–6, 3–6, 6–12, 12–20 m), quattro negli ambienti Re (1,5–3, 3–6, 6–10, 10–14 m). Per la resistenza al moto la prima fase ha estremi sul grigio e le altre ammettono anche il verde; per l'ambiente scivoloso gli estremi restano sul grigio. Sono comuni promozione all'80% per due valutazioni consecutive e riproposizione al 25% delle fasi precedenti. Queste regole riguardano gli estremi e non prescrivono il tragitto.

Anche gli artefatti PCA hanno calibrazioni distinte: quello speculare usa 40000 descrittori bilanciati fra quattro superfici; quello esteso 70000 fra sette superfici, con viste separate per il controllo. Il numero di materiali della calibrazione non coincide necessariamente con quelli delle missioni RL. Le etichette servono alla preparazione del dataset e non sono ingressi dell'actor. I modelli per le prove reali prevedono calibrazioni sulle due superfici delle rispettive scene.

## Esecuzione della policy, sezione 3.8 — 11 novembre 2026

Scritta e compilata la sezione 3.8, pagine 56–58, con descrizione dell'inferenza verificata nel codice e un placeholder di tabella per l'integrazione reale. L'autrice ha confermato esplicitamente che **la pipeline sul robot reale è ancora da definire e che i dettagli devono rimanere in un placeholder**. Non dedurre componenti hardware, posa, misura energetica o controllore effettivi dalla presenza degli ambienti Re o dei nodi ROS 2 nel workspace.

L'actor TorchScript include la normalizzazione e riceve l'osservazione grezza; DINOv2, PCA e memoria appartengono alla pipeline esterna. L'esportazione a fine training usa l'actor corrente e non seleziona automaticamente il checkpoint best. Frequenza simulata e frequenza effettiva sulla piattaforma reale rimangono distinte. Per sorgenti e riscontri vedere `NOTE_STESURA_CAPITOLO3.md`.

Il capitolo 3 ha ora tutte le otto sezioni visibili e si estende da pagina 25 a 58, circa 34 pagine. Si tratta di una prima stesura con contenuti segnaposto, inclusa l'integrazione reale concordata; protocollo sperimentale e capitolo 4 restano alla fase successiva prevista dal piano dell'autrice.

## Proposta separata del protocollo sperimentale — 11 novembre 2026

Su richiesta dell'autrice, creata una proposta autonoma da condividere con il relatore in thesis_latex/protocollo_sperimentale/, con sorgente LaTeX, PDF, piano tabellare delle missioni da completare e nota dei riscontri preliminari. Non sono stati modificati capitolo 4, capitoli già scritti o codice robotico.

Il nucleo proposto comprende 74 esecuzioni di navigazione simulate (20 speculari, 36 estese, 18 nella scena simulata per il reale) e 30 reali, oltre a sei attraversamenti di caratterizzazione reali. La caratterizzazione dei sette terreni simulati può riusare dati compatibili o richiedere 63 prove; l'eventuale caratterizzazione delle due superfici di S3 ne aggiunge sei. Sono numeri di una proposta ancora da discutere, non risultati o nuovi esperimenti già eseguiti.

Per le mappe speculari è proposto il confronto con un ramo roccioso guidato, perché la retta attraversa la regione centrale scivolosa. Negli altri scenari il riferimento è un controllore diretto verso il goal, senza percezione del costo. Il risparmio viene calcolato sulle coppie che completano lo stesso compito, insieme ai conteggi di successo e degli altri esiti. Le figure possono illustrare esempi riusciti; i riepiloghi conservano tutte le ripetizioni previste.

Il benchmark guidato esistente usa mappe e reset diversi da quelli congelati della policy speculare e non va confrontato direttamente con essa senza adattamento. Le suite salvate degli ambienti Re fissano anche lo yaw: cambiare solo seed non crea nuove ripetizioni. I dettagli delle verifiche e la provenienza dei dati preliminari sono in RISCONTRI_PRELIMINARI.md.

I campi ignoti, soprattutto luogo, computer, camera, pipeline e misura energetica reali, sono lasciati DA DEFINIRE come richiesto. I controllori di confronto, il caricamento del nuovo piano e il logging completo richiederanno implementazione successiva. Non sono stati lanciati nuovi training o esperimenti.

## Riscrittura breve del piano sperimentale — 11 novembre 2026

L'autrice ha rifiutato la prima proposta, giudicandola troppo lunga e formale. La sostituisce una versione di tre pagine, semplice, schematica e scritta dalla prospettiva dell'autrice. La definizione di missione ed esecuzione è all'inizio, prima dello scopo. Sono eliminate formule, dettagli dei computer e la campagna obbligatoria di caratterizzazione. Il costo energetico è definito come lavoro meccanico assoluto ai giunti sia in simulazione sia nel reale.

Nuova sequenza: breve esempio binario PPO sulle mappe speculari; breve esempio SAC su sette terreni, ancora da identificare; valutazione principale sulla mappa 7×7; laboratorio grigio–verde; laboratorio grigio–nero. L'outdoor rimane solo una possibilità, senza protocollo definito.

I nuovi numeri proposti sono 30 missioni diverse in simulazione per ciascuna delle tre valutazioni principali (7×7, grigio–verde, grigio–nero): dieci da 1,5–3 m, dieci da 3–6 m e dieci da 6–12 m. Nel reale cinque missioni ripetute due volte per ciascuna delle due scene indoor, quindi dieci esecuzioni per scena. Il PPO speculare dispone già di dieci esecuzioni, cinque per disposizione. L'esempio SAC preliminare da nove o dieci prove resta da identificare, senza attribuire arbitrariamente a esso un risultato esistente.

Le missioni sono selezionate casualmente entro fasce di distanza e casistiche geometriche: tratto semplice, deviazione da ghiaccio o scale, alternative di costo, scelte successive. Nel nuovo piano start e goal della valutazione 7×7 sono esclusi sia dal ghiaccio sia dalle scale. Non riscrivere retroattivamente le esclusioni del training: il riferimento corrente consente partenze sulle scale.

L'autrice indica ZED X in simulazione e ZED 2 nel reale. La pipeline reale e la lettura dei segnali articolari restano da definire. Nel grigio–verde il modello simulato aumenta la resistenza al moto sul verde; il maggiore lavoro richiesto dal telo fisico va verificato. I cartelloni neri sono descritti come rappresentazione visiva della superficie scivolosa, non come ghiaccio reale. Il caso nero reale valuta soprattutto evitamento visivo e raggiungimento del goal.

Per una prima misura del risparmio sono proposti cinque confronti diretti aggiuntivi simulati (due 7×7, tre grigio–verde) e due reali grigio–verde. Si confrontano le energie soltanto a compito completato da entrambi, mantenendo i conteggi di tutti gli esiti.

Aggiornati documento autonomo, PDF, CSV e LEGGIMI nella directory protocollo_sperimentale. I riscontri tecnici della prima proposta sono conservati e marcati come storici. Nessuna modifica ai capitoli della tesi o al codice robotico; nessuna nuova prova eseguita.

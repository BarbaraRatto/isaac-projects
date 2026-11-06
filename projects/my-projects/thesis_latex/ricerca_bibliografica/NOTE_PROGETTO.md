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

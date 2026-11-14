# Note sulla prima stesura del capitolo 2

Stesura del 9 novembre 2026. Il testo conserva le sei sezioni e le quattordici sottosezioni approvate. La richiesta di estensione riguarda il corpo del capitolo, escludendo la bibliografia comune del manoscritto.

Il capitolo cita i 40 lavori della selezione iniziale e tre fonti aggiuntive per i fondamenti. La trattazione privilegia il confronto fra oggetti appresi, segnali, obiettivi e livello di controllo; non presenta una classifica di prestazioni numeriche fra protocolli differenti. Il survey locale non pubblicato è escluso dalle citazioni.

## Integrazioni bibliografiche

- `Kaelbling1998POMDP`: [Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf), per la distinzione fra stato, osservazione e storia delle interazioni. DOI e metadati verificati nel registro Crossref.
- `Haarnoja2018Applications`: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), per la formulazione aggiornata di SAC e la temperatura appresa. Integra la fonte ICML già presente.
- `Tobin2017Randomization`: [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907), per il trasferimento delle rappresentazioni visive mediante variazione del rendering. Il capitolo distingue esplicitamente questo risultato dalla validazione della dinamica di contatto.

Le tre voci sono state aggiunte a `references.bib` senza rimuovere i riferimenti precedenti. La selezione iniziale di 40 lavori rimane documentata nei file del 6 novembre; queste sono integrazioni della stesura.

## Consultazione e limiti

Sono stati consultati i testi primari disponibili localmente e copie pubbliche da arXiv, PMLR, RSS e archivi degli autori. Per le singole affermazioni sono state verificate le parti pertinenti: abstract, descrizione del metodo o definizioni delle metriche. La consultazione non equivale a una revisione critica integrale di ogni articolo. Per TOP-Nav, Isaac Lab e il rapporto sulla valutazione sono state consultate anche le pagine ufficiali dei proceedings o di arXiv. I riferimenti di mappatura e quelli aggiuntivi sono stati verificati sulle fonti primarie; la definizione della PCA è stata riscontrata nel testo degli autori archiviato in PubMed Central.

Le interpretazioni che collegano i lavori alla tesi sono distinte dai risultati riportati dagli autori. In particolare, il testo separa costo energetico fisico, valore del controllore, anomalia visiva, successo di attraversamento e indicatori inerziali. Le equazioni introducono i fondamenti e non descrivono ancora la configurazione finale del progetto.

## Elementi grafici

Come richiesto, sono presenti soltanto placeholder per due figure concettuali e una tabella comparativa. I contenuti della tabella non sono stati sostituiti da dati inventati; il confronto è già sviluppato nel testo. Le figure previste riguardano rappresentazioni del terreno e meccanismi che collegano percezione, esperienza e memoria.

## Verifica editoriale

La prima compilazione ha evidenziato la necessità della codifica T1 per alcuni nomi della bibliografia. Sono stati aggiunti `fontenc` con T1 e i font vettoriali Latin Modern al preambolo, mantenendo l'impostazione del template. La verifica finale controlla risoluzione delle citazioni e dei rimandi, integrità della struttura e lunghezza effettiva del capitolo. Gli avvisi relativi alle sezioni ancora vuote del capitolo 3 restano estranei alla stesura dello stato dell'arte.

## Riscrittura del 10 novembre 2026

Su richiesta dell'autrice, il capitolo è stato riscritto integralmente con un linguaggio più semplice. La revisione usa frasi più dirette, spiega i termini tecnici quando vengono introdotti e riduce le ripetizioni nei confronti fra i lavori. Un esempio numerico illustra la differenza fra costo per metro ed energia totale; i valori sono didattici e non sono risultati sperimentali.

La struttura resta composta da sei sezioni e quattordici sottosezioni. Sono conservati tutti i 43 riferimenti, le otto equazioni e i placeholder per due figure e una tabella. La revisione riguarda la scrittura e non aggiunge fonti o nuove verifiche bibliografiche rispetto a quelle del 9 novembre.

La compilazione del manoscritto conferma una riduzione da **19 a 16 pagine**, con il capitolo alle pagine numerate **9–24**, bibliografia esclusa. Non sono stati modificati caratteri, margini o interlinea. La compilazione termina senza errori; non risultano citazioni o rimandi irrisolti, avvisi BibTeX o debordamenti nel capitolo 2. Sono state controllate anche pagine rappresentative del PDF, comprese formule e placeholder.

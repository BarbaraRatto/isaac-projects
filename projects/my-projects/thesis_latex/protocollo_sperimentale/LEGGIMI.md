# Proposta di protocollo sperimentale

Il documento da condividere con il relatore è **proposta_protocollo_sperimentale.pdf**, autonomo rispetto alla tesi. La sorgente modificabile è proposta_protocollo_sperimentale.tex.

La versione breve segue l'ordine indicato dall'autrice: esempio PPO speculare; esempio SAC preliminare da identificare; valutazione SAC sulla mappa 7×7; laboratorio grigio–verde; laboratorio grigio–nero. Le prove outdoor restano soltanto una possibilità.

Sono proposte 30 missioni diverse in simulazione per ciascuno dei tre scenari principali: mappa 7×7, grigio–verde e grigio–nero. Nel reale sono proposte cinque missioni ripetute due volte per ciascuna delle due scene indoor: dieci esecuzioni per scena. Il PPO speculare ha già dieci esecuzioni disponibili; il breve esempio SAC richiede conferma della prova e del numero esatto di missioni.

**piano_missioni_da_completare.csv** riepiloga fasce di distanza e casi reali, con il numero di missioni e ripetizioni. Contiene anche i pochi confronti diretti aggiuntivi. Le coordinate sono ancora da scegliere. È uno schema della proposta, non un input già collegato ai programmi.

**RISCONTRI_PRELIMINARI.md** è una nota di lavoro: documenta i file consultati, le differenze fra mappe e checkpoint e gli adattamenti necessari per applicare il protocollo. Il manoscritto e il codice robotico non sono stati modificati.

Questa versione sostituisce il precedente piano di sette pagine. Non comprende formule, specifiche dei computer o campagne aggiuntive obbligatorie di caratterizzazione.

Per ricompilare il documento dalla sua directory:

    latexmk -pdf -interaction=nonstopmode -halt-on-error proposta_protocollo_sperimentale.tex

# Analisi del confronto in simulazione

Missioni riuscite per entrambe le varianti: [1, 2, 3, 5, 7, 8, 9].

| Metrica | Originale | Solo goal 100k |
|---|---:|---:|
| Successi su tutte le missioni | 7/9 | 8/9 |
| Percorso medio, successi comuni (m) | 12.17 | 7.11 |
| Energia media, successi comuni (J) | 2811.09 | 1596.12 |
| Durata media, successi comuni (s) | 16.17 | 9.40 |

Totali comprensivi di allineamento. Non confrontare il minor consumo di una caduta con quello di una missione completata.
I grafici includono la posizione terminale esatta e verificano start, goal e missioni corrispondenti.
Un solo confronto per missione: risultati descrittivi, non prova statistica né identificazione della causa storica del training.

# Confronto simulato: normalizzazione goal

| Episodio | Variante | Esito | Percorso m | Energia J | Durata s | Rotazione assoluta rad* |
|---|---|---|---:|---:|---:|---:|
| 1 | originale | goal | 11.97 | 2694 | 16.6 | 14.74 |
| 1 | solo goal 100k | goal | 5.18 | 1207 | 9.8 | 2.40 |
| 2 | originale | goal | 12.65 | 2862 | 16.4 | 14.28 |
| 2 | solo goal 100k | goal | 10.95 | 2405 | 12.4 | 8.34 |
| 3 | originale | goal | 16.10 | 3932 | 20.8 | 21.09 |
| 3 | solo goal 100k | goal | 4.72 | 1040 | 6.4 | 1.97 |
| 4 | originale | timeout | 22.29 | 5657 | 35.6 | 36.55 |
| 4 | solo goal 100k | fallen | 7.27 | 2080 | 14.0 | 9.02 |
| 5 | originale | goal | 9.46 | 2040 | 11.4 | 8.70 |
| 5 | solo goal 100k | goal | 11.28 | 2454 | 12.4 | 8.56 |
| 6 | originale | timeout | 23.18 | 5951 | 37.8 | 37.14 |
| 6 | solo goal 100k | goal | 6.68 | 1559 | 11.2 | 3.00 |
| 7 | originale | goal | 12.06 | 2663 | 14.8 | 13.64 |
| 7 | solo goal 100k | goal | 4.89 | 1100 | 6.6 | 2.32 |
| 8 | originale | goal | 12.89 | 3148 | 18.6 | 19.35 |
| 8 | solo goal 100k | goal | 4.17 | 995 | 5.8 | 2.20 |
| 9 | originale | goal | 10.06 | 2339 | 14.6 | 13.92 |
| 9 | solo goal 100k | goal | 8.57 | 1971 | 12.4 | 7.46 |

*Rotazione: somma degli incrementi assoluti di yaw durante la navigazione, campionati a 0.2 s;
esclude allineamento iniziale e ultimo intervallo senza campione successivo. Non è una misura diretta dei giri inutili.
Percorso, energia e durata riportano i totali degli episodi, incluso allineamento.
Start, goal, orientamento iniziale e identificativi delle nove missioni verificati uguali.
Una sola esecuzione per variante: possibili differenze numeriche della simulazione; risultati da confermare.
Confrontare efficienza soprattutto nelle missioni completate da entrambe le varianti.


# Test 7: confronto prima/dopo il cambio di curriculum

| Fase | Checkpoint | Successi | Timeout | Cadute |
|---|---|---:|---:|---:|
| 0 | 150017 | 8/9 | 1 | 0 |
| 0 | 200000 | 8/9 | 1 | 0 |
| 1 | 150017 | 1/9 | 7 | 1 |
| 1 | 200000 | 0/9 | 9 | 0 |

Start, goal e orientamento verificati uguali fra checkpoint all’interno di ogni fase.
Le missioni della fase 0 e della fase 1 sono diverse: confrontare i checkpoint all’interno della stessa fase.
Il risultato 200k/fase 1 è riutilizzato dalla valutazione precedente; le altre tre valutazioni sono nuove.
Nove missioni per condizione sono una diagnosi preliminare, non una stima precisa della robustezza.
Successi inferiori a 200k in fase 0 suggeriscono regressione; successi conservati in fase 0 ma bassi in fase 1 suggeriscono difficoltà di trasferimento.
Entrambi i problemi possono coesistere. La simulazione può introdurre variabilità fra esecuzioni.

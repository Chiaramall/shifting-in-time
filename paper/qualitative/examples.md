# Esempi qualitativi SASRec con nuovi item dal catalogo

Selezione illustrativa a posteriori fra casi con metadati risolti; non è un campione rappresentativo. I conteggi MovieLens sono per genere, quelli Last.fm per artista. Non assegniamo un genere musicale a Chemistry in base al solo nome. Le selezioni sono quelle salvate negli esperimenti: non abbiamo scelto manualmente i nuovi item.

Nel report originale, FRBO è stato ricalcolato dai ranking per tutte e tre le destinazioni. La ricostruzione della finestra dalla storia training completa e l’assenza dei nuovi item dal train sono verifiche documentate nel report originale; i dati completi non sono inclusi in questa repository. Gli item inseriti restano raccomandabili: segnaliamo quelli presenti nella top-10.

## movielens10m — 59996, 20%

Storia training: 220 eventi. La dominanza riportata riguarda soltanto la finestra di 20 eventi.

### Nuovi item inseriti

- Posizione 17: Tokyo Gore Police (Tôkyô zankoku keisatsu) (2008) [Action, Horror]
- Posizione 18: Unknown Solider, The (Unbekannte Soldat, Der) (2006) [Documentary]
- Posizione 19: Letter from an Unknown Woman (1948) [Drama, Romance]
- Posizione 20: Invitation, The (Zaproszenie) (1986) [Drama]

### Occorrenze rimosse

- Posizione originale 1: Night Flier (1997)
- Posizione originale 12: Gate II: Trespassers, The (1990)
- Posizione originale 15: Sorority House Massacre II (1990)
- Posizione originale 20: Child's Play 3 (1991)

### FRBO@10

- old: 0.074202985
- middle: 0.074202985
- recent: 0.000000000

| Rank | Originale | Recent |
| --- | --- | --- |
| 1 | Pet Sematary II | The Invitation (Zaproszenie) **[inserito]** |
| 2 | Children of the Corn II | Two Deaths |
| 3 | Friday the 13th Part V | Two or Three Things I Know About Her |
| 4 | Friday the 13th Part VIII | Crows and Sparrows |
| 5 | Howling II | Lotto Land |
| 6 | Halloween 4 | Shadow of Angels |
| 7 | Poltergeist III | The Old Lady Who Walked in the Sea |
| 8 | Friday the 13th Part VI | Midnight Dancers |
| 9 | Children of the Corn III | The Day the Sun Turned Cold |
| 10 | Graveyard Shift | Every Other Weekend |

### Finestra originale

1. Night Flier
2. Child's Play 2
3. Bride of Chucky
4. American Werewolf in Paris, An
5. Dark Half, The
6. Candyman: Farewell to the Flesh
7. Exorcist III, The
8. Halloween H20: 20 Years Later
9. Tales from the Darkside: The Movie
10. Vampires
11. Psycho
12. Gate II: Trespassers, The
13. Leatherface: Texas Chainsaw Massacre III
14. Sleepwalkers
15. Sorority House Massacre II
16. Dracula: Dead and Loving It
17. Halloween: The Curse of Michael Myers
18. Wishmaster
19. Prophecy II, The
20. Child's Play 3

## lastfm1k — user_000653, 30%

Storia training: 5339 eventi. La dominanza riportata riguarda soltanto la finestra di 20 eventi.

### Nuovi item inseriti

- Posizione 15: David Guetta & Chris Willis — Tomorrow Can Wait (Radio Edit)
- Posizione 16: Yui — Che.R.Ry
- Posizione 17: Los Fastidios — Rudy Rude Boy
- Posizione 18: Yui — I Remember You
- Posizione 19: Nigel Kennedy — Air On The G String (J S Bach)
- Posizione 20: Chemistry — It Takes Two

### Occorrenze rimosse

- Posizione originale 4: The Beatles — Do You Want To Know A Secret
- Posizione originale 5: The Beatles — Ask Me Why
- Posizione originale 7: The Beatles — Ask Me Why
- Posizione originale 9: The Beatles — Do You Want To Know A Secret
- Posizione originale 17: The Beatles — Anna (Go To Him)
- Posizione originale 20: The Beatles — Anna (Go To Him)

### FRBO@10

- old: 0.296896430
- middle: 0.296896430
- recent: 0.000000000

| Rank | Originale | Recent |
| --- | --- | --- |
| 1 | The Beatles -- I Need You | Chemistry -- It Takes Two **[inserito]** |
| 2 | Sparks -- The Number One Song In Heaven | Chemistry -- Solid Dream |
| 3 | Telex -- Moskow Diskow | Chemistry -- Wings Of Words (Instrumental) |
| 4 | The Beatles -- Leave My Kitten Alone | Chemistry -- Grind For Me |
| 5 | Joaquín Sabina -- Juana La Loca | Chemistry -- Almost In Love |
| 6 | George Harrison -- What Is Life | T.M.Revolution -- Invoke -インヴォーク- |
| 7 | The Beatles -- Yes It Is | Chemistry -- 空の奇跡 |
| 8 | The Beatles -- Help! | Chemistry -- Pieces Of A Dream |
| 9 | The Beatles -- Sgt. Peppers Lonely Hearts Club Band | Manna -- でてこいとびきりZenkaiパワー! |
| 10 | The Human League -- Seconds | Chemistry -- 伝説の草原 (Patrick Stump Remix) |

### Finestra originale

1. The Beatles -- I Saw Her Standing There
2. The Beatles -- Anna (Go To Him)
3. The Beatles -- I Feel Fine
4. The Beatles -- Do You Want To Know A Secret
5. The Beatles -- Ask Me Why
6. The Beatles -- Anna (Go To Him)
7. The Beatles -- Ask Me Why
8. The Beatles -- I Saw Her Standing There
9. The Beatles -- Do You Want To Know A Secret
10. The Beatles -- I Feel Fine
11. The Beatles -- It Won'T Be Long
12. The Beatles -- I Feel Fine
13. The Beatles -- Roll Over Beethoven
14. The Beatles -- I'M Happy Just To Dance With You
15. The Beatles -- I Saw Her Standing There
16. The Beatles -- I Wanna Be Your Man
17. The Beatles -- Anna (Go To Him)
18. The Beatles -- It Won'T Be Long
19. The Beatles -- I Wanna Be Your Man
20. The Beatles -- Anna (Go To Him)

## Dati degli esempi

- [Dati dei due esempi](table_evidence.json): ranking, finestre, item inseriti e rimossi, metadati e FRBO.

Sono esempi archiviati, consultabili senza dati completi o checkpoint. La repository
non include gli script di ricerca o generazione degli esempi qualitativi.

Gli esempi riportano la Top-10 completa; il manoscritto ne mostra le prime cinque
posizioni. L'esempio MovieLens usa il 20% (quattro item), quello LastFM il 30%
(sei item). Per la condizione middle, i ranking archiviati danno FRBO 0.074203 su
MovieLens e 0.296896 su LastFM: questi valori differiscono da 0.06 e 0.25 nel PDF
fornito. Qui sono conservati i valori verificabili dai ranking, senza correggere
silenziosamente i dati per farli coincidere con il manoscritto.

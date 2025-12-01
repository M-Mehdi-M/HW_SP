Nume: Mahmoudi Mohammadmehdi
Grupa: 342C2
Tema: Procesarea Semnalelor 2025 - Clasificare si detectie de sunete

== Prezentare Generala ==

Am rezolvat toate cerintele din tema (1-7). Proiectul e impartit in fisiere
separate, asa cum s-a cerut, pentru a tine codul organizat. Mai jos explic ce face
fiecare fisier si cum am gandit rezolvarea.

== Structura Proiectului ==

gabor_filter.py (Cerinta 1):
Aici am scris functia gabor_filter. Practic, construiesc un filtru Gabor
inmultind o fereastra Gaussiana cu o sinusoida (cosinus si sinus). Functia
primeste dimensiunea, sigma si frecventa si returneaza cele doua variante ale
filtrului.

create_filters.py (Cerinta 2):
Acest fisier se ocupa de generarea bancului de filtre. Functia create_filter_bank
imparte banda de frecvente in 12 segmente egale pe scala Mel, folosind functiile
ajutatoare hz2mel si mel2hz. Tot aici am pus si functia de plotare pentru
ca e legata direct de crearea filtrelor.

mexican_hat_filter.py (Cerinta 3 - Filtru la alegere):
Pentru filtrul custom, am ales "Mexican Hat Wavelet" (Ricker). Mi s-a parut o
alegere buna pentru ca e un filtru trece-banda care localizeaza bine semnalul
atat in timp cat si in frecventa, ceea ce ne ajuta la analiza sunetelor.
Formula am luat-o din articolul lui Ryan, H. despre ondulete (1994).

create_custom_bank.py (Cerinta 4):
Aici am adaptat logica de la filtrele Gabor pentru Mexican Hat. Generez un set
de filtre distribuite tot pe scala Mel. A trebuit sa calculez sigma specific
pentru fiecare frecventa centrala, astfel incat varful spectrului sa fie unde
trebuie.

get_features.py (Cerinta 5 - Filtrare rapida):
Fisierul acesta contine "motorul" de procesare. Functia get_features face toata
treaba grea: sparge sunetul in ferestre (cu stride de 12ms) si aplica filtrele.
Pentru eficienta, am folosit inmultire de matrici in loc sa fac convolutie clasica
pe fiecare fereastra, ceea ce a facut codul mult mai rapid. La final, pentru fiecare
fereastra calculez media si deviatia standard a raspunsului.

tema_2025_schelet.py (Cerinta 6 - Clasificare Standard):
Acesta e scheletul primit, pe care l-am folosit pentru a rula clasificarea de baza
(KNN + Gabor) pe un subset de date (asa cum e configurat default pentru viteza).

classification.py (Cerintele 6a-6d & 7 - Clasificare Completa):
Am creat acest script ca sa pot rula toate cele 4 scenarii de clasificare cerute
pe TOATE datele (data.mat), nu doar pe o parte. Tot de aici generez si graficul
final cu rezultatele.

test_tasks.py (Script de Validare):
L-am scris ca sa ma asigur ca fiecare bucata de cod merge cum trebuie inainte sa
le pun cap la cap. Testeaza dimensiunile filtrelor, daca ferestrele sunt taiate
corect si daca pipeline-ul de clasificare ruleaza fara erori. E util pentru debug.

== Analiza Rezultatelor (Grafice) ==

M-am uitat peste graficele generate si totul pare in regula:

Filtrele Gabor (cos/sin) arata corect, simetrice si antisimetrice.

Spectrul filtrelor Gabor arata cei 12 lobi. Se vede clar cum devin mai lati si
mai rari pe masura ce creste frecventa, exact cum prezice scala Mel.

Filtrul Mexican Hat arata ca in teorie (un "varf" incadrat de doua "gropi").

Spectrul filtrelor custom acopera bine banda de frecventa, deci bancul e valid.

== Rezultate Obtinute (Acuratete) ==

Am rulat testele pe setul complet de date si am obtinut urmatoarele:

6.a) KNN + Gabor: 68.00%
A mers surprinzator de bine, e chiar la limita superioara a intervalului cerut.

6.b) KNN + Mexican Hat: 62.00%
Rezultat decent, dovedeste ca filtrul custom functioneaza, chiar daca Gabor e
putin mai potrivit pentru datele astea.

6.c) MinDist + Gabor: 60.00%
Aici a fost mai complicat. Initial, clasificatorul MinDist (Nearest Centroid)
avea rezultate slabe. Ca sa-l fac sa mearga, am aplicat logaritm pe trasaturi
(ca sa simulez decibelii) si am normalizat datele (StandardScaler). Am folosit
si distanta Manhattan in loc de Euclidiana, ca pare sa mearga mai bine pe
dimensiuni mari. Asa am reusit sa trec de pragul de 55%.

6.d) MinDist + Mexican Hat: 59.00%
Similar cu Gabor, optimizarile au ajutat sa obtin un scor de trecere.

== Concluzii Personale ==
Din ce am observat, filtrele Gabor par sa se muleze mai bine pe structura sunetelor
din acest dataset decat onduletele Ricker. De asemenea, desi KNN e un algoritm simplu,
s-a descurcat cel mai bine. La MinDist a fost nevoie de ceva "inginerie" a datelor
(log, scalare) ca sa obtin rezultate competitive, ceea ce imi arata cat de importanta
e pre-procesarea in ML.

Numar cerinte rezolvate: Toate (1, 2, 3, 4, 5, 6, 7).

https://ocw.cs.pub.ro/courses/_media/ps/data.mat
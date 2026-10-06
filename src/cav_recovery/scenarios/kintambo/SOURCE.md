# Topologie de Kintambo Magasin

© OpenStreetMap contributors, sous [ODbL 1.0](https://www.openstreetmap.org/copyright).
Cette licence concerne l'extrait géographique, pas le code Python du projet.

Source : API officielle OpenStreetMap, récupérée le 6 octobre 2026 :
https://api.openstreetmap.org/api/0.6/map?bbox=15.244,-4.348,15.274,-4.322

Bbox (ouest, sud, est, nord) : 15.244, -4.348, 15.274, -4.322.
Centre indicatif : longitude 15.25822, latitude -4.33404.
L'emprise couvre environ 3,3 × 2,9 km. Les ways complets peuvent dépasser ses
bords ; ils sont conservés pour ne pas casser leurs connexions.

Le fichier roads.osm.gz conserve les voies motorisées, leurs nœuds et les
restrictions complètes. Les bâtiments, les relations non routières et les
identifiants des contributeurs ne sont pas distribués. Aucune géométrie n'a
été inventée. La compression est déterministe.

SHA-256 de la réponse originale (6 800 707 octets) :
531f352384f558407eefef1dd5771f59a9fb2c83a9ffcbd44f782f5b02aba1e2

SHA-256 de l'extrait décompressé (484 319 octets) :
93a31d50a909221924ef820747d87035bbea5451a3bdd339f689d3783e40c45b

828 ways et 4 492 nœuds OSM, avant conversion.

La configuration transforme les nombres de voies et les vitesses selon la
fonction des axes : 4 voies/sens pour les axes majeurs (4 sur chaque chaussée
OSM séparée), 2 pour les axes intermédiaires, 1 pour les branches mineures.
Les vitesses nominales sont respectivement 50, 40 et 30 km/h.
Elle ajoute des feux statiques aux intersections à plusieurs
approches dans le rayon indiqué. Quatre groupes explicites de raccords centraux
sont regroupés par netconvert : après élargissement, leurs surfaces se
chevauchaient et les deux premiers essais légers ont détecté des collisions.
La jointure automatique ne regroupait pas correctement ces chaussées parallèles.
Le regroupement explicite conserve les
branches extérieures ; il ne remplace pas le noyau par une croix.
Il est complété par la jointure géométrique standard à 30 m. La configuration
retenue conserve quatre groupes centraux, les intersections voisines et leurs
boucles. Les regroupements plus importants éprouvés ont été écartés.
Les feux utilisent le layout SUMO opposites, un cycle demandé de 90 s,
un dégagement tout rouge de 2 s et des gauches protégées lorsque SUMO peut les
construire (tls.minor-left.max-speed=0). Le programme effectivement produit
est conservé dans chaque scénario. keepClear et les contrôles de collision
aux jonctions restent actifs.
Les sens OSM et les chaussées séparées restent
conservés. Ce réseau est synthétique, dérivé de la topologie de Kintambo :
ni voies, ni vitesses, ni feux, ni demande ne sont calibrés sur le trafic réel.

Limite connue des essais : LOW se vide pour les seeds 1 et 2 et HIGH/seed 1
est observé sans collision à l'horizon, mais MEDIUM/seed 1 a détecté une
collision de jonction. Cet essai est invalidé, pas écarté des diagnostics.
Ces contrôles ne certifient donc pas tous les couples charge/seed et ne
valident pas un gridlock ni la sécurité générale du réseau.

Le rapport JICA décrit historiquement un carrefour complexe à sept branches ;
il ne valide pas les paramètres de cette simulation :
https://openjicareport.jica.go.jp/pdf/12340345_03.pdf (figure 8.3.13).

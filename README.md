# Scan 3D

Application web de scan 3D par photos et planche ChArUco, export STL à l'échelle en mm,
avec ajustement manuel des dimensions. Tout le calcul se fait dans le navigateur du
téléphone : aucune photo n'est envoyée sur un serveur.

## Mise en ligne sur GitHub Pages

1. Créer un dépôt sur GitHub (public, ou privé avec un compte GitHub Pro), par exemple `scan3d`.
2. Y déposer tous les fichiers de ce dossier à la racine (bouton "Add file" puis "Upload files",
   ou `git push`).
3. Dans le dépôt : Settings > Pages > Build and deployment > Source : "Deploy from a branch",
   Branch : `main`, dossier `/ (root)`, puis Save.
4. Après une à deux minutes, l'appli est en ligne sur `https://<utilisateur>.github.io/scan3d/`.
5. Sur le téléphone, ouvrir l'adresse puis "Ajouter à l'écran d'accueil" pour l'avoir comme une appli.

Pour tester sur le PC avant de publier : `python -m http.server 8000` dans ce dossier, puis
ouvrir `http://localhost:8000` (ouvrir `index.html` directement depuis l'explorateur ne fonctionne pas).

## Fichiers

| Fichier | Rôle |
|---|---|
| `index.html` | Interface |
| `app.js` | Enchaînement des étapes, dimensions, aperçu 3D, export STL |
| `vision-worker.js` | Détection de la planche, pose de chaque photo, calibration, détourage (OpenCV.js) |
| `seg-worker.js` | Détourage optionnel par modèle d'IA (RMBG-1.4, ONNX Runtime Web) |
| `recon-worker.js` | Sculpture du volume, maillage, lissage |
| `planche.pdf` | Planche ChArUco A4 à imprimer (9 x 13 carrés de 20 mm) |

Les bibliothèques sont chargées depuis cdn.jsdelivr.net (OpenCV.js 5.0, three.js 0.160, et
ONNX Runtime Web 1.22 si l'option IA est activée), puis gardées en cache par le navigateur.

Après chaque mise à jour des fichiers, augmenter `VERSION` en haut de `app.js` et le `?v=` de
`app.js` dans `index.html` (même valeur) : sinon le navigateur peut mélanger anciens et nouveaux
fichiers pendant quelques minutes.

## Détourage

L'objet est séparé du fond en comparant chaque image au motif connu de la planche : grâce à la
pose, l'appli sait quel carré noir ou blanc devrait apparaître à chaque pixel. Ce qui ne
correspond pas est l'objet. Les pixels ambigus (bord d'un carré, objet très sombre posé sur du
noir) restent indéterminés et ne retirent rien du volume. Sur les vignettes : objet clair, fond
assombri, zones indéterminées à peine assombries.

Un modèle d'IA (RMBG-1.4, 44 Mo, licence non commerciale) reste disponible dans les réglages
avancés, mais il a tendance à prendre toute la feuille pour l'objet.

## Utilisation

1. **Planche** : imprimer `planche.pdf` en taille réelle, la coller sur un support rigide,
   mesurer la barre de 100 mm et saisir la valeur mesurée.
2. **Calibration** (une fois par téléphone) : 15 à 25 photos de la planche seule, angles et
   distances variés. Zoom x1, toujours la même orientation. La calibration est mémorisée dans le
   navigateur ; le bouton Exporter permet de la sauvegarder et de la réimporter ailleurs.
3. **Photos de l'objet** : objet au centre de la planche, 40 à 80 photos en trois tours (bas,
   45 degrés, presque vertical). Chaque vignette montre le détourage : objet clair, fond assombri.
   Toucher une vignette pour l'exclure si le détourage est faux.
4. **Reconstruction** : environ une seconde sur PC, quelques secondes sur téléphone avec les
   réglages par défaut.
5. **Dimensions et export** :
   - "Aligner sur les axes" tourne l'objet pour que ses arêtes suivent X et Y.
   - Saisir les cotes mesurées au pied à coulisse dans la colonne Final, puis "Appliquer les cotes".
     Case "Proportionnel" cochée : une seule cote suffit, les deux autres suivent.
     Décochée : chaque axe est corrigé indépendamment.
   - "Seuil de détection" : chaque point du volume a un score, la part des vues qui le voient
     comme objet. Seuls les points dont le score atteint le seuil (en % du score typique de
     l'objet) sont gardés. Monter le seuil enlève débris et artefacts. Réglé automatiquement :
     60 % si les scores sont fiables (objet coloré), 0 % sinon (objet sombre).
   - "Couper à la hauteur" pour un objet à dessus plat, dont la hauteur est surestimée par la méthode.
   - "Télécharger le STL" : fichier en mm, posé sur Z = 0, centré en X/Y.

## Prise en direct

Les étapes 2 et 3 peuvent se faire caméra ouverte, sans passer par l'appareil photo :

- **Calibrer en direct** : filmer la planche seule en tournant autour, en variant hauteur et
  distance. 20 images sont prises automatiquement, puis la calibration est calculée.
- **Prise en direct** : tourner lentement autour de l'objet. Une vue est prise dès que le
  téléphone est stable sur un angle pas encore couvert. Un anneau dessiné sur la feuille autour
  de l'objet montre la couverture sur 360 degrés : un secteur se colore en vert quand la vue
  prise de ce côté est faite. Anneau extérieur : vues basses ; anneaux suivants : vues de plus
  en plus hautes ; anneau intérieur : vues du dessus. Le secteur où se trouve le téléphone est
  entouré de blanc.
  Le cadre en pointillés est la zone de travail : l'objet doit tenir dedans. Quand chaque étage
  du dôme est couvert, la capture s'arrête, les vues sont détourées et l'aperçu 3D s'affiche.
  "Arrêter" garde les vues prises pour reprendre plus tard, "Terminer et reconstruire" lance
  l'aperçu sans attendre la couverture complète.

La calibration en direct et la calibration par photos ne sont pas interchangeables (cadrage
différent) : le scan en direct demande une calibration en direct, dans la même orientation du
téléphone. La caméra en direct a une résolution plus faible que les photos : le mode photos
reste un peu plus précis.

## Objets blancs, gris ou noirs

Sur la planche noir et blanc, un objet sans couleur se confond avec le damier. Deux solutions :

- **Planche couleur** (`planche-couleur.pdf`, imprimante couleur) : mêmes marqueurs, en bleu foncé
  et jaune clair. Tout le fond étant coloré, un objet blanc, gris ou noir ressort partout.
  L'appli reconnaît seule la planche utilisée. Éviter les objets bleu foncé ou jaunes.
- **Fond coloré** (case à cocher à l'étape 3) : avec la planche noir et blanc, poser une feuille
  de couleur vive d'environ 10 x 10 cm au centre, sous l'objet, d'une autre couleur que lui.

## Limites

- Les creux (intérieur d'un bol, rainure, trou borgne) ne sont pas reconstruits.
- Les dimensions brutes sont en général 2 à 5 % trop grandes : corriger avec une cote mesurée.
- Objets brillants, transparents, blancs ou noir profond : détourage peu fiable (un objet blanc
  se confond avec le papier, un objet noir avec l'encre). Un gris, même foncé, ou une couleur
  passent bien.
- Le modèle RMBG-1.4 (option IA, BRIA AI) est sous licence non commerciale : usage personnel
  uniquement. Un autre modèle ONNX (entrée 1 x 3 x 1024 x 1024, sortie 1 x 1 x 1024 x 1024)
  peut être indiqué dans les réglages avancés.

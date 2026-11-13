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
| `vision-worker.js` | Détection de la planche, pose de chaque photo, calibration (OpenCV.js) |
| `seg-worker.js` | Détourage de l'objet (modèle RMBG-1.4, ONNX Runtime Web) |
| `recon-worker.js` | Sculpture du volume, maillage, lissage |
| `planche.pdf` | Planche ChArUco A4 à imprimer (9 x 13 carrés de 20 mm) |

Les bibliothèques sont chargées depuis cdn.jsdelivr.net (OpenCV.js 5.0, ONNX Runtime Web 1.22,
three.js 0.160) et le modèle de détourage depuis huggingface.co, puis gardés en cache par le
navigateur.

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
   - "Couper à la hauteur" pour un objet à dessus plat, dont la hauteur est surestimée par la méthode.
   - "Télécharger le STL" : fichier en mm, posé sur Z = 0, centré en X/Y.

## Prise en direct

Les étapes 2 et 3 peuvent se faire caméra ouverte, sans passer par l'appareil photo :

- **Calibrer en direct** : filmer la planche seule en tournant autour, en variant hauteur et
  distance. 20 images sont prises automatiquement, puis la calibration est calculée.
- **Prise en direct** : tourner lentement autour de l'objet. Une vue est prise dès que le
  téléphone est stable sur un angle pas encore couvert. Le dôme en haut à droite montre les
  angles pris (centre : vues du dessus, bord : vues basses), le point blanc la position actuelle.
  Le cadre en pointillés est la zone de travail : l'objet doit tenir dedans. Quand chaque étage
  du dôme est couvert, la capture s'arrête, les vues sont détourées et l'aperçu 3D s'affiche.
  "Arrêter" garde les vues prises pour reprendre plus tard, "Terminer et reconstruire" lance
  l'aperçu sans attendre la couverture complète.

La calibration en direct et la calibration par photos ne sont pas interchangeables (cadrage
différent) : le scan en direct demande une calibration en direct, dans la même orientation du
téléphone. La caméra en direct a une résolution plus faible que les photos : le mode photos
reste un peu plus précis.

## Limites

- Les creux (intérieur d'un bol, rainure, trou borgne) ne sont pas reconstruits.
- Les dimensions brutes sont en général 1 à 2 % trop grandes : corriger avec une cote mesurée.
- Objets brillants, transparents ou de la couleur de la planche : détourage peu fiable.
- Le modèle RMBG-1.4 (BRIA AI) est sous licence non commerciale : usage personnel uniquement.
  Un autre modèle ONNX de détourage (entrée 1 x 3 x 1024 x 1024, sortie 1 x 1 x 1024 x 1024)
  peut être indiqué dans les réglages avancés.

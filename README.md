# Partie 1 — Traitement local : comptage de personnes par webcam

Projet *Architectures and Infrastructures for Artificial Intelligence* — Pratique 1 : Cloud Computing Services (Smart Lighting System).

Cette partie réalise le traitement **en local** : une image de la webcam est capturée toutes les 5 secondes, le nombre de personnes est calculé avec YOLOv8, puis les résultats sont enregistrés dans un fichier CSV et tracés sur un graphique.

| Tâche | Description | Réalisée par |
|---|---|---|
| 1.1 | Choisir une webcam publique fixe (personnes ou véhicules) | Paramètre `--url` |
| 1.2 | Capturer une image toutes les 5 s | `capture_count.py` |
| 1.3 | Compter les personnes, enregistrer en CSV, tracer le graphe | `capture_count.py` + `plot_results.py` |

## Structure

```
part1/
├── capture_count.py     # capture + comptage YOLOv8 + écriture du CSV
├── plot_results.py      # graphe « personnes par image au cours du temps »
├── requirements.txt     # dépendances Python
├── yolov8n.pt           # modèle YOLOv8 nano (téléchargé automatiquement si absent)
└── results/             # sorties (créé à l'exécution)
    ├── people_count.csv
    ├── people_over_time.png
    ├── first_frame.jpg
    ├── frames_raw/          # images originales
    └── frames_annotated/    # images avec cadres rouges (nom : AAAAMMJJ_HHMMSS_Np.jpg)
```

## Prérequis

- Python 3.12 ou supérieur
- Une webcam publique en direct (flux `.m3u8`, MJPEG, RTSP, ou lien YouTube en direct)

## Installation

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# Windows     : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### 1. Capture et comptage

```bash
python capture_count.py --url "SOURCE" [options]
```

Seul `--url` est obligatoire.

| Argument | Défaut | Description |
|---|---|---|
| `--url` | — (obligatoire) | Source : lien de flux (`.m3u8`, MJPEG, RTSP), lien YouTube en direct, ou fichier image/vidéo local |
| `--referer` | aucun | Adresse du site envoyée comme en-tête Referer, si le flux refuse de s'ouvrir |
| `--interval` | `5` | Secondes entre deux captures |
| `--duration` | `0` | Durée en minutes ; `0` = jusqu'à Ctrl+C |
| `--model` | `yolov8n.pt` | Modèle YOLO (ex. `yolov8s.pt`, plus précis mais plus lent) |
| `--conf` | `0.5` | Seuil de confiance (0 à 1) |
| `--out` | `results` | Dossier de sortie |

Exemples :

```bash
# Test rapide avec l'image fournie (sans réseau)
python capture_count.py --url ../material/image.png --duration 0.05

# Lien YouTube en direct, 15 minutes
python capture_count.py --url "https://www.youtube.com/watch?v=XXXX" --duration 15 --out run1

# Flux .m3u8 direct, modèle small, seuil plus bas
python capture_count.py --url "https://.../playlist.m3u8" --model yolov8s.pt --conf 0.4 --duration 15 --out run_s
```

> **Important** : mettez toujours l'URL entre guillemets (sous PowerShell, le caractère `&` est sinon interprété comme un opérateur). Utilisez un dossier `--out` différent pour chaque essai : le CSV est complété (mode ajout), les anciens résultats seraient mélangés aux nouveaux.

### 2. Graphe

```bash
python plot_results.py --csv run1/people_count.csv --out run1/people_over_time.png
```

Sans arguments, le script lit `results/people_count.csv` et écrit `results/people_over_time.png`. Il affiche le nombre de points, la moyenne et le maximum.

## Format de sortie

`people_count.csv` :

```
timestamp,people
2026-11-22 17:53:01,4
2026-11-22 17:53:06,3
```

## Vérifier la qualité de détection

1. Ouvrir `first_frame.jpg` pour confirmer que la bonne caméra est utilisée.
2. Comparer quelques images de `frames_raw/` avec leur version dans `frames_annotated/` (le nom se termine par `_Np`, N = nombre détecté).
3. Compter à la main sur environ 10 images et comparer avec le comptage de YOLO.
4. Si besoin, baisser `--conf` (plus de détections, plus de faux positifs) ou passer à `--model yolov8s.pt`.

## Problèmes courants

| Symptôme | Cause probable | Solution |
|---|---|---|
| `The ampersand (&) character is not allowed` (PowerShell) | URL sans guillemets | Mettre l'URL entre guillemets |
| `could not read a frame` avec un `.m3u8` | Jeton expiré dans l'URL ou site qui exige un Referer | Récupérer une URL fraîche (F12 → Network → filtre `m3u8`) et ajouter `--referer` |
| `Requested format is not available` (YouTube) | Aucun format audio+vidéo combiné proposé | Le script utilise déjà le sélecteur `bv*[height<=1080]/bv*/b` (vidéo seule) ; mettre `yt-dlp` à jour : `pip install -U yt-dlp` |
| `CAP_IMAGES: expected '0?[1-9][du]' pattern` | Message secondaire d'OpenCV quand le flux n'a pas pu être ouvert | Résoudre l'ouverture du flux (voir lignes précédentes) |
| Même nombre de personnes à chaque image avec YouTube | La vidéo n'est pas un direct | Choisir un flux en direct (badge « LIVE ») |
| Graphe plat à 0 | Rue vide pendant l'enregistrement | Choisir une caméra avec passage, ou enregistrer plus longtemps |

## Éléments à joindre au rapport

- Choix de la caméra et justification (fixe, personnes visibles), avec capture d'écran du flux
- `first_frame.jpg`, intervalle (5 s) et résolution
- `people_count.csv`, deux ou trois images annotées et le graphe
- Captures d'écran de l'installation et de l'exécution
- Problèmes rencontrés et décisions de conception (modèle, seuil de confiance, classe « personne » uniquement)

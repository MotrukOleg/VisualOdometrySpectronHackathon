# CLAUDE.md

Цей файл задає контекст для Claude Code (claude.ai/code) у цьому репозиторії.

## Мета проєкту

Тестове завдання It-Jim «Classic CV: Drone navigation» 2026, **рівень Advanced**. Повна постановка, етапи і чекліст —
у [docs/ROADMAP.md](docs/ROADMAP.md). Перед роботою над новим етапом читати роадмапу й позначати виконані пункти.

- **Standard:** є супутникова мапа (`global_map.png`) і послідовність знімків із дрона. Треба зіставити кожен знімок
  із мапою (keypoint / descriptor matching) і **відновити й намалювати маршрут дрона на мапі**.
- **Advanced:** власне відео, де телефон знімає мапу з екрана монітора (імітація прольоту). Кадри відео
  зіставляються з мапою, будується траєкторія.
- **Результат:** зображення мапи з маршрутом (START/FINISH, порядок точок, бажано контури кадрів із розміром і
  поворотом) і/або відео з поточною позицією. Плюс охайний код, `README.md`, `requirements.txt` / `environment.yml`.
- **Наша додаткова мета:**
  - склеїти кадри в мозаїку (VO між кадрами);
  - систематично порівняти методи (SIFT, ORB, AKAZE, SuperPoint, DISK, ALIKED, XFeat, SuperGlue, LightGlue,
    LoFTR, …) і вибрати найкращу комбінацію за метриками.
- **Фінальний результат — презентація** з отриманими результатами (етап 7 роадмапи).
  - Робиться **останньою**, після завершення всіх етапів і фіналізації цифр.
  - Матеріали для неї збираємо **протягом усієї роботи** (див. розділ «Презентація» нижче).

## Дані

`data/` у `.gitignore`: дані не комітимо, архіви `*.zip` теж. Виняток — `data/manifest.yaml`.

**Отримання даних — лише через скрипт**, щоб у всієї команди були побайтово однакові файли:
- `python scripts/download_data.py`: завантаження з Google Drive за `data/manifest.yaml` → розпаковка → перевірка
  SHA256 кожного файлу.
- `--from-zip <архів>` — якщо архів уже завантажено вручну.
- `--verify` — лише перевірка локальних файлів.
- `--hash <файл>` — SHA256 для нового запису в маніфесті.
- **Нові дані** (власні відео тощо): файл на спільний Google Drive з доступом «усі, хто має посилання» → запис у
  `data/manifest.yaml` (id, хеш, `dest`) → коміт маніфесту. Самі файли в git не додаємо.
- Завантажені архіви кешуються в `data/cache/`.

- `data/train/global_map.png` — мапа 3645×1845, місто з річкою.
- `data/train/crops/crop_0..49.png` — 50 кадрів по 640×640.
  - Порядок — числовий (`crop_2` < `crop_10`); `FrameSource` сортує через `natural_key`.
  - Масштаб кадр → мапа ≈ **0.25**; крок ≈ 90 px мапи на кадр; сусідні кадри сильно перекриваються.
  - Ракурс косий (видно фасади й тіні), кольори відрізняються від мапи.
  - SIFT + FLANN + MAGSAC++ по всій мапі знаходить кожен перевірений кадр (50–180 inliers).
- `data/synthetic/` — згенерований політ із точним GT (`scripts/make_synthetic.py`).
  - `--world data/train/global_map.png --frame-scale 0.25` дає GT-бенчмарк на реальній мапі.
- `data/video/` — тут будуть власні відео (етап 4).
- **GT для 50 кадрів немає.** Оцінювання: синтетика над мапою + ручна розмітка (етап 1 роадмапи).

## Середовище

- Windows 10, PowerShell (Git Bash теж доступний).
- **Conda-середовище `MachineLearning`**: `E:\Anaconda\envs\MachineLearning\python.exe`.
  - Python 3.11, torch 2.11 + CUDA 12.8, opencv-contrib 4.12, numpy, scipy, pandas, h5py, pytest.
  - Ще **не встановлено**: `kornia`, `lightglue` (див. `requirements-learned.txt`). Ставити лише в це середовище.
  - `conda` не в PATH. Викликати Python за повним шляхом, у Bash — `/e/Anaconda/envs/MachineLearning/python.exe`.
- GPU: **GTX 1650, 4 GB VRAM**.
  - Навчені моделі: `torch.inference_mode()`, fp16, де можна, тайли мапи; зменшувати вхід.
  - RoMa / MASt3R можуть не влізти в пам'ять → зменшений вхід, CPU або Colab.
  - Завжди мати CPU-фолбек (`device: auto`).
- **У шляху профілю користувача кирилиця.**
  - Зображення читати/писати лише через `vo.io.imread` / `vo.io.imwrite` (`imdecode` / `imencode`);
    `cv2.imread` з такими шляхами на Windows не працює.
  - Текстові файли — з `encoding="utf-8"`.

## Команди

```powershell
$py = "E:\Anaconda\envs\MachineLearning\python.exe"
& $py scripts/download_data.py                                        # дані за маніфестом + перевірка SHA256
& $py -m pytest -q                                                    # тести (~20 с)
& $py scripts/make_synthetic.py --out data/synthetic --frames 120     # синтетичний політ із GT
& $py scripts/make_synthetic.py --out data/synthetic_map --world data/train/global_map.png --frame-scale 0.25 --frame-size 640x640
& $py scripts/run_experiment.py --config configs/baseline.yaml --set data.max_frames=50
& $py scripts/run_benchmark.py --configs "configs/*.yaml"             # усі конфіги → experiments/summary.csv
& $py scripts/extract_frames.py video.mp4 --out data/video/frames --every 5
& $py scripts/make_area.py --area lviv_old_town                       # будь-яка місцевість: мапа з тайлів + політ із GT
& $py scripts/export_gps.py --run <run> --geo data/areas/<name>/geo.json   # lat/lon, GeoJSON, інтерактивна мапа
```

**Будь-яка місцевість** (`src/vo/tiles.py`, `scripts/make_area.py`, райони — `configs/areas.yaml`): мапа з тайлів
Esri World Imagery на zoom 17 + та сама ділянка на zoom 19 для кадрів (масштаб 0.25, як у ТЗ) → `data/areas/<name>/`
(`map.jpg`, `geo.json` з геоприв'язкою піксель ↔ lat/lon, `flight/frames`, `flight/gt.csv`). Тайли кешуються в
`data/cache/tiles/`. Esri віддає заглушку «Map data not yet available» (HTTP 200) там, де немає знімків —
`tiles.is_placeholder` їх відкидає, а `make_area.py` бере найдетальніший доступний zoom.
`src/vo/gps.py` переводить траєкторію в lat/lon за `geo.json`, рахує похибку в метрах і пише GeoJSON та
самодостатню Leaflet-сторінку `route_map.html`.

## Архітектура коду (`src/vo/`)

**Наявні модулі (VO між кадрами + мозаїка):**
- `config.py` — `DEFAULTS` + YAML з успадкуванням (`base: baseline.yaml`) + оверрайди CLI `--set a.b=value`.
  Усі пороги — тільки тут або в YAML.
- `io.py` — `FrameSource`: директорія кадрів, `step`, `max_frames`; `resize_long` лише зменшує; `scale` — явний
  масштаб. Також `load_gt` і Unicode-безпечні `imread` / `imwrite`.
- `features/` — `Extractor.extract(img) -> Features`.
  - Класичні: OpenCV.
  - Навчені: обгортки над пакетом `lightglue` (SuperPoint, DISK, ALIKED, SIFT).
- `matching/` — `PairMatcher.match(i, img_i, j, img_j) -> Correspondences`.
  - `FeatureMatcher` = екстрактор + дескрипторний матчер (BF / FLANN / LightGlue); кешує ознаки кожного кадру.
  - `ImagePairMatcher` — методи без детектора, що працюють із парою зображень: LoFTR (kornia), XFeat (torch.hub),
    SuperGlue (офіційний репозиторій у `third_party/`).
  - Фабрика: `build_pair_matcher(cfg)`.
- `geometry.py` — `estimate_transform`: моделі similarity / affine / homography, оцінювачі RANSAC / MAGSAC++ /
  LMedS / USAC; перевірка правдоподібності.
  - **Конвенція:** `T_ij` переводить пікселі кадру j у кадр i; `G_j = G_i @ T_ij`.
- `graph.py` — ребра, кандидати на закриття петель, `optimize` (sparse `least_squares`, `G_0` фіксована).
- `pipeline.py` — `run(cfg)`: реєстрація пар → граф → метрики → мозаїка з траєкторією.
  Результати пише в `experiments/results/<name>_<timestamp>/`.
- `mosaic.py`, `trajectory.py`, `eval.py` (Umeyama, ATE, RPE), `benchmark.py` (`summary.csv`),
  `synthetic.py` (процедурний світ або реальна мапа + політ із GT).

**Заплановані модулі** (див. роадмапу):
- `mapindex.py` — тайли мапи, кеш її ознак;
- `localize.py` — кадр → мапа, глобальна ініціалізація + локальний трекінг;
- `filtering.py` — відсів викидів, інтерполяція, Kalman / RTS;
- `video.py` — відбір різких кадрів, маска екрана;
- візуалізація фінального зображення і відео;
- `scripts/localize.py`, `scripts/annotate_gt.py`, `scripts/render_video.py`.

У режимі мапи світова система координат — **пікселі `global_map.png`**, а не кадр 0.

## Бейзлайн і експерименти

- **Бейзлайн локалізації на мапі:** SIFT (кадр зменшено до масштабу мапи) + FLANN + ratio 0.75 + MAGSAC++ гомографія
  по всій мапі, без трекінгу і згладжування.
- **Фінальний метод (обрано 2026-11-20):** `configs/map/final.yaml` — SIFT + LightGlue, трекінг у ROI з деротацією,
  ре-ініціалізація SIFT + FLANN по всій мапі. Обґрунтування — `docs/presentation/notes.md` (розділ «Фінальний метод»).
- **Бейзлайн VO між кадрами:** `configs/baseline.yaml` — SIFT + FLANN + ratio 0.75 + RANSAC, ланцюжок.
- **Головні кандидати:** SuperPoint / ALIKED / DISK + LightGlue, XFeat, Efficient LoFTR, SuperGlue — разом із
  деротацією за прогнозом курсу, пошуком у ROI мапи, злиттям з VO і згладжуванням.
- **Метрики:**
  - похибка центру й кутів (px мапи), success rate @ 5 / 10 / 25 px;
  - частка кадрів `map` / `vo` / `lost`, inliers, FPS, VRAM;
  - для VO: ATE, RPE, дрейф на петлях, фотометрична похибка в зонах перекриття.
- Кожен запуск описує YAML-конфіг. Висновки робимо **тільки за таблицею `experiments/summary.csv`**, а не за окремими
  вдалими кадрами.
- Ліцензія SuperGlue і ваг SuperPoint від Magic Leap — некомерційна; у звіті згадати.

## Презентація

- **Структура слайдів** — у етапі 7 [docs/ROADMAP.md](docs/ROADMAP.md): задача → дані → підхід → протокол
  оцінювання → бейзлайн → порівняння методів → абляції → фінальний метод → результати Standard і Advanced →
  провали й обмеження → висновки.
- **Під час роботи:**
  - після кожного вагомого експерименту копіювати фінальні рисунки в `docs/presentation/figures/` (ці файли
    комітяться, на відміну від `experiments/results/`);
  - записувати ключові числа й спостереження з `run_id` у `docs/presentation/notes.md`.
- **Рисунки одразу робити придатними для слайдів:** великі шрифти, підписи осей, легенда, без зайвих деталей;
  PNG ≥ 1600 px по ширині.
- **Числа на слайдах** беруться лише з `experiments/summary.csv` / `metrics.json`.
- **Інструмент — reveal.js** (HTML-слайди), робоча папка `docs/presentation/`. MCP не потрібен: Node 22 і npm уже є.
  - Версії фіксуються в `docs/presentation/package.json` (`reveal.js@5.1.0`); `node_modules/` не комітимо.
  - Слайди: `index.html`, тема `css/theme.css`, рисунки `figures/`, відео й GIF `media/` (≤ 10 МБ на файл).
  - Плагіни: Notes, Highlight, Math (KaTeX), Zoom. Відео вставляти через `<video data-autoplay>`.
  - Перегляд: `npx http-server docs/presentation -p 8000` + вбудований браузер Claude (`preview_start`).
  - PDF: `npx decktape reveal http://localhost:8000 docs/presentation/slides.pdf` або `?print-pdf` у Chrome.
    PPTX reveal.js не створює, тільки за окремим запитом.
  - Цифри для слайдів генерує скрипт із `summary.csv` (`scripts/make_slides_data.py`), не вручну.
  - Стиль і тему уточнити в користувача перед створенням.

## Робочі домовленості

- Мова спілкування й документації — українська; ідентифікатори й коментарі в коді — англійською.
- Нові методи додаються як реалізація `Extractor`, `DescriptorMatcher` або `ImagePairMatcher` + запис у фабрику +
  YAML-конфіг. Конвеєр під конкретний метод не змінюємо.
- Важкі залежності (torch, kornia, lightglue) імпортувати ліниво всередині класів, щоб класичний конвеєр працював
  без них.
- Після змін запускати `pytest`. Для нової логіки — тест на синтетиці з GT.
- Не починати новий етап роадмапи без явної команди користувача.
- Великі файли (дані, ваги, `third_party/`, результати) не комітити. Комітити лише за проханням користувача.

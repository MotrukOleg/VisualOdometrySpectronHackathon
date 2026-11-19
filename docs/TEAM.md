# Розподіл роботи на 4 учасників

Кожен **новий** файл має одного власника. Наявні файли, які змінюють кілька людей, перелічено окремо в розділі
«Спільні файли»: їх змінюють маленькими окремими PR. Етапи — за [ROADMAP.md](ROADMAP.md).

| # | Роль | Етапи | Гілка |
|---|---|---|---|
| 1 | Локалізація на мапі (ядро, критичний шлях) | 2, 3 | `feat/localize` |
| 2 | Оцінювання і бенчмарк | 1, 5 | `feat/eval` |
| 3 | Навчені методи і GPU | частина 5 | `feat/learned` |
| 4 | Відео, візуалізація, презентація | 4, 6, 7 | `feat/video-viz` |

---

## Наявні файли каркаса: хто за що відповідає

Розподіл уже написаних файлів за ролями. Кожен файл має рівно одного власника; перевірено, що список покриває всі
файли репозиторію, крім `README.md` з першого коміту. Власник має вміти пояснити свої файли на співбесіді.

**1. Ядро конвеєра і геометрія** (13 файлів, ~950 рядків):
`.gitignore`, `pyproject.toml`, `requirements.txt`, `src/vo/__init__.py`, `src/vo/config.py`, `src/vo/io.py`,
`src/vo/preprocess.py`, `src/vo/geometry.py`, `src/vo/graph.py`, `src/vo/pipeline.py`, `scripts/run_experiment.py`,
`configs/baseline.yaml`, `tests/test_geometry.py`

**2. Оцінювання, синтетика, дані** (12 файлів, ~660 рядків):
`src/vo/eval.py`, `src/vo/benchmark.py`, `src/vo/synthetic.py`, `scripts/make_synthetic.py`,
`scripts/run_benchmark.py`, `scripts/download_data.py`, `data/.gitkeep`, `data/manifest.yaml`, `configs/orb_bf.yaml`,
`configs/akaze_bf.yaml`, `configs/rootsift_magsac_opt.yaml`, `tests/test_pipeline_smoke.py`

**3. Детектори, дескриптори, матчери** (15 файлів, ~575 рядків):
`src/vo/features/__init__.py`, `src/vo/features/base.py`, `src/vo/features/classical.py`,
`src/vo/features/learned.py`, `src/vo/matching/__init__.py`, `src/vo/matching/base.py`,
`src/vo/matching/descriptor.py`, `src/vo/matching/learned.py`, `requirements-learned.txt`,
`configs/superpoint_lightglue.yaml`, `configs/aliked_lightglue.yaml`, `configs/disk_lightglue.yaml`,
`configs/loftr.yaml`, `configs/xfeat.yaml`, `configs/superglue.yaml`

**4. Мозаїка, візуалізація, документація** (10 файлів, ~770 рядків):
`src/vo/mosaic.py`, `src/vo/trajectory.py`, `scripts/extract_frames.py`, `CLAUDE.md`, `docs/ROADMAP.md`,
`docs/TEAM.md`, `docs/presentation/notes.md`, `docs/presentation/figures/.gitkeep`,
`docs/presentation/media/.gitkeep`, `notebooks/.gitkeep`

**Порядок комітів: 1 → 3 → 2 → 4.**
- `src/vo/__init__.py` імпортує весь конвеєр, тому `import vo` і `pytest` запрацюють лише після четвертого коміту.
- Перед кожним комітом — `git pull`, після останнього — `pytest`.

---

## День 1: спільний контракт (до початку паралельної роботи)

| Файл | Статус | Власник | Що містить |
|---|---|---|---|
| `src/vo/schema.py` | **новий** | 1 (рев'ю: усі) | Колонки і читання/запис `trajectory.csv` та GT CSV, dataclass `MapFix` |
| `src/vo/config.py` | змінити | 1 (рев'ю: усі) | Порожні секції `DEFAULTS`: `map`, `localize`, `tracking`, `smoothing`, `video`, `viz` |

Формат `trajectory.csv` (пише людина 1, читають 2 і 4):
`frame, name, x, y, heading_deg, scale, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, n_inliers, status`.
- Координати — у пікселях `global_map.png`.
- `c0..c3` — кути контуру кадру.
- `status` ∈ `map | vo | interp | lost`.

Формат GT CSV (пише людина 2): `name, x, y, heading_deg, c0x … c3y, source` (`synthetic | manual`).

До готовності реального коду люди 2 і 4 працюють на фіктивному `trajectory.csv`, який згенеровано із синтетичного GT.

---

## Людина 1: локалізація на мапі

**Створює:**

| Файл | Призначення |
|---|---|
| `src/vo/mapindex.py` | `MapIndex`: завантаження мапи, піраміда масштабів, тайли з перекриттям, кеш ознак мапи (`data/cache/`), вирізання ROI |
| `src/vo/localize.py` | Кадр → мапа: глобальна ініціалізація (масштаби × повороти), локальний трекінг у ROI, поворот кадру за курсом, перевірка правдоподібності, `run_localization(cfg)` |
| `src/vo/filtering.py` | Відсів викидів (стрибки, узгодженість з VO), інтерполяція пропусків, Kalman + RTS, Savitzky–Golay |
| `scripts/localize.py` | Головна точка входу завдання: `--map --frames/--video --config --out` → `trajectory.csv`, `route.png` |
| `configs/map/baseline_sift.yaml` | Бейзлайн: SIFT + FLANN + ratio 0.75 + MAGSAC++, вся мапа |
| `configs/map/sift_tracking.yaml` | + трекінг у ROI і поворот за курсом |
| `configs/map/sift_tracking_fusion.yaml` | + злиття з VO і згладжування |
| `tests/test_mapindex.py` | Тайли й ROI покривають мапу, координати тайл → мапа правильні |
| `tests/test_localize.py` | Локалізація на синтетиці над мапою: похибка центру < N px |
| `tests/test_filtering.py` | Згладжування й інтерполяція на зашумленій траєкторії |

**Змінює:**
- `src/vo/graph.py` — якорі «кадр↔мапа» (`MapAnchor`) і їхні залишки в `optimize`;
- `src/vo/geometry.py` — помічники `footprint(H, size)`, `heading(H)`, `scale(H)`.

**Залежить від:** 3 (`tiling.py`, навчені матчери), 4 (`VideoSource` у `video.py`, `render_route` у `viz.py`).
До їхньої готовності використовує SIFT і простий `plot_xy`.

---

## Людина 2: оцінювання і бенчмарк

**Створює:**

| Файл | Призначення |
|---|---|
| `src/vo/degrade.py` | Імітація зйомки з екрана: муар, розмиття в русі, відблиск, гамма і баланс білого, JPEG, рамка монітора |
| `src/vo/plots.py` | Графіки для звіту й слайдів: Парето (точність vs швидкість), success rate за порогами, похибка по кадрах |
| `scripts/annotate_gt.py` | Інструмент ручної розмітки (OpenCV): точки кадр↔мапа → гомографія → рядок GT CSV |
| `scripts/evaluate.py` | `trajectory.csv` + GT → `metrics.json` (оцінка будь-якого запуску окремо від конвеєра) |
| `scripts/make_report.py` | `summary.csv` → таблиці й графіки в `experiments/report/` і `docs/presentation/figures/` |
| `configs/datasets.yaml` | Набори для бенчмарку: `crops`, `synthetic_easy`, `synthetic_hard`, `synthetic_screen`, `video_*` (шляхи до кадрів і GT) |
| `data/gt/crops_gt.csv` | Ручний GT для 50 кадрів (комітиться: малий файл) |
| `data/gt/video_*_gt.csv` | GT ключових кадрів власних відео (після етапу 4) |
| `tests/test_degrade.py` | Ефекти детерміновані при фіксованому seed, розмір і тип зображення зберігаються |
| `tests/test_eval_localization.py` | Метрики на відомих прикладах (зсув, поворот, success rate) |

**Змінює:**
- `src/vo/eval.py` — похибка центру, кутів і курсу, success rate @ 5/10/25 px, частки статусів;
- `src/vo/synthetic.py` — використання `degrade.py`, кути контуру в GT, пресети `easy` / `hard` / `screen`;
- `scripts/make_synthetic.py` — прапорець `--preset`;
- `src/vo/benchmark.py` — нові колонки `COLUMNS`;
- `scripts/run_benchmark.py` — матриця «конфіг × набір даних», режим мапи;
- `.gitignore` — виняток `!data/gt/`.

---

## Людина 3: навчені методи і GPU

**Створює:**

| Файл | Призначення |
|---|---|
| `src/vo/tiling.py` | `extract_tiled(extractor, image, tile, overlap)`: ознаки великої мапи по тайлах для 4 GB VRAM (використовує `mapindex.py`) |
| `src/vo/gpu.py` | Вибір пристрою, fp16/autocast, вимір пікової VRAM, фолбек при OOM (менший розмір або CPU) |
| `src/vo/matching/dense.py` | Нові методи без детектора: Efficient LoFTR, tiny-RoMa / RoMa |
| `src/vo/matching/rotation.py` | Обгортка для методів, не інваріантних до повороту: тестові повороти (4 або 8) і деротація за заданим курсом |
| `scripts/download_weights.py` | Попереднє завантаження ваг (LightGlue, LoFTR, XFeat) і клонування SuperGlue в `third_party/` |
| `scripts/match_pair.py` | Візуалізація inliers для пари кадр↔мапа або кадр↔кадр (налагодження + приклади для слайдів) |
| `environment.yml` | Conda-середовище для здачі (вимога завдання) |
| `configs/map/superpoint_lightglue.yaml` | + аналогічно: `aliked_lightglue`, `disk_lightglue`, `loftr`, `eloftr`, `xfeat`, `xfeat_star`, `superglue`, `roma_tiny` |
| `tests/test_learned.py` | Форми й типи виходів кожного методу; пропускається без torch / lightglue або GPU |

**Змінює:**
- `src/vo/features/learned.py`, `src/vo/features/__init__.py` — fp16, XFeat і DeDoDe як екстрактори, реєстрація;
- `src/vo/matching/learned.py`, `src/vo/matching/__init__.py` — налаштування LightGlue і SuperGlue, реєстрація нових
  методів у `build_pair_matcher`;
- `requirements-learned.txt` — фінальні версії;
- наявні VO-конфіги `configs/*_lightglue.yaml`, `loftr.yaml`, `xfeat.yaml`, `superglue.yaml` — перевірити, що
  запускаються.

---

## Людина 4: відео, візуалізація, презентація

**Створює:**

| Файл | Призначення |
|---|---|
| `src/vo/video.py` | `VideoSource` з тим самим інтерфейсом, що `FrameSource` (`len`, `load`, `size`, `name`): крок кадрів, відбір різких кадрів (дисперсія лапласіана), пошук чотирикутника екрана / маска, фільтр проти муару |
| `src/vo/viz.py` | Фінальне зображення маршруту: пронумеровані точки, градієнт за часом, стрілки, START/FINISH, контури кадрів, колір за статусом, легенда, збільшена вставка |
| `scripts/render_route.py` | `trajectory.csv` + мапа → `route.png` |
| `scripts/render_video.py` | Відео: кадр камери + мапа з траєкторією й поточною позицією + збільшена ділянка → MP4 і GIF для слайдів |
| `scripts/make_slides_data.py` | `summary.csv` / `metrics.json` → `docs/presentation/data.json` |
| `docs/RECORDING.md` | Протокол зйомки відео (екран, масштаб, налаштування телефона, сценарії `easy` / `hard` / `loop`) |
| `docs/presentation/package.json` | Зафіксовані `reveal.js@5.1.0`, `decktape` |
| `docs/presentation/index.html` | Слайди reveal.js |
| `docs/presentation/css/theme.css` | Тема презентації |
| `.claude/launch.json` | Конфіг сервера перегляду презентації для вбудованого браузера |
| `tests/test_video.py` | Відбір кадрів, маска екрана на синтетичному кадрі з рамкою |
| `tests/test_viz.py` | `render_route` на фіктивному `trajectory.csv` створює зображення потрібного розміру |

**Змінює:**
- `README.md` — повний опис рішення (вимога завдання);
- `data/manifest.yaml` — записи для власних відео;
- `scripts/extract_frames.py` — опція відбору різких кадрів через `video.py`.

**Дані (не файли репозиторію):** 2–3 відео на спільному Google Drive.

---

## Спільні файли

| Файл | Хто змінює | Правило |
|---|---|---|
| `src/vo/config.py` | усі | Кожен змінює лише свою секцію `DEFAULTS` (1: `map/localize/tracking/smoothing`, 2: `eval`, 3: `features/matching`, 4: `video/viz`) |
| `src/vo/schema.py` | 1, рев'ю всіх | Змінюється лише за згодою всіх: від нього залежать 2 і 4 |
| `docs/ROADMAP.md`, `CLAUDE.md` | усі | Відмічати свої пункти; великі зміни — через обговорення |
| `docs/presentation/notes/<n>-<роль>.md` | кожен свій | `1-localization.md`, `2-eval.md`, `3-learned.md`, `4-video-viz.md` замість спільного `notes.md` |
| `docs/presentation/figures/` | усі | Префікс імені файлу = номер ролі (`1_route_baseline.png`, `2_pareto.png`) |

## Файли, які не змінюються без потреби

`src/vo/io.py`, `preprocess.py`, `mosaic.py`, `pipeline.py`, `trajectory.py` (VO між кадрами), `features/base.py`,
`features/classical.py`, `matching/base.py`, `matching/descriptor.py`, `scripts/download_data.py`.
Якщо зміна все-таки потрібна — окремий PR з рев'ю власника відповідної ролі.

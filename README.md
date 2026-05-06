# Семантические иерархии в языковых моделях

Ноутбуки к дипломной работе: зондирование иерархической структуры WordNet
в моделях Pythia с помощью TransformerLens.

**Модели:** `EleutherAI/pythia-160m` (12 слоёв, d=768) и `EleutherAI/pythia-1b` (16 слоёв, d=2048)  
**Библиотеки:** TransformerLens · PyTorch · NumPy · SciPy · Matplotlib · UMAP · NLTK

---

## Структура репозитория

```
repo/
├── 01_dataset.ipynb          # Построение датасета
├── 02_exps_1_2_3.ipynb       # Эксперименты 1–3: RSA, асимметрия, ортогональность
├── 03_exp_4.ipynb             # Эксперимент 4: динамика обучения
├── 04_exp_5.ipynb             # Эксперимент 5: декомпозиция MLP vs Attention
├── 05_exp_6.ipynb             # Эксперимент 6: типичность (Orthogonality Ratio)
└── outputs/                   # Все графики и CSV
```

> **Порядок запуска:** `01` → `02` → `03` → `04` → `05`.  
> Ноутбуки `03–05` требуют `outputs/dataset.json`, созданного в `01`.

---

## Ноутбуки и визуализации

### 01 — Построение датасета WordNet

`01_dataset.ipynb`

Фильтрация WordNet по 5 семантическим категориям (animal, food, furniture, tool, vehicle),
проверка 4-уровневой гиперонимической цепочки, балансировка по уровням.

| Файл | Описание |
|------|----------|
| `outputs/dataset.json` | 123 слова с уровнями WordNet и категориями |
| `outputs/dataset_report.txt` | Отчёт о распределении по уровням и категориям |

---

### 02 — Эксперименты 1–3: RSA, асимметрия иерархии, ортогональность направлений

`02_exps_1_2_3.ipynb`

**Эксперимент 1 — RSA по слоям.**
Spearman ρ между матрицей косинусных сходств активаций и GT-матрицей WordNet
$$GT[i,j] = \frac{1}{1 + |level_i - level_j|}$$
для слоёв 0–12 (pythia-160m). Пермутационный тест (1 000 перестановок) на каждом слое.

| Файл | Описание |
|------|----------|
| `outputs/ext_rsa_per_layer_160m.png` | RSA-кривая ρ по слоям с доверительными интервалами |
| `outputs/ext_rsa_per_layer_160m.csv` | Числовые значения ρ и p-value по слоям |

**Эксперимент 2 — Асимметрия иерархии.**
Проверяем, что пары слов с большей разницей уровней (upper–lower) имеют меньшее сходство.
Mann-Whitney U (pythia-160m L8: p=0.014 *).

| Файл | Описание |
|------|----------|
| `outputs/ext_asymmetry_160m.png` | KDE-распределение сходств upper vs lower пар + скобка значимости |

**Эксперимент 3 — Ортогональность направлений.**
Косинус между разностными векторами смежных уровней иерархии.
Наблюдаемый \|cos\|=0.401 против теоретического базиса √(2/πd)=0.029 (14×).
Wilcoxon p=1.82e-12 ***.

| Файл | Описание |
|------|----------|
| `outputs/ext_orthogonality_160m.png` | Бар-чарт \|cos\| по парам с базисом 0.029 |
| `outputs/ext_orthogonality_160m.csv` | \|cos\| для каждой пары направлений |

**Дополнительные визуализации (из replot-скриптов):**

| Файл | Описание |
|------|----------|
| `outputs/ext_rdm_160m.png` | RDM heatmap — обученная модель (pythia-160m L8) |
| `outputs/ext_rdm_random_160m.png` | RDM heatmap — случайно инициализированная модель |
| `outputs/ext_rdm_gt.png` | GT-матрица WordNet (эталонная структура) |
| `outputs/ext_umap_160m.png` | UMAP-проекция активаций L8 с цветовой разметкой по категориям |

---

### 03 — Эксперимент 4: динамика обучения, фазовый переход

`03_exp_4.ipynb`

RSA (Spearman ρ) на 31 чекпоинте от step 0 до step 143 000 для pythia-160m и pythia-1b.
Гипотеза о фазовом переходе около step 1 000 (~1% обучения),
совпадающем с формированием induction heads (Olsson et al., 2022).

| Файл | Описание |
|------|----------|
| `outputs/checkpoint_rsa_160m.csv` | ρ по слоям для каждого чекпоинта pythia-160m |
| `outputs/checkpoint_rsa_1b.csv` | То же для pythia-1b |
| `outputs/checkpoint_curve_160m.png` | RSA-кривая pythia-160m по чекпоинтам (symlog шкала) |
| `outputs/checkpoint_curve_1b.png` | RSA-кривая pythia-1b по чекпоинтам |
| `outputs/comparison_160m_vs_1b.png` | Сравнение 160m vs 1b на слое L8 |

---

### 04 — Эксперимент 5: декомпозиция MLP vs Attention

`04_exp_5.ipynb`

Вклад MLP-дельт и Attention-дельт в RSA по слоям.
Paired permutation test на разность ρ_MLP − ρ_Attn; Wilcoxon signed-rank глобально.
Результаты: 160m Wilcoxon p=0.212 (n.s.); 1b p=0.0017 (**).

| Файл | Описание |
|------|----------|
| `outputs/decomp_layer_profile_160m.png` | Профиль ρ MLP / Attn / resid по слоям, pythia-160m |
| `outputs/decomp_layer_profile_160m.csv` | Числовые значения |
| `outputs/decomp_layer_profile_1b.png` | То же для pythia-1b |
| `outputs/decomp_layer_profile_1b.csv` | |
| `outputs/decomp_l8_comparison.png` | Сравнение 160m vs 1b на L8: MLP и Attn |
| `outputs/decomp_checkpoint_160m.png` | Динамика MLP/Attn RSA по чекпоинтам, pythia-160m |
| `outputs/decomp_checkpoint_160m.csv` | |
| `outputs/decomp_checkpoint_1b.png` | То же для pythia-1b |
| `outputs/decomp_checkpoint_1b.csv` | |
| `outputs/decomp_checkpoint_comparison.png` | Сравнительный график динамики 160m vs 1b |

---

### 05 — Эксперимент 6: типичность (Orthogonality Ratio)

`05_exp_6.ipynb`

Метрика OR измеряет долю нормы вектора слова, лежащую перпендикулярно гипернониму:

$$OR(w, h) = \frac{|w - \text{proj}_h(w)|}{|w|} \in [0, 1]$$

Гипотеза: атипичные члены категории (penguin, platypus, skateboard…) имеют
более высокий OR, чем типичные (dog, horse, bicycle…).
Pooled Mann-Whitney по 5 категориям: p=0.0017 **.

| Файл | Описание |
|------|----------|
| `outputs/typicality_OR_profiles_160m.png` | OR-кривые по слоям для 5 категорий (типичные vs атипичные) |
| `outputs/typicality_animal_focus_160m.png` | OR и cos(hypernym) на L8 для категории animal |
| `outputs/typicality_pooled_160m.png` | Boxplot + jitter: типичные vs атипичные, все 5 категорий |

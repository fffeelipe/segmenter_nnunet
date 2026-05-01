# Análisis de resultados — Dataset501 ALT T1

Este documento resume el análisis del dataset `Dataset501_ALT_T1`
(46 pacientes, secuencia T1, segmentación binaria de Tumores
Lipomatosos Atípicos).

- **§1–§5**: diagnóstico de los fallos del baseline DA5 100ep y
  plan mínimo (ya implementado).
- **§6–§7**: resultados 5-fold de `ALT_os033_250ep` (plan mínimo
  ganador) y del ensemble. **Cierre de Semana 1**.
- **§8**: plan de Semana 2, experimentos priorizados y estado del
  código. Punto de partida para retomar el trabajo.

## TL;DR — Semanas 1–2: cierre de pretrain y gate sigmoide


| Configuración (n=46)                                                          | Dice 5-fold                | Δ vs DA5 baseline         | Estado         |
| ----------------------------------------------------------------------------- | -------------------------- | ------------------------- | -------------- |
| DA5_100ep baseline `2d`                                                       | 0.5924                     | —                         | ref            |
| DA5_100ep baseline `3d_fullres`                                               | 0.6370                     | —                         | ref            |
| ALT_os033_250ep `2d` (nopretrain)                                             | 0.7235                     | +0.131                    | actual         |
| ALT_os033_250ep `3d_fullres` (TS-MRI pretrain)                                | 0.7076                     | +0.071                    | actual         |
| Ensemble soft-avg (nnU-Net find_best)                                         | 0.7247                     | +0.0877 vs 2D baseline    | —              |
| Ensemble gated v2 (abs+rel, ρ=0.10)                                           | 0.7730                     | +0.181 vs 2D baseline     | —              |
| Ensemble gated v2b (hard, T=50, ρ=0.40)                                       | 0.7805                     | +0.188 vs 2D baseline     | reemplazado    |
| **Ensemble gated v2c (sigmoid, v_min=1000, τ=20, use-conf) ← ganador actual** | **0.7822**                 | **+0.190 vs 2D baseline** | **producción** |
| Oracle `max(2D, 3D)` (techo sin reentrenar)                                   | 0.7933                     | —                         | referencia     |
| Exp. 1 full pretrain 2D (5-fold)                                              | 0.7155 solo / 0.7749 gated | −0.008 / −0.006           | **archivado**  |
| Exp. 1a K=1 shallow (fold 2)                                                  | 0.5414                     | −0.085 vs nopre           | **archivado**  |
| Exp. 1a mix A=0.5 (fold 2)                                                    | 0.6102                     | −0.016 vs nopre           | **archivado**  |
| T2-only ALT_os033_250ep `3d_fullres` (Dataset502, TS-MRI pretrain)            | 0.6198                     | −0.088 vs T1/3D           | **archivado**  |
| T1+T2 union ALT_os033_250ep `3d_fullres` (Dataset503, no pretrain)            | 0.6038                     | −0.104 vs T1/3D           | **archivado**  |
| T1+T2 **union_v2** ALT_os033_250ep `3d_fullres` (Dataset503, GT fusionado)    | 0.7268*                    | n/a                       | multichannel (503) |
| **T1+T2 union_v2 baseline (503) = `fusion_union_v2_503_gated_v2c`**           | **0.8032**                 | n/a                       | **baseline (503)** |


**Estado al cierre de Semana 2 (Exp. 2a)**: el gate sigmoide con
tie-breaker por confianza media en la región fg levanta +0.0017 sobre
el hard gate (0.7822 vs 0.7805). Cierra ~13 % de la brecha restante al
oracle (0.7933). Es +0.0017 reproducible, gratis y sin reentrenar, así
que pasa a producción como **gated v2c**.

**Cierre de la rama pretrain**: tres variantes (Exp. 1 full, 1a-K,
1a-mix) todas regresan respecto al nopretrain. El 3D→2D slicing de
TS-MRI no genera features útiles para IO T1w.

**Estado Exp. 2b (augment de inversión de intensidad) — CERRADO**:

- **Exp. 2b (`inv`, linear flip p=0.15)**: sanity fold 2 **falla**
(fold 2 mean 0.590 vs baseline 0.627, −0.037). Archivado. Dejó
el insight de que IOG38 sí sube +0.16 con flip.
- **Exp. 2b' (`invgamma`, boost Gamma(p_invert=1) de 0.1→0.2)**:
fold 2 mean +0.044, fold 0 mean −0.021 (inestable). **Gated test
decisivo** (invgamma 2D + baseline 3D vs v2c sobre mismos 19
casos): **−0.0177 mean**. Los rescates grandes en 2D (IOG28 +0.78,
IOG36 +0.20) caen en B-regime y no se transmiten al gated. Las
regresiones (IOG35 −0.18, IOG31 −0.25) sí se transmiten. Única
ganancia real: IOG38 +0.155. Rama **archivada**.
- **Exp. 2c (case-aware sampler)**: descartado tras el gated test
(ganancia esperada solo por IOG38 = +0.003 en 5-fold; ver §8.2.3).

**Siguiente paso activo**: Exp. 3 (cascada detect→seg) para atacar
IOG40, IOG45, IOG1 (fallos duros) y potencialmente IOG38.

- **Exp. 3-B1 ejecutado y ARCHIVADO (2026-04-23)**: variante "barata"
`oversample=1.0 + GT dilate 1 vóxel` (trainer
`nnUNetTrainerALT_os1_dilate1_250epochs`, 5-fold en 2× RTX 5080).
Standalone 3D cae a 0.558 (−0.15 vs 0.708 del 3D base). Mejor
variante gated (sigconf v_min=50 τ=200 conf²) = **0.7086**, −0.0736
vs v2c. Gana en IOG10 (+0.26) y IOG1 (+0.16) pero 9 regresiones
masivas (IOG4 −0.67, IOG48 −0.63, IOG36 −0.61…) lo sepultan.
IOG38/40/45 siguen en 0. Ver §8.2 `Exp. 3-B1`.
- **Fallos absolutos residuales (Dice = 0)**: IOG40, IOG45. Ambos
  modelos fallan; sus tumores tienen solo 156 y 182 voxels
  predichos respectivamente → requieren cascada detect→seg en
  Semana 2.
- **Resultados DA5 100ep** viven en `nnunet_env_base/` (congelado).
- **Resultados ALT os033 250ep** viven en `nnunet_env/` (actual).
- **Scripts nuevos relevantes**: `scripts/ensemble_gated.py`
  (ensemble gated), `scripts/summarize_5fold.py` (reportes).

Configuraciones baseline evaluadas (DA5, 100 epochs por fold):


| Config       | Plans              | Pretrain             | Dice medio 5-fold |
| ------------ | ------------------ | -------------------- | ----------------- |
| `2d`         | `nnUNetPlans`      | ninguno              | **0.591**         |
| `3d_fullres` | `nnUNetTSMRIPlans` | TotalSegMRI task 852 | **0.636**         |


## 1. Pacientes con Dice ≈ 0 y características

Los casos problemáticos comparten un patrón muy claro: son tumores
**pequeños, finos y/o de bajo contraste**, casi siempre dominados por
falsos negativos (`n_pred ≪ n_ref`).


| Caso  | Fold | Dice 2D | Dice 3D | `tumor_vox` | slices c/tumor | Spacing in-plane (mm) | Contraste* |
| ----- | ---- | ------- | ------- | ----------- | -------------- | --------------------- | ---------- |
| IOG45 | 0    | 0.000   | 0.000   | **405**     | 4              | 1.07                  | 1.23       |
| IOG35 | 0    | 0.076   | 0.000   | 9 221       | 8              | 0.37                  | 2.27       |
| IOG36 | 1    | 0.000   | 0.786   | 14 985      | 27             | 0.78                  | 3.39       |
| IOG4  | 1    | 0.000   | 0.609   | 6 085       | 5              | 0.31                  | 1.35       |
| IOG31 | 1    | 0.074   | 0.066   | 36 466      | 7              | **0.23**              | 2.82       |
| IOG1  | 2    | 0.000   | 0.000   | 12 945      | 15             | 0.62                  | 2.01       |
| IOG38 | 2    | 0.109   | 0.000   | 5 232       | 5              | 1.25                  | **−0.06**  |
| IOG40 | 2    | 0.000   | 0.000   | 4 508       | 5              | 0.83                  | 1.01       |
| IOG28 | 3    | 0.002   | 0.073   | 80 445      | 7              | **0.31**              | 1.67       |
| IOG29 | 3    | n/a     | 0.000   | 2 073       | 9              | 1.0                   | —          |
| IOG43 | 4    | 0.000   | 0.586   | 14 336      | 7              | 0.55                  | 3.13       |
| IOG48 | 4    | 0.394   | 0.000   | 8 661       | —              | —                     | —          |
| IOG9  | 4    | 0.259   | 0.160   | 50 547      | 22             | 0.78                  | 1.77       |


`*` Contraste estimado como `(mean_tumor − mean_background) / std_background`.

### Patrones detectados

1. **Tumores muy pequeños** (≤ 15 k voxels, 4–8 slices) concentran los
   fallos: IOG45, IOG29, IOG38, IOG40, IOG1, IOG4, IOG43, IOG35.
   Con `batch_dice=True` y `oversample_foreground_percent=0.33` (defaults
   de nnU-Net 2D) estas lesiones casi nunca aparecen en los parches del
   lote y pesan poco en el gradiente.
2. **IOG38 tiene contraste T1 invertido** (tumor más oscuro que el tejido
   adyacente, SNR = −0.06). El modelo aprendió que ALT = tejido hiperintenso.
3. **Spacing in-plane extremo** (IOG28 0.31 mm, IOG31 0.23 mm) implica un
   submuestreo de 2.5×–3.4× durante el resampling al spacing objetivo de
   0.78 mm. La lesión delgada se "difumina".
4. **IOG1 es el único fallo "sin razón aparente"** (12 k voxels, 15 slices,
   contraste 2.0): requiere inspección visual de la GT / localización.
5. **IOG47 tiene un bug de dtype en el NIfTI original**: todos sus voxels
   caen en `[-32768, -29681]` (rango entero negativo del int16). Es un
   clásico wrap-around: valores uint16 `>= 2^15` quedan guardados como
   int16 y se leen como negativos. En el DA5 100-epoch baseline este
   caso "sobrevivía" (Dice 0.87 en fold 0) porque el contraste interno
   del tumor se preservaba aunque la imagen entera estuviera en el
   rango negativo. Un clip naive a 0 convierte la imagen en
   completamente plana (ver §3). Fix correcto: sumar `2^16` en la
   conversión para devolver el rango a uint16 sin perder contraste.
   Esto también explica el `min = −32 555` / `percentile_00_5 = −31 721`
   del fingerprint original: ese outlier venía **solo de IOG47**
   (ningún otro paciente tiene negativos en los raw). Al retirarlo,
   `std` del foreground cae de 4796 a 677.

### Heterogeneidad de adquisición (46 pacientes)


| Variable              | Rango / Valores                 |
| --------------------- | ------------------------------- |
| Spacing in-plane      | 0.23 – 1.25 mm (factor 5×)      |
| Spacing through-plane | 3.3 – 7.2 mm (factor 2×)        |
| Matriz axial          | 320² – 704²                     |
| Voxels tumorales      | 405 – 267 190 (factor **660×**) |
| Slices con tumor      | 4 – 57                          |
| Contraste tumor/bg    | −0.06 – 4.23                    |


### 2D ↔ 3D: failures complementarios

Hay casos donde **2D acierta y 3D falla**, y viceversa:

- `IOG35`: 2D 0.08 / 3D 0.00 (3D sobre-segmenta: 44 774 vs 9 221 voxels GT)
- `IOG48`: 2D 0.39 / 3D 0.00
- `IOG9` : 2D 0.26 / 3D 0.16
- `IOG36`: 2D 0.00 / 3D 0.79
- `IOG4` : 2D 0.00 / 3D 0.61
- `IOG7` : 2D 0.20 / 3D 0.85

→ **Ensamblar 2D + 3D** es la mejora de mayor retorno por menor esfuerzo.

## 2. Causas raíz probables


| #   | Causa                                                                                                              | Evidencia                                                                                           |
| --- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 1   | **El 2D no tiene pretraining.** TotalSegMRI solo publica 3d_fullres; el 2D se entrena desde cero en este pipeline. | `scripts/fetch_totalseg_mri.py` lo documenta. Dice 2D < 3D en la mayoría de folds.                  |
| 2   | **Desbalance por volumen**: `batch_dice=True` en 2D hace que los tumores pequeños pesen muy poco en el loss.       | 7 de 9 casos fallidos tienen `n_ref ≤ 15 k`.                                                        |
| 3   | **Sampling foreground insuficiente** (33 %) para un dataset tan desbalanceado.                                     | Lesiones de 4–5 slices prácticamente no aparecen en los lotes.                                      |
| 4   | **Solo 100 epochs**. El pseudo-Dice EMA seguía subiendo (0.82 al final del fold 0).                                | Log de entrenamiento.                                                                               |
| 5   | **Splits aleatorios sin estratificar**.                                                                            | Fold 2 recibió IOG1+IOG38+IOG40 (todos pequeños / bajo contraste) y colapsa a 0.537 / 0.557.        |
| 6   | **Outlier espurio en el fingerprint** (`−32 555`).                                                                 | Raw min = 0 en todos los casos inspeccionados; fingerprint dice −32 555. Z-Score global se degrada. |
| 7   | **Resampling agresivo** para pacientes con spacing in-plane muy fino (0.23–0.31 mm).                               | IOG28, IOG31 → ambos fallan en 2D y 3D.                                                             |


## 3. Plan mínimo (implementado en este repo)

El objetivo es atacar las causas 1–6 con cambios compactos y reversibles.
Todos los artefactos quedan bajo un trainer / plans / splits diferentes,
sin romper las corridas ya existentes.

1. **Fix selectivo de intensidades en la conversión** → detecta el
   wrap-around uint16→int16 de IOG47 y le suma `2^16` para recuperar su
   rango correcto; solo clipea a 0 los voxels aislados negativos en
   casos con `min < 0` pero `max > 0` (nunca en una imagen
   enteramente negativa). Un clip global a 0 destruye IOG47 (Dice
   0.87 → 0.00 observado en la primera corrida ALT).
   *(modifica `scripts/convert_to_nnunet.py`: función `_fix_intensity`)*
2. **Trainer custom `nnUNetTrainerALT_250epochs`**: 250 epochs,
   `oversample_foreground_percent = 0.66` y loss DiceCE con `batch_dice=False`
   forzado. *(nuevo: `custom_trainers/nnUNetTrainerALT.py`)*
3. **Patch del plan 2D** (`nnUNetPlans.json`) para dejar `batch_dice=False`
   también persistido en disco. *(nuevo: `scripts/patch_plans.py`)*
4. **Splits estratificados** por volumen tumoral (terciles) + por rango de
   contraste. *(nuevo: `scripts/make_stratified_splits.py`)*
5. **Modo T1-only** en el pipeline, para poder iterar solo sobre el dataset
   501 mientras terminamos la estrategia de T2. *(flag `T1_ONLY=1` en
   `run_training.sh` y `scripts/convert_to_nnunet.py --modalities T1`)*
6. **Ensemble 2D + 3D** ya resuelto por `nnUNetv2_find_best_configuration`
   al final del script; no requiere cambios.

Cambios **no incluidos** en el plan mínimo (seguimiento propuesto):

- Pretrain 2D inicializando desde el encoder 3D de TotalSegMRI (requiere
  conversión `Conv3d → Conv2d` de pesos; es un proyecto aparte).
- Preprocesador custom con clipping por percentiles 0.5/99.5 antes del
  Z-Score.
- Cascada detection → segmentation para lesiones < 1 cm³ (IOG45).
- Data augmentation de inversión de intensidad para casos hipointensos
  tipo IOG38.
- Revisión manual de GT de IOG1 y IOG28 (probable problema clínico o de
  etiqueta).

## 4. Cómo correr el plan mínimo

```bash
# (Opcional) reiniciar completamente la conversión y los plans para que los
# cambios de clipping y splits surtan efecto:
rm -rf nnunet_env/nnUNet_raw/Dataset501_ALT_T1
rm -rf nnunet_env/nnUNet_preprocessed/Dataset501_ALT_T1
rm -rf nnunet_env/nnUNet_results/Dataset501_ALT_T1

# Entrenamiento T1 con el plan mínimo (post-ablación, oversample 0.33):
T1_ONLY=1 EPOCHS=250 TRAINER=ALT bash run_training.sh

# Variante legacy con oversample 0.66 (solo para reproducir corridas
# anteriores; regresa varios casos, no recomendada).
T1_ONLY=1 EPOCHS=250 TRAINER=ALT_OS066 bash run_training.sh
```

Variables honradas:

- `T1_ONLY=1` → convierte solo `T1/` y entrena solo `Dataset501_ALT_T1`.
- `TRAINER=ALT` (o `ALT_OS033`, alias) → usa
  `nnUNetTrainerALT_os033_250epochs` (plan mínimo ganador: oversample
  0.33, batch_dice=False, 250 épocas, splits estratificados, plans
  parcheados, fix de intensidad IOG47). Es el default recomendado
  post-ablación.
- `TRAINER=ALT_OS066` → usa `nnUNetTrainerALT_250epochs` (oversample
  0.66). Conservado por reproducibilidad; en nuestros datos regresa
  IOG47/IOG36/IOG24.
- `TRAINER=DA5` (default histórico) → mantiene el comportamiento
  baseline.
- `EPOCHS=250` → válido con `TRAINER=ALT`, `ALT_OS033` y `ALT_OS066`
  (también admite `500`).
- `STRATIFY_SPLITS=1` (default con cualquier `TRAINER=ALT*`) →
  reemplaza `splits_final.json` por el split estratificado por volumen.

## 5. Resultados observados del plan mínimo (ablación)

La primera corrida del plan mínimo con `oversample=0.66` produjo una
regresión de ~0.10 Dice respecto al baseline DA5 en 2 folds. La
ablación con `oversample=0.33` (manteniendo todo lo demás del plan
mínimo) lo revirtió y además mejora el baseline. Medido sobre los
**mismos 19 pacientes de validación** (folds 0+1 estratificados):


| Configuración                            | Dice medio | Δ vs DA5 100ep |
| ---------------------------------------- | ---------- | -------------- |
| DA5 100ep, oversample 0.33 (baseline)    | 0.641      | —              |
| ALT 250ep, **oversample 0.66**           | 0.542      | **−0.099**     |
| ALT 250ep, **oversample 0.33** (ganador) | **0.694**  | **+0.053**     |


### Hallazgos clave

1. **El oversample 0.66 era el culpable de la regresión.** Con solo ~36
   imágenes de entrenamiento y 2.5× más épocas, forzar que 2/3 de los
   parches estén centrados en foreground hace que el modelo pierda
   contexto background y se sobre-especialice. Bajarlo a 0.33 recupera
   ~0.15 Dice sin tocar nada más.
2. **El fix `_fix_intensity` para IOG47 funciona como se esperaba.**
   Dice de IOG47: 0.871 (DA5) → 0.000 (clip naive) → 0.850 (fix
   wraparound). El caso estaba almacenado como uint16-saved-as-int16;
   sumar `2^16` recupera el contraste tumoral.
3. **Casos históricamente difíciles mejoran sustancialmente:**
   - IOG29: 0.000 → 0.380 (tumor de 2 073 voxels)
   - IOG9 : 0.160 → 0.564
   - IOG20: 0.531 → 0.809
   - IOG24: 0.379 → 0.612
4. **Regresiones puntuales que quedan abiertas** (ver §6):
   IOG10 (0.714 → 0.589), IOG4 (0.609 → 0.486). IOG45 e IOG48 siguen
   en 0.

El `TRAINER=ALT` del pipeline queda re-apuntado a este ganador
(`nnUNetTrainerALT_os033_250epochs`).

## 6. Resultados 5-fold completos: ALT vs DA5 (terminado)

La 5-fold CV con `TRAINER=ALT` (alias de
`nnUNetTrainerALT_os033_250epochs`) ya terminó en ambas configs.
Los números están medidos sobre los **46 pacientes** (todos los folds
concatenados), usando los splits estratificados por volumen tumoral.

**Convenio de entornos en disco (para reproducir):**

- `nnunet_env_base/` → resultados DA5 100ep (previo; congelado).
- `nnunet_env/`      → resultados ALT os033 250ep (actual).

### 6.1. Resumen agregado


| Config       | Trainer                       | Dice 5-fold | Δ vs DA5   | Fails (=0) | Low (0<D<0.5) |
| ------------ | ----------------------------- | ----------- | ---------- | ---------- | ------------- |
| `2d`         | DA5_100ep (base)              | 0.5924      | —          | 6          | 10            |
| `2d`         | **ALT_os033_250ep (current)** | **0.7235**  | **+0.131** | **3**      | **5**         |
| `3d_fullres` | DA5_100ep (base)              | 0.6370      | —          | 7          | 6             |
| `3d_fullres` | **ALT_os033_250ep (current)** | **0.7076**  | **+0.071** | **5**      | **3**         |


Observaciones:

- **2D sube más que 3D** (+0.131 vs +0.071). Ahora 2D ≥ 3D en promedio
  (0.724 > 0.708), invirtiendo la relación del baseline (0.592 < 0.637).
  El oversample 0.33 + batch_dice=False + 250ep + splits estratificados
  benefician desproporcionadamente al 2D, porque: (i) partía sin
  pretraining y tenía mucho margen, (ii) los tumores pequeños —que
  dominan los fallos 2D— son justo los casos en los que `batch_dice=False`
  - splits balanceados ayudan.
- **Oracle `max(2d, 3d)` por caso = 0.793** sobre los 46 pacientes.
  Esa es la cota superior del ensemble por configuración: un buen
  `nnUNetv2_find_best_configuration` debería aterrizar entre **0.74 y
  0.79** según cómo se comporte el softmax-averaging en los casos
  donde una config acierta y la otra predice vacío.
- Fails residuales compartidos (2D=0 y 3D=0): **IOG40, IOG45**. Los
  demás fallos totales se "rescatan" combinando configs (ver §6.3).

### 6.2. Per-fold means (ALT vs DA5)

**2D (`nnUNetPlans`):**


| Fold | DA5 mean | ALT mean  | Δ          |
| ---- | -------- | --------- | ---------- |
| 0    | 0.634    | 0.659     | +0.024     |
| 1    | 0.624    | **0.801** | **+0.177** |
| 2    | 0.557    | 0.627     | +0.069     |
| 3    | 0.667    | 0.761     | +0.094     |
| 4    | 0.475    | **0.778** | **+0.303** |


**3D (`nnUNetTSMRIPlans 3d_fullres`):**


| Fold | DA5 mean | ALT mean  | Δ          |
| ---- | -------- | --------- | ---------- |
| 0    | 0.676    | 0.687     | +0.011     |
| 1    | 0.769    | 0.703     | −0.066     |
| 2    | 0.537    | 0.577     | +0.040     |
| 3    | 0.604    | **0.803** | **+0.199** |
| 4    | 0.594    | **0.771** | **+0.177** |


El único fold que retrocede en 3D es el 1 (−0.066). En 2D todos los
folds mejoran. Esto confirma que el efecto dominante es positivo
también a nivel de fold, no solo promedio global.

### 6.3. Cambios por caso

**Improvers (Δ ≥ +0.10 en cualquier config; unión 2D∪3D, n=20):**


| Caso  | 2D DA5→ALT                | 3D DA5→ALT                | Nota                              |
| ----- | ------------------------- | ------------------------- | --------------------------------- |
| IOG35 | 0.076 → **0.845** (+0.77) | 0.000 → 0.636 (+0.64)     | antes fail doble                  |
| IOG28 | 0.002 → 0.043             | 0.073 → **0.810** (+0.74) | spacing 0.31mm rescatado por 3D   |
| IOG43 | 0.000 → **0.754** (+0.75) | 0.586 → 0.874 (+0.29)     | antes fail 2D total               |
| IOG4  | 0.000 → **0.710** (+0.71) | 0.609 → 0.486 (−0.12)     | 2D gana decisivo                  |
| IOG31 | 0.074 → **0.746** (+0.67) | 0.066 → 0.000 (−0.07)     | solo 2D lo soluciona              |
| IOG19 | 0.428 → 0.916 (+0.49)     | 0.786 → 0.769             |                                   |
| IOG9  | 0.259 → 0.603 (+0.34)     | 0.160 → 0.564 (+0.40)     |                                   |
| IOG29 | 0.549 → 0.863 (+0.31)     | 0.000 → 0.380 (+0.38)     | 3D sale de 0                      |
| IOG54 | **0.674 → 0.228 (−0.45)** | 0.487 → 0.862 (+0.38)     | ver §6.4                          |
| IOG7  | 0.197 → 0.561 (+0.36)     | 0.853 → 0.849             |                                   |
| IOG24 | 0.571 → 0.864 (+0.29)     | 0.379 → 0.612 (+0.23)     |                                   |
| IOG50 | 0.451 → 0.743 (+0.29)     | 0.376 → 0.647 (+0.27)     |                                   |
| IOG48 | 0.394 → 0.634 (+0.24)     | 0.000 → 0.000             | 3D sigue fallando                 |
| IOG53 | 0.714 → 0.946 (+0.23)     | 0.931 → 0.924             |                                   |
| IOG26 | 0.550 → 0.739 (+0.19)     | 0.764 → 0.811             |                                   |
| IOG38 | 0.109 → 0.232 (+0.12)     | 0.000 → 0.000             | contraste invertido — persistente |
| IOG5  | 0.837 → 0.958 (+0.12)     | 0.896 → 0.923             |                                   |
| IOG49 | 0.808 → 0.932 (+0.12)     | 0.919 → 0.908             |                                   |
| IOG10 | 0.299 → 0.404 (+0.10)     | 0.714 → 0.589 (−0.12)     | regresión en 3D                   |
| IOG1  | 0.000 → 0.050             | 0.000 → 0.178 (+0.18)     | sigue muy bajo                    |


**Regressors (Δ ≤ −0.10 en alguna config):**


| Caso  | 2D DA5→ALT                | 3D DA5→ALT                | Diagnóstico                                                                    |
| ----- | ------------------------- | ------------------------- | ------------------------------------------------------------------------------ |
| IOG54 | 0.674 → **0.228** (−0.45) | 0.487 → 0.862 (+0.38)     | **2D perdió este caso** pese a ganar +0.38 en 3D. Inspección visual pendiente. |
| IOG12 | 0.915 → 0.893             | 0.869 → **0.653** (−0.22) | 3D perdió un caso "fácil".                                                     |
| IOG10 | 0.299 → 0.404 (+0.10)     | 0.714 → **0.589** (−0.12) |                                                                                |
| IOG4  | 0.000 → 0.710 (+0.71)     | 0.609 → **0.486** (−0.12) | 2D compensa con creces.                                                        |


Contando solo regresiones netas por caso (peor config de las dos),
son 4: **IOG54** (grave, solo en 2D), IOG12, IOG10, IOG4 (menores).

### 6.4. Fallos residuales (Dice = 0 en la config correspondiente)


| Caso  | 2D    | 3D    | Rescatable con ensemble | Hipótesis dominante                                                        |
| ----- | ----- | ----- | ----------------------- | -------------------------------------------------------------------------- |
| IOG45 | 0.000 | 0.000 | **No**                  | 405 voxels, 4 slices — necesita detección→segmentación.                    |
| IOG40 | 0.000 | 0.000 | **No**                  | 4 508 vox, 5 slices, contraste 1.01 — probable falta de foreground sample. |
| IOG36 | 0.000 | 0.772 | Sí (3D)                 | 2D no converge; 3D ya funciona.                                            |
| IOG48 | 0.634 | 0.000 | Sí (2D)                 | 3D predice vacío; revisar si tumor sale del patch 3D.                      |
| IOG31 | 0.746 | 0.000 | Sí (2D)                 | spacing in-plane 0.23mm; 3D colapsa al resamplear.                         |
| IOG38 | 0.232 | 0.000 | Parcial                 | contraste T1 invertido (SNR −0.06); no hay augment simétrico.              |
| IOG1  | 0.050 | 0.178 | Parcial                 | "sin causa aparente", ambas configs flojas.                                |
| IOG28 | 0.043 | 0.810 | Sí (3D)                 | spacing fino; 3D resuelve casi perfecto.                                   |


**Conclusión**: el ensemble 2D+3D rescata 4 de los 6 fails parciales
(IOG36, IOG48, IOG31, IOG28). Quedan **IOG45 e IOG40** como fallos
duros, más IOG38 e IOG1 como "muy bajos".

## 7. Recomendaciones y próximos pasos

### A corto plazo (esta semana)

1. **Ensemble 2D + 3D — hecho.** `nnUNetv2_find_best_configuration`
   dio Dice **0.7247** (apenas +0.001 vs 2D solo). El soft-average
   colapsa los casos asimétricos como anticipó el oracle: cuando una
   config predice vacío, su softmax ~0 arrastra el argmax de la otra
   bajo 0.5. Detalle:

  | Caso  | 2D    | 3D    | Ens soft-avg | Oracle | Pérdida |
  | ----- | ----- | ----- | ------------ | ------ | ------- |
  | IOG48 | 0.634 | 0.000 | 0.000        | 0.634  | −0.634  |
  | IOG31 | 0.746 | 0.000 | 0.085        | 0.746  | −0.661  |
  | IOG36 | 0.000 | 0.772 | 0.224        | 0.772  | −0.547  |
  | IOG29 | 0.863 | 0.380 | 0.575        | 0.863  | −0.288  |
  | IOG38 | 0.232 | 0.000 | 0.000        | 0.232  | −0.232  |

   Solo esos 5 casos explican casi toda la brecha del soft-avg vs
   oracle (0.793).
2. **Ensemble gated — el paso que sí mueve la aguja.** Script
   `scripts/ensemble_gated.py`. Regla: si `|pred_2d| < T` o
   `|pred_2d| < ρ · |pred_3d|` (y simétrico), tratar esa predicción
   como vacía y usar la otra; en cualquier otro caso, conservar el
   soft-avg.
   Resultados 5-fold medidos sobre 46 pacientes:

  | Configuración                         | Dice       | Δ vs soft-avg |
  | ------------------------------------- | ---------- | ------------- |
  | 2D solo                               | 0.7235     | −0.0012       |
  | 3D solo                               | 0.7076     | −0.0172       |
  | Soft-avg (nnU-Net `find_best`)        | 0.7247     | —             |
  | Gated (T=200 voxels, ρ=0)             | 0.7561     | +0.0314       |
  | Gated v2 (T=50, ρ=0.10)               | 0.7730     | +0.0483       |
  | **Gated v2b (T=50, ρ=0.40) ← actual** | **0.7805** | **+0.0558**   |
  | Oracle `max(2D, 3D)`                  | 0.7933     | +0.0686       |

   La diferencia v2 → v2b (+0.008) viene del sweep documentado en
   §8.2: subir `min-fg-ratio` de 0.10 a 0.40 filtra las predicciones
   2D "tibias" (florid pero < 40 % del volumen 3D) que degradaban el
   soft-avg, sin matar los rescates por 2D empty que ya hacía v2.
   Desglose por caso de la mejora del gated v2 (abs+rel) vs
   soft-avg: IOG31 +0.661, IOG48 +0.634, IOG36 +0.547, IOG38
   +0.232, IOG28 +0.148. El resto de los 41 casos quedan idénticos
   al soft-avg. Suma: 2.222 / 46 = +0.0483 ✓.
   La regla absoluta (T=200) captura los 3 casos donde una config
   predice literalmente 0 o casi-0 voxels (IOG31, IOG36, IOG38). La
   regla relativa (ρ=0.10) añade IOG48 (ratio 4.5 %) y IOG28 (ratio
   5.6 %) donde la config "perdedora" tiene una predicción pequeña
   en la zona equivocada que contamina el soft-avg.
3. **Fallos residuales tras gated** (ninguno más rescatable por
   ensemble):

  | Caso  | 2D    | 3D    | Ruta                                         |
  | ----- | ----- | ----- | -------------------------------------------- |
  | IOG40 | 0.000 | 0.000 | Cascada detect → seg                         |
  | IOG45 | 0.000 | 0.000 | Cascada detect → seg                         |
  | IOG1  | 0.050 | 0.178 | Inspección manual GT                         |
   | IOG29 | 0.863 | 0.380 | Stays en 2D vía soft-avg; augment específico |

4. **Ensemble oficial por CLI (bug conocido).**
   `nnUNetv2_find_best_configuration` hace producto cartesiano de
   `-p × -c`, así que busca combinaciones imposibles como
   `nnUNetTSMRIPlans__2d`. Workaround: llamarlo por la API de
   Python con la lista explícita de pares reales (ver
   `reports/find_best_alt.log` si existe):
   Nota: con `python - <<EOF` (heredoc sobre stdin) truena por el
   spawn de workers de multiprocessing que intentan re-ejecutar
   `<stdin>`. Hay que guardarlo como `.py` real.
5. **Reporte ALT vs DA5 consolidado** (útil para el writeup/Felipe):
   ```bash
   # Nota: los trainers viven en envs distintos. Hay que ejecutar dos
   # veces cambiando nnUNet_results y luego hacer el diff, o copiar
   # ambos trainers al mismo dataset_dir antes de correrlo.
   export nnUNet_results=$(pwd)/nnunet_env/nnUNet_results
   python scripts/summarize_5fold.py --dataset 501 \
       --trainer nnUNetTrainerALT_os033_250epochs --all-configs \
       --output reports/alt_5fold.txt

   export nnUNet_results=$(pwd)/nnunet_env_base/nnUNet_results
   python scripts/summarize_5fold.py --dataset 501 \
       --trainer nnUNetTrainerDA5_100epochs --all-configs \
       --output reports/da5_5fold.txt
   ```
6. **Inspeccionar IOG54, IOG12 manualmente** (los 2 únicos regresores
   fuertes). En particular IOG54 2D: pasó de 0.674 a 0.228 sin
   razón obvia y solo en 2D, mientras 3D lo mejora +0.38. Abrir los
   NIfTI de predicción de ambos trainers (base y current) en ITK-SNAP
   y ver si se desplazó/pegoteó. Candidato más probable: el split
   estratificado lo movió a un fold con menos vecinos de su volumen
   tumoral.

### A mediano plazo (1–2 semanas, orden de ROI esperado)

1. **Pretrain 2D desde TotalSegMRI (`Conv3d → Conv2d`)**. Sigue siendo
   el cambio con mayor upside para 2D (ahora la config ganadora
   individual). Retornos esperados: +0.02 – +0.04 Dice medio; más
   estabilidad en folds 0 y 2 que subieron menos. Prep: ya hay
   `scripts/prepare_pretrain_plans.py`; falta el conversor de pesos.
2. **Cascada detección → segmentación para IOG45 / IOG40** (tumores
   < 5 k voxels). Detector 3D ligero (nnDet o UNet + heatmap) a
   spacing nativo, luego segmentación en ROI 64³. Cierra los dos
   únicos fails duros que el ensemble no rescata.
3. **Augment de inversión de intensidad** (simulación T1 hipointenso)
   solo en 10 – 20 % de los batches. Dirigido a IOG38 y potencialmente
   IOG1 si es un subtipo similar. Cambio pequeño en
   `custom_trainers/nnUNetTrainerALT.py` (hookear `data_aug_params` o
   añadir un `SpatialTransform` con `p_invert`).
4. **Re-plan con spacing objetivo más fino para 3D** solo en los
   casos con spacing in-plane < 0.5 mm (IOG28, IOG31, IOG4). No es
   trivial en nnU-Net (requiere plans custom por paciente o un
   `ExperimentPlanner` con percentil 25 en vez de mediana). Baja
   prioridad porque 2D ya los rescata.
5. **Preprocesador con clipping por percentiles 0.5/99.5** antes del
   Z-Score. Probable retorno < +0.01 ahora que `_fix_intensity` ya
   arregló IOG47; útil solo si se añaden nuevos pacientes con rangos
   raros.

### Barato y rápido (opcional, 1 fold cada uno)

- **Sweep de `oversample ∈ {0.25, 0.33, 0.40}`** en fold 1 (el único
  fold 3D que retrocedió). ~45 min por punto; espero que el óptimo
  siga siendo 0.33 pero vale la pena confirmar.
- `**batch_size` +1 en 2D** si cabe en VRAM: con oversample 0.33 y
  splits balanceados, batches más grandes suelen ayudar a lesiones
  pequeñas.

### Objetivo revisado (con datos)

- **Dice ensemble 5-fold (medido)**: soft-avg = 0.7247; gated
v2 = 0.7730; gated v2b (hard) = 0.7805; gated **v2c (actual,
sigmoid+conf)** = **0.7822**. Superamos el objetivo original de
0.72 – 0.76.
- **Techo sin re-entrenar (oracle)**: 0.7933. Brecha restante
gated v2c → oracle = **0.011** concentrada en: IOG10 (Δ≈0.18),
IOG19 (Δ≈0.10), IOG33 (Δ≈0.05). Son casos donde ambas configs
predicen volúmenes similares y confianzas similares; el sigmoid+conf
ya rescató IOG12 y IOG35.
- **Fallos residuales absolutos (Dice = 0 tras gated)**: IOG40,
  IOG45. Ambos modelos fallan en predecir la lesión (tumores 156 y
  182 voxels respectivamente) → requieren cascada detect→seg.
- **Casos de Dice bajo a atacar en Exp. 2b/2c**: IOG1 (0.14),
IOG38 (0.23), IOG10 (0.41).
- **Siguiente salto** (Semana 2, actualizado post Exp. 2a): el
pretrain 2D desde TotalSegMRI **queda descartado** (ver §8.2). El
gate sigmoide con confianza **cerrado con +0.0017** (0.7822, ver
§8.2). Plan activo: **Exp. 2b (augment de inversión para IOG38)**
→ si pasa sanity, combinar con gated v2c → Exp. 2c (case-aware) o
**Exp. 3 (cascada detect→seg para IOG40/IOG45)**. Brecha restante
= 0.011 pts hasta el oracle; fails duros
IOG40/IOG45 fuera del techo del oracle.

## 8. Plan Semana 2 — estado y experimentos

### 8.1. Estado del repo al iniciar Semana 2

**Código existente relevante:**


| Archivo                               | Propósito                                                                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `run_training.sh`                     | Entrenamiento 5-fold end-to-end. Soporta `TRAINER=DA5|ALT|ALT_OS066`, `T1_ONLY=1`, `EPOCHS`, etc. Idempotente (resume desde checkpoints). |
| `custom_trainers/nnUNetTrainerALT.py` | Familia `ALT`: `_250/_500epochs` (os=0.66) y `_os033_250/_500epochs` (os=0.33, ganador actual). Hereda de `nnUNetTrainerDA5`.             |
| `scripts/convert_to_nnunet.py`        | Conversión raw→nnU-Net. Incluye `_fix_intensity` para el wrap-around uint16→int16 de IOG47.                                               |
| `scripts/fetch_totalseg_mri.py`       | Descarga pesos TotalSegmentator MRI (task 852) para pretrain 3D.                                                                          |
| `scripts/prepare_pretrain_plans.py`   | Prepara plans `nnUNetTSMRIPlans` compatibles con los pesos TotalSegMRI.                                                                   |
| `scripts/make_stratified_splits.py`   | Splits estratificados por volumen tumoral.                                                                                                |
| `scripts/patch_plans.py`              | Persiste `batch_dice=False` en `nnUNetPlans.json` para el 2D.                                                                             |
| `scripts/summarize_5fold.py`          | Reporte 5-fold por trainer/config con comparación vs baseline.                                                                            |
| `scripts/ensemble_gated.py`           | Ensemble gated 2D+3D (regla absoluta + relativa). Producto final de Semana 1.                                                             |


**Resultados en disco:**

- `nnunet_env_base/` → DA5_100ep (congelado, baseline histórico).
- `nnunet_env/` → ALT_os033_250ep (actual) + gated en
  `nnUNet_results/Dataset501_ALT_T1/gated_ensemble_gated_v2_rel10/`.

**Reportes generados (en `reports/`):**

- `find_best_alt.log` — salida de `find_best_configuration`.
- `ensemble_gated_v2.log` — gated abs+rel (0.7730).
- `ensemble_gated_abs.log` — gated solo absoluto (0.7561).
- `alt_5fold.txt` — per-case 5-fold report.

### 8.2. Experimentos priorizados (orden de ROI)

#### Exp. 1 — Pretrain 2D desde TotalSegMRI (máximo upside)

**Hipótesis**: subir el 2D solo de 0.7235 → 0.75–0.76 (similar al
jump que el pretrain 3D le dio al 3D baseline). Esto desplaza el
oracle hacia arriba y por transitividad el gated también sube.

**Bloqueador de código**: falta `scripts/convert_tsmri_weights_3d_to_2d.py`.
Qué debe hacer:

1. Cargar el state_dict del modelo 3D ya entrenado
   (`$nnUNet_preprocessed/Dataset850_TotalSegMRI/<trainer>__nnUNetPlans__3d_fullres/checkpoint_final.pth`
   o el path que indique `pretrain_manifest.json`).
2. Para cada `Conv3d(kxkxk)` → `Conv2d(kxk)` haciendo `mean` en el
   eje z del kernel (inflation inverso, estrategia estándar).
3. Para cada `InstanceNorm3d` / `BatchNorm3d` → copia paramétrica a
   `InstanceNorm2d` / `BatchNorm2d`.
4. Re-mapear nombres al plan 2D objetivo (validar número de canales
   y estructura contra `nnUNetPlans.json` del Dataset501).
5. Guardar en formato compatible con
   `nnUNetv2_train ... -pretrained_weights <ckpt>`.

**Comando esperado una vez exista el script**:

```bash
python scripts/convert_tsmri_weights_3d_to_2d.py \
    --manifest $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json \
    --target-plans $nnUNet_preprocessed/Dataset501_ALT_T1/nnUNetPlans.json \
    --target-config 2d \
    --output $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d.pth

# Re-entrenar 2D con los pesos nuevos (los 3D quedan igual):
for F in 0 1 2 3 4; do
  nnUNetv2_train 501 2d $F \
      -tr nnUNetTrainerALT_os033_250epochs \
      -p  nnUNetPlans \
      -pretrained_weights $nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d.pth \
      --npz 2>&1 | tee -a reports/alt_2d_pretrain_fold${F}.log
done

# Re-evaluar ensemble (reutilizar los 3d ya entrenados)
python /tmp/find_best_alt.py
python scripts/ensemble_gated.py --out-name gated_v3_pretrain2d \
    2>&1 | tee reports/ensemble_gated_pretrain2d.log
```

**Costo**: ~2h por fold × 5 folds ≈ 10h GPU. Idempotente; paralelizable
con `NUM_GPUS=2`. **Objetivo**: Dice gated final **0.78 – 0.81**.

##### Resultado 5-fold (2D pretrain "full")


| Fold             | 2D nopretrain (baseline) | 2D pretrain (Exp. 1 full) | Δ          |
| ---------------- | ------------------------ | ------------------------- | ---------- |
| 0                | 0.659                    | 0.634                     | −0.025     |
| 1                | 0.801                    | 0.812                     | +0.011     |
| 2                | 0.627                    | **0.594**                 | **−0.033** |
| 3                | 0.761                    | 0.773                     | +0.012     |
| 4                | 0.778                    | 0.774                     | −0.005     |
| **media 5-fold** | **0.7235**               | **0.7155**                | **−0.008** |


Veredicto: **el 2D solo con pretrain queda neutral-ligeramente negativo
vs nopretrain**. 3 folds regresan (0, 2, 4), 2 mejoran (1, 3). La media
global cae 0.008 pts; estadísticamente "empate", pero la dispersión
per-caso es mucho más alta que el promedio y es donde está la señal.

Patrón por dificultad de fold:

- Folds "fáciles" (1, 3, baseline ≥ 0.76): el pretrain ayuda levemente
(+0.01 – +0.01). Los casos mayoritarios son anatómicamente cercanos
a lo que TS-MRI vio.
- Folds "duros" (0, 2, baseline < 0.66): el pretrain arrastra sesgos y
regresa (−0.025 a −0.033). Los casos difíciles (IOG1, IOG28, IOG38
en fold 2) quedan cerca de 0 igual que en baseline.
- Fold 4 (baseline 0.778, alto): empate técnico, Exp. 1 no aporta.

Per-case swings (pretrain − nopretrain, medidos 5-fold):

- **Gana mucho** (rescates): IOG9 (+0.30), IOG7 (+0.16), **IOG36
(+0.16** — antes fail 2D total, ahora 0.158 en fold 0), IOG54 (+0.09
en fold 3, sigue problemático pero ya no perdido), IOG28 (+0.07 en
fold 2, empieza a moverse pero sigue < 0.5).
- **Pierde mucho** (regresiones): IOG10 (**−0.32** en fold 0, la peor),
IOG48 (−0.17 en fold 1), IOG29 (−0.12 en fold 1), IOG43 (−0.09 en
fold 4), IOG31 (−0.05 en fold 2), IOG50 (−0.04 en fold 3).
- **Shared fails intactos**: IOG1 (fold 2), IOG40 (fold 4), IOG45 (fold
  1. siguen en 0.000. Pretrain no rescata ninguno → sigue siendo
  trabajo de Exp. 3.

Que IOG36 salga del 0.00 del 2D es la ganancia más "útil" para el gated
porque el 3D ya lo tenía en 0.77 — el gated lo va a cerrar en ≥ 0.80
sin pagar el costo de los casos que perdieron (IOG10/48/29 el 3D los
tenía mal o similar, así que el gate los va a mantener por el 2D de
todas formas). **Hipótesis fuerte**: Exp. 1 full aporta ~+0.005–+0.015
al gated pese a quedar neutro en 2D solo. A validar con `gated_v3`.

Con esto, la decisión post-fold-4 es:

- **Exp. 1 full queda en producción como backbone del 2D** mientras
aporte al gated (ver §8.2 — pending).
- **Exp. 1a (K=1 shallow y A=0.5 mix) pasa de "por si acaso" a "debería
mejorar esto"**, porque un pretrain que no envenene IOG10/48/29 sí
podría dar +0.01–+0.02 en 2D solo y más en gated.

##### Resultado gated v3 (pretrain 2D full + 3D TS-MRI)


| Ensemble                                                                        | Dice 5-fold |
| ------------------------------------------------------------------------------- | ----------- |
| Oracle `max(2D, 3D)`                                                            | 0.7933      |
| **Gated v2b** (2D nopretrain + 3D TS-MRI, **T=50, ρ=0.40**) **← nuevo ganador** | **0.7805**  |
| Gated v2 (2D nopretrain, T=50, ρ=0.10)                                          | 0.7730      |
| Gated v3 (2D pretrain, mejor sweep T=200–500 ρ=0.25–0.40)                       | 0.7749      |
| Gated v3 (2D pretrain, T=50 ρ=0.10)                                             | 0.7510      |
| Soft-avg (con o sin pretrain)                                                   | 0.7247      |
| 2D solo pretrain                                                                | 0.7155      |
| 2D solo nopretrain                                                              | 0.7235      |
| 3D solo TS-MRI pretrain                                                         | 0.7076      |


**Veredicto: Exp. 1 full queda archivado**. Dos resultados
independientes:

1. El pretrain 2D full aplicado tal cual con los umbrales del gated v2
  regresa el ensemble −0.022 (0.7730 → 0.7510) por el mecanismo de
   "partial-rescue rompe el gate" (ver abajo).
2. Incluso con el mejor gate para cada variante, el pretrain pierde
  **−0.006** (0.7749 vs 0.7805). El costo en predicciones 2D débiles
   supera a las ganancias en IOG9/IOG7/IOG36.

**Mecanismo del daño** — el pretrain genera predicciones 2D "tibias"
(cerca de 0.1 de Dice pero con ≥ 1000 voxels predichos) en casos que
el 2D nopretrain dejaba completamente vacíos. Eso desactiva el gate
(`min-fg-voxels=50`, `min-fg-ratio=0.10`) y fuerza soft-avg con un 2D
débil que arrastra al 3D.


| Caso  | 2D nopre Dice (vox) | 2D pre Dice (vox) | Gated v2        | Gated v3             | Δ         |
| ----- | ------------------- | ----------------- | --------------- | -------------------- | --------- |
| IOG36 | 0.000 (0)           | 0.158 (1318)      | ~0.77 (gate→3D) | **0.224** (soft-avg) | **−0.55** |
| IOG10 | 0.404 (?)           | 0.084 (5645)      | ~0.57           | 0.409                | −0.16     |
| IOG28 | 0.043 (?)           | 0.110 (7396)      | ~0.81           | 0.663                | −0.15     |


Rescates reales que aportó el gate v3 (suma +1.21 crudo / +0.026 sobre
46 casos): IOG31 (+0.614), IOG48 (+0.464), IOG38 (+0.129). Pero la
suma de regresiones silenciosas (−0.86 / −0.019) los anula y deja Δ
negativo.

##### Sweep de gate — hallazgo colateral

Barrido T × ρ sobre ambas variantes (2D nopretrain y 2D pretrain full),
3D TS-MRI común, 5-fold.

**Nopretrain 2D (mean 2D solo = 0.7235):**


| T \ ρ | 0.10   | 0.25       | 0.40       |
| ----- | ------ | ---------- | ---------- |
| 50    | 0.7730 | 0.7742     | **0.7805** |
| 200   | 0.7730 | 0.7742     | **0.7805** |
| 500   | 0.7793 | **0.7805** | **0.7805** |
| 1500  | 0.7727 | 0.7738     | 0.7738     |
| 3000  | 0.7680 | 0.7692     | 0.7692     |


**Pretrain 2D full (mean 2D solo = 0.7155):**


| T \ ρ | 0.10   | 0.25       | 0.40       |
| ----- | ------ | ---------- | ---------- |
| 50    | 0.7510 | —          | —          |
| 200   | 0.7510 | 0.7712     | 0.7749     |
| 500   | 0.7547 | **0.7749** | **0.7749** |
| 1500  | 0.7602 | 0.7685     | 0.7685     |
| 3000  | 0.7574 | 0.7657     | 0.7657     |


Dos hallazgos:

1. **El gated v2 original estaba sub-tuneado.** Subir `min-fg-ratio` de
  0.10 → 0.40 (sin tocar `min-fg-voxels=50`) sube el gated con
   nopretrain de **0.7730 → 0.7805** (+0.008). Cero costo de entrena-
   miento. La intuición: filtrar predicciones 2D floridas pero con
   pocos voxels relativos al 3D (p. ej. 2D predice 1 000 vox cuando
   el 3D predice 10 000) elimina el ruido sin matar rescates reales.
2. **El pretrain full no remonta con ningún (T, ρ)**. Su techo es
  0.7749 contra los 0.7805 de nopretrain. Diferencia consistente de
   −0.005 a −0.006 en todo el rango de umbrales útiles.

**Mecanismo del daño del pretrain (antes llamado "partial-rescue
rompe el gate")** — el pretrain genera predicciones 2D "tibias" (cerca
de 0.1 de Dice pero con ≥ 1000 voxels predichos) en casos que el 2D
nopretrain dejaba completamente vacíos o casi. Eso desactiva el gate
por `min-fg-voxels` y fuerza soft-avg con un 2D débil que arrastra al
3D.


| Caso  | 2D nopre Dice (vox) | 2D pre Dice (vox) | Gated v2 (ρ=0.10) | Gated v3 (ρ=0.10)    | Δ         |
| ----- | ------------------- | ----------------- | ----------------- | -------------------- | --------- |
| IOG36 | 0.000 (0)           | 0.158 (1318)      | ~0.77 (gate→3D)   | **0.224** (soft-avg) | **−0.55** |
| IOG10 | 0.404 (?)           | 0.084 (5645)      | ~0.57             | 0.409                | −0.16     |
| IOG28 | 0.043 (?)           | 0.110 (7396)      | ~0.81             | 0.663                | −0.15     |


Rescates reales que aportó gate v3 (IOG31 +0.614, IOG48 +0.464,
IOG38 +0.129, suma +1.21 / +0.026 sobre 46 casos) no alcanzan a
compensar las regresiones silenciosas (−0.86 / −0.019). Subir `ρ` a
0.40 parcialmente lo arregla (redirige IOG36 al 3D otra vez) pero
**no revierte el déficit** frente al nopretrain con el mismo ρ.

##### Decisión y plan

1. **Adoptar gated v2b como default de producción**:
  `--min-fg-voxels 50 --min-fg-ratio 0.40` con nopretrain 2D +
   TS-MRI pretrain 3D. Dice 5-fold = **0.7805** (+0.019 vs gated v2,
   +0.057 vs 2D baseline, brecha vs oracle = 0.013).
2. **Exp. 1 full: archivar**. El pretrain genera predicciones 2D
  intermedias que arruinan la complementariedad 2D↔3D. No se vuelve
   a probar como default.
3. **Exp. 1a pasa a ser el experimento principal**. El target ya no
  es 0.7730, es **0.7805**. La hipótesis de shallow/mix es que al
   no emitir predicciones tibias en casos duros (por no transferir
   las capas mid-level donde se forman), el 2D mantendría la calidad
   de complementariedad del nopretrain y sumaría algo de rescate real
   en IOG9/IOG7 sin el daño colateral de IOG36/10/28. Criterio duro:
   **gated_v4 ≥ 0.7805** con los mismos umbrales gate v2b.

#### Exp. 1a — Partial-depth & soft warm-start (si Exp. 1 queda neutral)

**Hipótesis**: el pretrain TS-MRI aporta features bajos (bordes MR,
texturas, gradientes) pero las capas mid-level arrastran priors
"anatómicos" (forma de órgano) que chocan con la morfología lipomatosa.
Dos mitigaciones complementarias para conservar lo bueno y dejar que el
modelo reaprenda lo malo:

1. **Shallow partial transfer** (`--max-transfer-encoder-stage K`):
  sólo transfiere tensores cuyo nivel piramidal esté en `0..K`. El
   mapeo es:
  - `encoder.stages.<i>.`*         → nivel `i`
  - `decoder.encoder.stages.<i>.*` → nivel `i` (referencia duplicada)
  - `decoder.stages.<j>.*`         → nivel `n_stages - 2 - j`
  - `decoder.transpconvs.<j>.*`    → nivel `n_stages - 2 - j`
   Con K=1 solo transfiere las 2 primeras capas bajas (32 y 64
   features) y la sección del decoder que escribe a esos mismos
   niveles. Con K=2 añade la capa de 128 features. K=3 equivale
   prácticamente a "full" porque más allá ya no hay coincidencia de
   widths con el source (features 320 vs 512).
2. **Soft warm-start** (`--mix-ratio A`): para cada tensor
  efectivamente transferido, `w = A · w_pretrain + (1−A) · w_init`. Se
   aplica después del gate de stages. Ejemplos útiles: `A=0.5`
   (balanceado), `A=0.75` (transferencia dominante pero regularizada).

Sanity-check rápido (1 fold cada uno, ~2h/variante; usar **fold 1**,
el más estable históricamente):

```bash
# Variante 1a — shallow K=1
python scripts/convert_tsmri_weights_3d_to_2d.py \
    --manifest "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json" \
    --target-plans "$nnUNet_preprocessed/Dataset501_ALT_T1/nnUNetPlans.json" \
    --target-dataset-json "$nnUNet_preprocessed/Dataset501_ALT_T1/dataset.json" \
    --target-config 2d \
    --max-transfer-encoder-stage 1 \
    --output "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d_k1.pth"

# Variante 1b — soft warm-start A=0.5
python scripts/convert_tsmri_weights_3d_to_2d.py \
    --manifest "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_manifest.json" \
    --target-plans "$nnUNet_preprocessed/Dataset501_ALT_T1/nnUNetPlans.json" \
    --target-dataset-json "$nnUNet_preprocessed/Dataset501_ALT_T1/dataset.json" \
    --target-config 2d \
    --mix-ratio 0.5 \
    --output "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d_mix50.pth"

# Sanity en fold 2 (recuperar baseline nopretrain = 0.627;
# Exp. 1 full cayó a 0.594)
nnUNetv2_train 501 2d 2 \
    -tr nnUNetTrainerALT_os033_250epochs -p nnUNetPlans \
    -pretrained_weights "$nnUNet_preprocessed/Dataset850_TotalSegMRI/pretrain_2d_k1.pth" \
    --npz 2>&1 | tee reports/alt_2d_pretrain_k1_fold2.log
```

Criterio de decisión (con los datos parciales de Exp. 1 full actualizados):

- Ahora mismo Exp. 1 full está en **−0.009 media 0–3** y pierde claro
en folds 0 y 2. Una variante útil debe (i) recuperar fold 2 a ≥ 0.63
(baseline nopretrain) o (ii) mantener fold 1/3 y subir la media
global.
- Probar primero **fold 2** (no fold 1) porque es donde el full
regresa. Si `K=1` o `A=0.5` lo recuperan a ≥ 0.63 sin perder fold 1,
promover esa variante al 5-fold.
- Si las 3 variantes (full, K=1, A=0.5) empatan en fold 2 por debajo
del baseline, el 2D solo no se beneficia del TS-MRI y Exp. 1 aporta
sólo vía gated/oracle en casos como IOG36. En ese escenario: saltar
a Exp. 2/3 y reutilizar únicamente el pretrain en el gated.

**Costo**: 2h × 2 variantes en fold 1 = 4h GPU adicionales antes del
5-fold final.

##### Resultado sanity fold 2 (Exp. 1a)

Ejecutado en paralelo en dos consolas (GPU 0 + GPU 1) con
`nnUNet_results` distinto por variante. Valores Dice (fold 2, n=9
casos de validación):


| Variante                  | Fold 2 Dice 2D | Δ vs nopre (0.6266) | Δ vs Exp. 1 full (0.5937) |
| ------------------------- | -------------- | ------------------- | ------------------------- |
| **nopre (baseline)**      | **0.6266**     | —                   | +0.033                    |
| mix50 (A=0.5, all stages) | 0.6102         | −0.016              | +0.017                    |
| full (A=1, all stages)    | 0.5937         | −0.033              | —                         |
| **K=1 (shallow, A=1)**    | **0.5414**     | **−0.085**          | −0.052                    |


Per-case (fold 2), variantes de Exp. 1a vs nopre:


| Caso  | nopre | K=1       | mix50     | Ganador       |
| ----- | ----- | --------- | --------- | ------------- |
| IOG1  | 0.050 | 0.008     | 0.008     | nopre         |
| IOG28 | 0.043 | 0.161     | 0.147     | K=1 / mix50   |
| IOG31 | 0.746 | **0.223** | 0.814     | mix50         |
| IOG33 | 0.940 | 0.955     | 0.959     | mix50 ≈ K=1   |
| IOG35 | 0.845 | 0.776     | **0.731** | nopre         |
| IOG38 | 0.232 | 0.072     | 0.053     | nopre         |
| IOG39 | 0.947 | 0.950     | 0.955     | ≈             |
| IOG46 | 0.905 | **0.780** | 0.902     | nopre ≈ mix50 |
| IOG49 | 0.932 | 0.948     | 0.923     | K=1           |


Veredicto por variante:

- **K=1 (shallow) es catastrófico** (−0.085). Peor que Exp. 1 full.
La hipótesis de "solo transferir low-level y aprender mid/high
desde cero" falla: transferir stages 0–1 con stages 2+ aleatorios
crea una inconsistencia interna (filtros bajos calibrados a
estadística multi-órgano + mid-layers sin inicializar para IO)
que el modelo no reconcilia en 250 épocas. IOG31 cae de 0.746 a
0.223, IOG46 de 0.905 a 0.780. El patrón de "partial rescue rompe"
casos ya funcionales es aún más fuerte que en el full.
- **mix50 (A=0.5) es neutral-malo** (−0.016). Más cerca del baseline
pero aún por debajo. Gana en IOG31/IOG33/IOG39 marginal, pierde
fuerte en IOG35/IOG38 y el mismo patrón de daño del full.
Esencialmente es un "pretrain full diluido": menos toxicidad pero
tampoco aporta.

**Ninguna variante de Exp. 1a supera a nopre en fold 2.** Proyección
al ensemble gated con T=50, ρ=0.40:

- K=1 extrapolado a 5-fold: ~0.638 (peor que Exp. 1 full = 0.7155).
Gated estimado ≤ 0.75.
- mix50 extrapolado a 5-fold: ~0.710 (similar a Exp. 1 full).
Gated estimado ~0.77.
- Ambos muy por debajo del baseline gated v2b = **0.7805**.

##### Decisión: archivar toda la rama de pretrain

Tres resultados coherentes:

1. **Exp. 1 full** (A=1, all stages): 2D 5-fold = 0.7155 (−0.008 vs
  nopre). Gated óptimo = 0.7749 (−0.006 vs v2b).
2. **Exp. 1a K=1** (A=1, stages 0–1): 2D fold 2 = 0.5414 (−0.085 vs
  nopre). Peor que full.
3. **Exp. 1a mix50** (A=0.5, all stages): 2D fold 2 = 0.6102 (−0.016
  vs nopre). Mejor que full pero aún negativo.

El problema **no es la dosis ni la profundidad del transfer**: el
problema es que el **3D→2D slicing de TotalSegMRI no genera features
útiles para IO T1w**. Cualquier variante envenena los casos donde
nopretrain lograba 0 o predicción vacía y ahora emite "predicciones
tibias" (pocos voxels, Dice bajo) que desactivan el gate o ensucian
el soft-avg. Las ganancias en IOG9/IOG7/IOG36 no compensan las
regresiones en IOG10/IOG28/IOG31/IOG35/IOG38/IOG46.

**Pretrain 2D desde TS-MRI queda archivado (Exp. 1, 1a-K, 1a-mix).**
El `__2d.pretrain` permanece en disco como evidencia pero no se
promueve. Producción sigue con **nopretrain 2D + gated v2b = 0.7805**.

#### Exp. 2 — Cerrar la brecha gated → oracle sin reentrenar o con bajo costo

Con Exp. 1 archivado, el techo alcanzable sin tocar los pesos es el
oracle `max(2D, 3D)` = **0.7933**. Gated v2b está en **0.7805**.
Brecha = **0.0128**. Los casos que la componen (orden de impacto):


| Caso  | 2D    | 3D    | Gated v2b | Oracle | Δ gated→oracle | Perfil                                         |
| ----- | ----- | ----- | --------- | ------ | -------------- | ---------------------------------------------- |
| IOG29 | 0.863 | 0.380 | 0.575     | 0.863  | **+0.288**     | soft-avg floja; 2D domina pero gate no dispara |
| IOG10 | 0.404 | 0.714 | 0.559     | 0.714  | **+0.155**     | soft-avg tibia; 3D domina pero no gana gate    |
| IOG12 | 0.893 | 0.653 | ~0.77     | 0.893  | ~+0.12         | 3D arrastra 2D en soft-avg                     |
| IOG9  | 0.603 | 0.564 | ~0.58     | 0.603  | ~+0.02         | marginal                                       |


Más abajo de la brecha están los fails absolutos que ninguna de las
configs individuales resuelve (IOG40, IOG45) — esos exceden a Exp. 2
y caen en Exp. 3.

La lectura principal: **~85% de la brecha está en 3–4 casos donde
ambas configs predicen volúmenes decentes pero una está más
desplazada; el soft-avg los mezcla mal y el gate actual (hard
switch por `min-fg-voxels`/`min-fg-ratio`) no los atrapa**. Cualquier
mejora del gate o de la augment pipeline que no introduzca "weak
predictions" nuevas (como hizo el pretrain) puede capturar parte de
esto.

Propongo **tres sub-experimentos ortogonales**, ordenados por ROI:

##### Exp. 2a — Gated sigmoide / confidence-weighted (sin reentrenar) — **CERRADO**

**Hipótesis probada**: gate continuo por volumen + tie-breaker por
confianza media de softmax en la región positiva.

**Implementación final** (`scripts/ensemble_gated.py` extendido con
`--gate-mode {hard, sigmoid}`, `--tau`, `--use-confidence`,
`--conf-power`):

```
w2 = sigmoid((n_pred_2d − v_min) / τ)
w3 = sigmoid((n_pred_3d − v_min) / τ)
if --use-confidence:
    w2 *= mean(p2_softmax[fg] | mask_2d) ** k
    w3 *= mean(p3_softmax[fg] | mask_3d) ** k
w2, w3 := w2/(w2+w3), w3/(w2+w3)
p_final = w2 * p2_softmax[fg] + w3 * p3_softmax[fg]
mask    = (p_final > 0.5)
```

**Barrido (46 casos, softmax `.npz` leídos de disco)**:

Sigmoid **volume-only** (sin `--use-confidence`), `v_min=50`:


| τ    | 20     | 50     | 100    | 200    | 500    | 1000   |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ |
| Dice | 0.7530 | 0.7549 | 0.7656 | 0.7717 | 0.7749 | 0.7756 |


Siempre por debajo del hard v2b (0.7805) → los casos de volumen
mixto-moderado (IOG28, IOG48, IOG31, IOG36) se benefician del hard
switch, el sigmoid suave los atenúa.

Sigmoid **volume + confidence** (`v_min` × τ):


| v_min \ τ | 50         | 100        | 200        | 500    |
| --------- | ---------- | ---------- | ---------- | ------ |
| 50        | 0.7788     | 0.7789     | 0.7797     | 0.7795 |
| 100       | 0.7792     | 0.7792     | 0.7799     | 0.7794 |
| 200       | 0.7796     | 0.7799     | 0.7804     | 0.7795 |
| **500**   | **0.7815** | **0.7816** | **0.7815** | 0.7797 |
| **1000**  | **0.7821** | 0.7816     | 0.7806     | 0.7801 |
| 2000      | 0.7751     | 0.7801     | 0.7802     | 0.7788 |


Refinamiento en la zona `v_min ∈ [500, 1500]` × `τ ∈ [10, 80]`:


| v_min \ τ | 10         | 20         | 30         | 50     | 80     |
| --------- | ---------- | ---------- | ---------- | ------ | ------ |
| 500       | 0.7810     | 0.7811     | 0.7813     | 0.7815 | 0.7816 |
| 700       | **0.7822** | **0.7822** | 0.7821     | 0.7821 | 0.7821 |
| **1000**  | **0.7822** | **0.7822** | **0.7822** | 0.7821 | 0.7821 |
| 1500      | 0.7751     | 0.7751     | 0.7751     | 0.7802 | 0.7802 |


Plateau en **0.7822** con `v_min ∈ {700, 1000}, τ ∈ {10, 20, 30}, --use-confidence`. Aumentar `conf-power ∈ {2…20}` degrada
monótonamente (0.7821 → 0.7799), así que `k=1` es óptimo.

**Resultado**: ganador `gated v2c = 0.7822` vs hard v2b `0.7805`
→ **+0.0017 (13 % de la brecha al oracle 0.7933)**.

**Per-case diff vs hard v2b** (|Δ| ≥ 0.01):


| caso  | hard  | sigmoid   | Δ          | regla_hard → regla_sig              |
| ----- | ----- | --------- | ---------- | ----------------------------------- |
| IOG12 | 0.810 | **0.877** | +0.067     | soft-avg → sig w2=0.51 w3=0.49      |
| IOG35 | 0.739 | **0.802** | +0.062     | soft-avg → sig w2=0.50 w3=0.50      |
| IOG11 | 0.881 | 0.909     | +0.029     | soft-avg → sig                      |
| IOG17 | 0.900 | 0.921     | +0.021     | soft-avg → sig                      |
| IOG7  | 0.824 | 0.845     | +0.021     | soft-avg → sig                      |
| IOG1  | 0.124 | 0.142     | +0.019     | soft-avg → sig                      |
| IOG43 | 0.851 | 0.863     | +0.012     | soft-avg → sig                      |
| IOG28 | 0.810 | 0.791     | −0.019     | 3d (2d empty) → sig w2=0.44 w3=0.56 |
| IOG9  | 0.690 | 0.661     | −0.030     | soft-avg → sig                      |
| IOG33 | 0.890 | 0.848     | −0.041     | soft-avg → sig                      |
| IOG4  | 0.752 | 0.674     | **−0.078** | soft-avg → sig                      |


**Interpretación**: el sigmoid+conf gana 0.23 Dice crudo (7 casos) y
pierde 0.17 (4 casos) → neto +0.06 / 46 = +0.0014. La mejora viene de
pequeños skews `w2/(w2+w3)` cuando los volúmenes son comparables pero
una config es ligeramente más confiada — sobre todo en IOG12 y IOG35
donde el 2D (Dice 0.893 / 0.845) se impone al 3D (0.653 / 0.636).
Los losers IOG4, IOG33, IOG9 son casos donde el soft-avg por sí solo
era mejor que cualquier config individual (boosting por
complementariedad) y el skew por confianza lo rompe ligeramente.

**Costo real**: 4h de dev + minutos de cómputo.
**Veredicto**: ganancia modesta (+0.0017) pero reproducible y gratis
→ **gated v2c pasa a producción**. Brecha restante al oracle:
0.0111 → queda para Exp. 2b/2c/3.

##### Exp. 2b — Augment de inversión de intensidad (IOG38, IOG1)

**Hipótesis**: IOG38 (SNR −0.06, tumor más oscuro que bg) queda en
2D=0.232 / 3D=0.000 porque todo el entrenamiento asume "ALT ≈
hiperintenso". Una transformación que con `p=0.15` invierte la
intensidad de la imagen en un subconjunto de batches fuerza al
modelo a no depender del signo del contraste. IOG1 (0.124 gated)
podría beneficiarse si su fallo es también de contraste atípico
(pendiente inspección visual).

**Contexto — augmentations ya presentes en `nnUNetTrainerDA5`**
(base del actual `nnUNetTrainerALT_os033_250epochs`):


| #   | Transform                                                       | Prob            | Notas                                                      |
| --- | --------------------------------------------------------------- | --------------- | ---------------------------------------------------------- |
| 1   | `SpatialTransform` (rot 0.4 / scale 0.2 / `p_elastic_deform=0`) | —               | elastic desactivado a propósito                            |
| 2   | `Rot90` + `TransposeAxes`                                       | 0.5 c/u         | solo con ejes iguales                                      |
| 3   | `OneOf(MedianFilter, GaussianBlur)`                             | 0.2 c/u         |                                                            |
| 4   | `GaussianNoise` (var [0, 0.1])                                  | 0.1             |                                                            |
| 5   | `BrightnessAdditive` (μ=0, σ=0.5)                               | 0.1             |                                                            |
| 6   | `OneOf(Contrast preserve=T/F)`                                  | 0.2 c/u         |                                                            |
| 7   | `SimulateLowResolution` (scale 0.25–1)                          | 0.15            |                                                            |
| 8   | `Gamma(γ=[0.7,1.5], p_invert_image=1)` **×2**                   | 0.1 c/u         | **ya invierte ~20 % de batches** pero con gamma encadenada |
| 9   | `MirrorTransform`                                               | detereminístico |                                                            |
| 10  | `BlankRectangle`                                                | 0.4             | oclusiones                                                 |
| 11  | `BrightnessGradientAdditive`                                    | 0.3             |                                                            |
| 12  | `LocalGamma`                                                    | 0.3             |                                                            |
| 13  | `Sharpening`                                                    | 0.2             |                                                            |


DA5 ya invierte parcialmente (los dos `GammaTransform(p_invert=1)`),
pero siempre **encadenada con una gamma** (γ ∈ [0.7, 1.5]) y
reteniendo stats. Esa inversión "curva" no equivale a un flip lineal
puro: para imágenes con el tumor más oscuro que bg, la gamma post-flip
atenúa de nuevo el contraste. Por eso IOG38 no se está aprendiendo.

**Augmentations descartadas** para Exp. 2b (ortogonales):

- `ElasticDeformation`: DA5 lo desactiva (`p_elastic_deform=0`) por
inestabilidad en datasets pequeños; cambiar ese default tiene
riesgo de regresión que excede el scope del experimento.
- `MixUp` / `CutMix`: mezcla batches y rompe stats por caso;
incompatible con el DS loss de nnU-Net.
- `RandomClip` (existe en `batchgeneratorsv2/intensity/random_clip.py`):
robustez a outliers; ortogonal a IOG38, no es la hipótesis aquí.

**Diseño — dos variantes que idealmente corren en paralelo** (una por
GPU) para falsear dos hipótesis en la misma GPU-día de sanity check.
En la práctica la rama acabó ejecutándose en serie por disponibilidad
de hardware (1 GPU, ver §8.2 resultado Exp. 2b más abajo):


| Variante     | Trainer                                     | Qué hace                                                                                                                                                                         | Hipótesis                                                                                 |
| ------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Exp. 2b**  | `nnUNetTrainerALT_os033_inv_250epochs`      | Inserta `InvertImageTransform(p=0.15)` justo después del `SpatialTransform` del pipeline DA5. La inversión es *mean-preserving* (`x ← 2·mean − x`), no rompe z-score.            | El modelo necesita un **flip lineal puro**; la gamma-on-flipped de DA5 no es equivalente. |
| **Exp. 2b'** | `nnUNetTrainerALT_os033_invgamma_250epochs` | No añade transforms. Sube la `apply_probability` de los dos `RandomTransform(GammaTransform(p_invert=1))` de DA5 de 0.1 → **0.2** c/u (~20 % → ~40 % de batches con gamma-flip). | El pipeline DA5 ya era suficiente; **solo estaba subexpuesto**.                           |


Ambas se implementan en `custom_trainers/nnUNetTrainerALT_inv.py` con
mixins sobre `_ALTOs033Base` (MRO: `Mixin → _ALTOs033Base → _ALTBase → nnUNetTrainerDA5 → nnUNetTrainer`), y se exponen en `run_training.sh`
vía `TRAINER=ALT_OS033_INV` y `TRAINER=ALT_OS033_INVGAMMA`.

Interpretación de la carrera (cuatro esquinas):


| 2b pasa | 2b' pasa | Lectura                                                                                                  |
| ------- | -------- | -------------------------------------------------------------------------------------------------------- |
| ✓       | ✓        | La dirección (más supervisión invertida) es correcta. Promoción a 5-fold: la que tenga mayor IOG38.      |
| ✓       | ✗        | El flip lineal puro es la clave; promoción de 2b.                                                        |
| ✗       | ✓        | El pipeline DA5 ya funcionaba; bastaba subir la exposición. Promoción de 2b' (cambio mucho más trivial). |
| ✗       | ✗        | El problema de IOG38 no es de augment de intensidad → saltar a Exp. 2c o Exp. 3.                         |


**Sanity check escalonado** (fold 2 contiene IOG38 y IOG1, ~2h con
compile c/u):

```bash
# En la máquina con GPU. install_trainers.py ya copia ambos trainers
# al package en un solo call (auto-discover de nnUNetTrainer*.py).
python custom_trainers/install_trainers.py

# GPU 0 — Exp. 2b
T1_ONLY=1 TRAINER=ALT_OS033_INV FOLDS=2 CONFIGS=2d \
  bash run_training.sh 2>&1 | tee reports/alt_2d_inv_fold2.log

# GPU 1 — Exp. 2b' (en paralelo)
T1_ONLY=1 TRAINER=ALT_OS033_INVGAMMA FOLDS=2 CONFIGS=2d \
  bash run_training.sh 2>&1 | tee reports/alt_2d_invgamma_fold2.log
```

Si el node tiene 2 GPUs y querés que `run_training.sh` se encargue del
pinning, podés lanzar cada uno en su propia shell con `CUDA_VISIBLE_DEVICES`.

Criterio de promoción a 5-fold: **IOG38 ≥ 0.45 Y fold 2 mean ≥ 0.63**
(no regresa el baseline nopretrain, que anotó 0.6266 en fold 2). Si en
la tabla de cuatro esquinas cae en "ambos fallan": archivar la rama
completa y pasar a Exp. 2c (case-aware sampler) o Exp. 3 (cascada).

Si pasa sanity, 5-fold completo con la variante ganadora → evaluar con
gated v2c (sigmoid, v_min=1000, τ=20, use-conf) sobre los softmax
nuevos + los softmax 3D existentes (el 3D no se retocó: IOG38 está en
0 vox en 3D y reentrenar 3D no cambia eso sin spacing más fino).

**Costo**: 1h dev (hecho) + 2h sanity × 2 GPUs en paralelo + 10h 5-fold
si pasa (una sola variante).

###### Exp. 2b (InvertImage p=0.15) — sanity fold 2, **FALLA criterio de promoción**

Trainer `nnUNetTrainerALT_os033_inv_250epochs`, fold 2, 2d, 250 epochs.


| Caso      | Baseline `os033` | Exp. 2b `os033_inv` | Δ          | Observación                                   |
| --------- | ---------------- | ------------------- | ---------- | --------------------------------------------- |
| IOG1      | 0.050            | **0.000**           | **−0.050** | Contra-hipótesis: el flip empeora el caso     |
| IOG28     | 0.043            | 0.037               | −0.006     | neutral                                       |
| IOG31     | 0.746            | 0.540               | **−0.206** | regresión fuerte (caso mediano, n_ref=36 k)   |
| IOG33     | 0.940            | 0.937               | −0.003     | neutral                                       |
| IOG35     | 0.845            | 0.719               | **−0.126** | regresión (caso mediano)                      |
| **IOG38** | **0.232**        | **0.394**           | **+0.161** | **target del experimento — mejora real**      |
| IOG39     | 0.947            | 0.955               | +0.007     | neutral                                       |
| IOG46     | 0.905            | 0.934               | +0.030     | ligera mejora                                 |
| IOG49     | 0.932            | 0.791               | **−0.141** | regresión en el caso más grande (n_ref=148 k) |
| **mean**  | **0.6266**       | **0.5896**          | **−0.037** | **regresión neta**                            |


**Criterios de promoción**:

- IOG38 ≥ 0.45? **NO** (0.394, aunque +0.161 vs baseline — la
dirección es correcta).
- Fold 2 mean ≥ 0.63? **NO** (0.590 ⇒ regresión de 3.7 Dice points).

**Lecturas clave**:

1. **La hipótesis central se confirma direccionalmente**: IOG38, el
  caso con contraste atípico (SNR −0.06), sube +0.161 con más
   supervisión de imagen invertida. El insight no se descarta.
2. **IOG1 no era un target válido**: empeoró (−0.050). Su fallo no
  es de contraste invertido sino de otra causa (tamaño minúsculo,
   n_ref=12 k, ruido, …). **Se quita de la lista de beneficiarios
   esperados** del flip para los siguientes experimentos.
3. **p=0.15 es demasiado agresivo**: los totales muestran que el
  modelo se volvió más conservador (n_pred medio baja 12 %,
   39 776 → 34 918). En los casos grandes (IOG31, IOG35, IOG49) la
   pérdida de TPs (20–21 k vox) domina el Dice. El modelo está
   *olvidando* el régimen normal ALT-hiperintenso mientras aprende
   el invertido.

**Veredicto**: Exp. 2b con p=0.15 se **archiva**. Antes de abandonar
la rama completa, pasa el relevo a Exp. 2b' (gamma-boost: ataca el
mismo objetivo pero con un cambio estructuralmente más suave ⇒ menos
daño a casos normales).

###### Exp. 2b' (DA5 Gamma p_invert=1 boosted to apply_p=0.2) — sanity fold 2, **resultado mixto, promoción recomendada**

Trainer `nnUNetTrainerALT_os033_invgamma_250epochs`. Hipótesis
explícita tras 2b: subir la prob de los dos wrappers
`Gamma(p_invert=1)` *preexistentes en DA5* (de 0.1→0.2 c/u) es un
cambio **estructuralmente más suave** que insertar un flip lineal
(Exp. 2b), porque:

1. La inversión en `GammaTransform` viene siempre encadenada con una
  gamma γ∈[0.7, 1.5] y `p_retain_stats=1` — la imagen resultante
   preserva media y std. Preserva mejor la distribución de training.
2. La regresión de Exp. 2b concentrada en casos grandes sugiere que
  el modelo se volvió conservador al *no poder reconciliar* dos
   regímenes de contraste muy distintos; un flip con gamma+retain_stats
   interpola entre ambos regímenes en vez de saltar entre ellos.

**Resultado fold 2 sanity** (trainer, 2d, 250 epochs):


| Caso      | Baseline   | Exp. 2b (`inv`) | Exp. 2b' (`invgamma`) | Δ vs baseline |
| --------- | ---------- | --------------- | --------------------- | ------------- |
| IOG1      | 0.050      | 0.000           | 0.090                 | +0.040        |
| **IOG28** | **0.043**  | 0.037           | **0.823**             | **+0.780**    |
| IOG31     | 0.746      | 0.540           | 0.494                 | −0.252        |
| IOG33     | 0.940      | 0.937           | 0.956                 | +0.016        |
| **IOG35** | **0.845**  | 0.719           | **0.486**             | **−0.359**    |
| IOG38     | 0.232      | 0.394           | 0.388                 | +0.156        |
| IOG39     | 0.947      | 0.955           | 0.954                 | +0.006        |
| IOG46     | 0.905      | 0.934           | 0.935                 | +0.030        |
| IOG49     | 0.932      | 0.791           | 0.909                 | −0.023        |
| **mean**  | **0.6266** | 0.5896          | **0.6705**            | **+0.0439**   |


**Criterios de promoción**:

- IOG38 ≥ 0.45? **NO** (0.388, prácticamente idéntico a 2b).
- Fold 2 mean ≥ 0.63? **SÍ** (0.671, **+0.044** sobre baseline).

Pasa 1 de 2. El criterio "IOG38 ≥ 0.45" estaba mal calibrado: lo
fijamos asumiendo que IOG38 era EL target del flip. Empíricamente la
mejora dominante **no** está en IOG38 sino en **IOG28**, un caso que
no habíamos identificado como de contraste atípico y que salta de
0.04 → **0.82**. La métrica que importa para el producto final (mean
Dice 5-fold → gated) sí mejora +0.044, muy por encima del umbral.

**Hallazgos adicionales**:

1. ~~IOG28 es otro caso de contraste atípico~~ — **hipótesis
  refutada con medición directa**. Con la misma fórmula del §2
   (`scripts/case_contrast_stats.py`):

  | Caso  | SNR_shell | SNR_brain | tumor_mean | shell_mean |
  | ----- | --------- | --------- | ---------- | ---------- |
  | IOG38 | **−0.10** | **−0.46** | 170        | 185        |
  | IOG28 | **+1.57** | **+1.28** | 1 552      | 816        |
  | IOG35 | +1.35     | +2.18     | 1 413      | 801        |
  | IOG1  | +0.59     | +0.24     | 834        | 621        |

   IOG28 es **hiperintenso normal**, no invertido. El rescate de 2b'
   no viene de "enseñarle contraste invertido". Ver §2 para una
   lectura nueva del mecanismo de `GammaTransform(p_invert=1,  p_retain_stats=1)`: no es un flip de signo, es un **remapeo
   fuerte del histograma preservando media y std**. El efecto neto
   es forzar al modelo a usar *forma* y no valores absolutos de
   intensidad. IOG38 se rescata *como caso particular* de esa
   agnostia. IOG28 se rescata porque su spacing raro (0.31 mm vs
   target 0.78 mm) produce texturas resampleadas inusuales y el
   modelo baseline se aferraba a ellas.
2. **IOG35 regresa peor que en 2b** (0.85 → 0.49, −0.36). El
  failure mode del baseline en IOG35
   (`scripts/analyze_failure_mode.py`) muestra:
  - Tumor en 8 slices (z=3–10), bien localizado en el centro
  (Dice 0.87–0.95 en slices con más GT).
  - Fallos existentes: z=10 (201 vox completamente missed) y z=3
  (Dice 0.63, sólo 631 de 1 288 GT). Ambos tienen FN con
  mean intensity 1 331 vs TP 1 432 — son voxels de **borde
  con contraste bajo** contra el brain (p95=1511).
  - El contraste local tumor-vs-shell es alto (1.35) pero
  tumor-vs-brain es 2.18 porque el brain circundante ya es
  brillante. Los bordes son la única información discriminante.
   Perturbar esa información de intensidad con 40 % de frecuencia
   (2b') hace que el modelo dude de los bordes y se vuelva
   ultra-conservador: TP cae 7 443 → 3 138 (−58 %), FP también cae
   (962 → 548) → el modelo simplemente encuentra menos tumor.
3. **IOG1 sí mejora ligeramente con 2b'** (0.05 → 0.09): dirección
  correcta pero magnitud irrelevante. IOG1 tiene SNR_brain=0.24
   (contraste bajo, no invertido); el fallo es otro.
4. **Casos grandes (IOG31, IOG49) ya no caen tanto como en 2b**:
  IOG49 baja solo −0.02 vs −0.14 con 2b. Confirma que preservar
   mean y std (vía `retain_stats`) protege los casos normales.

**Reinterpretación del mecanismo de 2b'**: la explicación
"contraste invertido" era demasiado literal. El efecto real de
boost-gamma-inv es **regularización de dependencia de intensidad**:

- Ayuda a casos con texturas inusuales (IOG28 con spacing raro,
IOG38 con inversión real) al forzar dependencia en forma.
- Daña casos con tumor pequeño + contraste de borde crítico
(IOG35) porque la señal discriminante *es* la intensidad.

Implicación para Exp. 2c: un "case-aware" sampler debería usar
boost-gamma-inv **a alta frecuencia en cases con spacing anómalo o
SNR bajo**, y **a baja frecuencia en cases con tumor pequeño y
contraste de borde crítico**. Queda pendiente implementar esa
lógica.

**Veredicto**: promoción a 5-fold **recomendada**, con salvaguardas.

**Plan de promoción**:

1. **5-fold completo** en 2d (~10 h GPU, una sola GPU en serie):
  ```bash
   T1_ONLY=1 TRAINER=ALT_OS033_INVGAMMA FOLDS="0 1 2 3 4" CONFIGS=2d \
     bash run_training.sh 2>&1 | tee reports/alt_2d_invgamma_5fold.log
  ```
2. **Gate intermedio tras fold 0 y fold 1** (~4 h desde el punto 1):
  si ambos regresan su mean vs baseline de ese fold, **abortar** y
   pasar a Exp. 2c. Fold 2 ya cuenta como 1 validación positiva, así
   que con fold 0 **o** fold 1 positivos tenemos mayoría.
3. **Final**: aplicar gated v2c (sigmoid, v_min=1000, τ=20,
  use-confidence) sobre los softmax 2D nuevos + softmax 3D
   existentes. Target: 5-fold gated **0.79–0.80**.
4. **Tareas paralelas (CPU, sin GPU)**:
  - Re-medir SNR de IOG28; actualizar §4 si es atípico.
  - Inspección visual ITK-SNAP de IOG35 → insight para Exp. 2c.

**Plan elegido (conservador)**: fold 0 como 2° sanity **antes** del
5-fold. Fold 0 tiene un set de casos completamente distinto
(IOG10/16/17/20/24/36/4/45/51/53), así que valida la estabilidad del
gain sobre cases nuevos.

```bash
T1_ONLY=1 TRAINER=ALT_OS033_INVGAMMA FOLDS=0 CONFIGS=2d \
  bash run_training.sh 2>&1 | tee reports/alt_2d_invgamma_fold0.log
```

Baseline fold 0: mean Dice **0.6589**. Criterios de paso:

1. **fold 0 mean ≥ 0.66** (no-regresión).
2. **Ningún caso grande regresa > 0.15** (IOG16/17/20/24/51/53). Es
  la salvaguarda nueva post-fold 2: IOG35 (2b') regresó −0.36 y
   queremos asegurarnos que no es patrón general.
3. Bonus: rescate o mejora clara de IOG36, IOG45 (ambos 0.0 en
  baseline) o IOG10 (0.40) — sería confirmación análoga al rescate
   de IOG28 en fold 2 (aunque el mecanismo real es "shape-prior
   regularization", no "contrast inversion"; ver §8.2.2).

**Scripts CPU de análisis creados en paralelo**:

```bash
# SNR / contraste por case (extiende la tabla del §2)
python scripts/case_contrast_stats.py IOG28 IOG35 IOG38

# Failure mode comparison. En el remote (donde viven las pred de
# invgamma), corre:
python scripts/analyze_failure_mode.py IOG35 \
  --gt   /workspace/nnunet_env/nnUNet_raw/Dataset501_ALT_T1/labelsTr/IOG35.nii.gz \
  --image /workspace/nnunet_env/nnUNet_raw/Dataset501_ALT_T1/imagesTr/IOG35_0000.nii.gz \
  --pred baseline=/workspace/nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__2d/fold_2/validation/IOG35.nii.gz \
  --pred invgamma=/workspace/nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os033_invgamma_250epochs__nnUNetPlans__2d/fold_2/validation/IOG35.nii.gz
```

**Árbol tras fold 0**:

- Pasa 1+2 → comprometer folds 1, 3, 4 (~6 h). Total: 8 h de GPU.
- Regresa mean pero hay rescates → evaluación caso a caso, probable
archivar.
- Regresa mean + case grande > 0.15 → confirmar que 2b' no es
estable; archivar, pasar a Exp. 2c.

**Resultado fold 0** (`nnUNetTrainerALT_os033_invgamma_250epochs`):


| Caso     | Baseline   | Invgamma   | Δ           |
| -------- | ---------- | ---------- | ----------- |
| IOG10    | 0.404      | 0.266      | **−0.138**  |
| IOG16    | 0.936      | 0.922      | −0.014      |
| IOG17    | 0.939      | 0.952      | +0.013      |
| IOG20    | 0.864      | 0.780      | −0.084      |
| IOG24    | 0.864      | 0.827      | −0.036      |
| IOG36    | 0.000      | 0.201      | **+0.201**  |
| IOG4     | 0.710      | 0.557      | **−0.154**  |
| IOG45    | 0.000      | 0.000      | 0           |
| IOG51    | 0.926      | 0.921      | −0.004      |
| IOG53    | 0.946      | 0.951      | +0.005      |
| **mean** | **0.6589** | **0.6378** | **−0.0211** |


**Criterios**:

- fold 0 mean ≥ 0.66: **FALLA** (0.638).
- Ningún caso grande regresa > 0.15: **FALLA al borde** (IOG4
exactamente −0.154).
- Rescate parcial de IOG36 (+0.20), pero IOG10 regresa −0.14 casi
cancelándolo.

**Evidencia combinada fold 2 + fold 0**:


| Fold     | Baseline | Invgamma | Δ          |
| -------- | -------- | -------- | ---------- |
| 2        | 0.6266   | 0.6705   | **+0.044** |
| 0        | 0.6589   | 0.6378   | **−0.021** |
| media 2f | 0.6428   | 0.6542   | **+0.011** |


Varianza fold-a-fold de 0.065 → **inestable**. Media neta positiva
pero pequeña.

**Punto crítico — los rescates caen en el B-regime**:


| Caso rescatado | baseline 2D | baseline 3D | quién gana el gated v2c          |
| -------------- | ----------- | ----------- | -------------------------------- |
| IOG28 (fold 2) | 0.002       | **0.810**   | 3D (gated **ya cubre**)          |
| IOG36 (fold 0) | 0.000       | **0.772**   | 3D (gated **ya cubre**)          |
| IOG38 (fold 2) | 0.232       | 0.000       | 2D — **este sí aporta al gated** |


Las 2 rescates más grandes (IOG28 +0.78, IOG36 +0.20) ocurren donde
el 3D baseline ya dominaba y el gated ya las tomaba de esa rama. No
se traducen 1-a-1 al gated ensemble. En cambio las 5 regresiones
fuertes (IOG35 −0.36, IOG31 −0.25, IOG4 −0.15, IOG10 −0.14,
IOG20 −0.08) están en A-regime (2D dominante) o C-regime
(competitivo), que **sí se transmiten al gated**.

**Predicción**: gated con invgamma 2D probablemente queda en
[0.78, 0.7822] sobre 5-fold — igual o ligeramente peor que gated v2c
actual.

**Test gated decisivo (sin GPU, sobre los `.npz` de folds 0 y 2)**:

Se extendió `scripts/ensemble_gated.py` para aceptar trainers
distintos por rama (`--trainer-2d`, `--trainer-3d`). Corrido con
invgamma 2D + baseline 3D y los hparams de v2c (sigmoid, v_min=1000,
τ=20, use-conf, conf-power=1) sobre los 19 casos de folds 0 y 2.
Comparado contra gated v2c (baseline 2D + baseline 3D) en el mismo
subset con `scripts/compare_gated_summaries.py`:


| Caso            | gated v2c_base | gated v2c_inv2d | Δ           | Notas                     |
| --------------- | -------------- | --------------- | ----------- | ------------------------- |
| IOG1            | 0.142          | 0.147           | +0.004      |                           |
| IOG10           | 0.404          | 0.387           | −0.017      |                           |
| IOG16           | 0.926          | 0.916           | −0.010      |                           |
| IOG17           | 0.921          | 0.928           | +0.007      |                           |
| IOG20           | 0.859          | 0.835           | −0.023      |                           |
| IOG24           | 0.861          | 0.851           | −0.010      |                           |
| **IOG28**       | **0.791**      | **0.819**       | **+0.028**  | rescate B-regime limitado |
| **IOG31**       | 0.746          | **0.494**       | **−0.252**  | **REG grande**            |
| IOG33           | 0.848          | 0.858           | +0.010      |                           |
| **IOG35**       | 0.802          | **0.619**       | **−0.183**  | **REG grande**            |
| IOG36           | 0.772          | 0.729           | −0.043      | duplicación perjudicial   |
| **IOG38**       | **0.232**      | **0.388**       | **+0.155**  | **ÚNICO WIN grande**      |
| IOG39           | 0.942          | 0.944           | +0.002      |                           |
| IOG4            | 0.674          | 0.670           | −0.004      |                           |
| IOG45           | 0.000          | 0.000           | 0           |                           |
| IOG46           | 0.938          | 0.942           | +0.004      |                           |
| IOG49           | 0.914          | 0.904           | −0.010      |                           |
| IOG51           | 0.930          | 0.931           | +0.001      |                           |
| IOG53           | 0.936          | 0.940           | +0.003      |                           |
| **mean (n=19)** | **0.7178**     | **0.7001**      | **−0.0177** |                           |


**Lectura**: balance neto claramente negativo sobre el gated. La
hipótesis de §8.2.2 (los rescates IOG28/IOG36 caen en B-regime y no
aportan al gated) quedó confirmada cuantitativamente:

- IOG28 sube solo +0.028 (de 0.79 a 0.82) vs el +0.78 que vemos en
2D puro — el gated ya estaba cubierto por 3D.
- IOG36 directamente **empeora** (−0.043) porque el invgamma 2D
(0.20) diluye el gate cuando 3D ya da 0.77.
- Las regresiones A-regime (IOG35, IOG31) se transmiten al gated
completas: IOG35 −0.18, IOG31 −0.25.
- IOG38 es la **única rescate real** del gated (+0.155).

**Extrapolación a 5-fold gated**: −0.0177 × 19/46 ≈ −0.007 pesado
por ratio. Gated esperado ~**0.775** vs 0.7822 actual → clara
regresión.

**Decisión: archivar 2b'**. Junto con 2b (también archivado), la
**rama completa "augment de intensidad" queda cerrada**. Única
ganancia sostenida: IOG38 +0.155, que aislado vale +0.0034 al 5-fold
gated global — no justifica un 5-fold ni una variante 2c.

**Siguiente paso**: **Exp. 3 — cascada detect→seg** (ver §7). Los
fallos duros restantes son de detección/localización, no de
contraste. Ver §8.2.3 abajo.

###### Exp. 2c — case-aware sampler (descartado tras gated test)

Inicialmente pensado como "boost-gamma-inv solo en casos A-regime
con spacing anómalo + deshabilitar en B-regime con tumor pequeño".
Descartado por:

1. El único beneficiario neto en el gated es IOG38 (+0.155).
2. Un sampler case-aware requiere heurística confiable de clasificación
  por caso; nuestra medición del §8.2.2 mostró que las categorías
   "A-regime" y "B-regime" no son predecibles a priori (IOG28 tiene
   SNR positivo pero baseline 2D=0.002; IOG31 tiene SNR positivo y
   baseline 2D=0.75). La única señal clara es "SNR ≤ 0" ⇒ IOG38, y
   quizás algún otro que no tenemos en nuestra tabla medida.
3. El esfuerzo vs. ganancia esperada (+0.003) es pobre vs. Exp. 3.

Se deja la opción abierta como "Exp. 2c' — trainer dedicado a IOG38
solo" si Exp. 3 no basta: un SpatialTransform con `p_invert` alto
(e.g. 0.5) aplicado solo a ese case via oversample dedicado. Por
ahora queda pendiente.

**Árbol de decisión si 5-fold falla**:

- Si gated 5-fold **< 0.7822** (peor que v2c actual) → **archivar
2b'** y pasar a Exp. 2c (case-aware sampler) o Exp. 3 (cascada).
- Si gated 5-fold en **[0.7822, 0.79)** → mejora marginal; decidir
si vale la pena. Probablemente sí (es la mejor ganancia disponible
sin cambiar arquitectura).
- Si gated 5-fold **≥ 0.79** → Exp. 2b' a producción. Seguir
buscando mejoras complementarias (2c / 3).
**Objetivo**: IOG38 0.23 → 0.50+ (= +0.006 en la media global),
posible mejora marginal en IOG1.

##### Exp. 2c — Case-aware oversample (focus en casos con Dice bajo)

**Hipótesis**: durante el entrenamiento, el oversample de 0.33 es
uniforme sobre todos los pacientes del split. Ponderar el sampling
por "dificultad histórica" (1/Dice del fold previo, o peso fijo
manual para los 4–5 casos hard) fuerza más lotes con esos pacientes
sin subir el oversample global (que ya se vio que a 0.66 destruye
el modelo).

**Diseño**: Modificar el `DataLoader2D` (o generar `weights` en el
`case_sampler`) para que IOG1/IOG10/IOG38/IOG29/IOG28 tengan peso
~2× el resto. Usar solo en el split de training, no tocar
validation. Riesgo: sobreajuste a esos casos → regresión en el
resto.

**Sanity check en fold 2** (contiene IOG1, IOG38 — ambos hard):

```bash
nnUNetv2_train 501 2d 2 \
    -tr nnUNetTrainerALT_os033_hard2x_250epochs -p nnUNetPlans --npz
```

Criterio: IOG1 > 0.2 Y IOG38 > 0.4 Y fold 2 mean ≥ 0.63.

**Costo**: ~3h dev (sampler personalizado no es trivial en
nnU-Net v2) + 2h sanity + 10h 5-fold.
**Objetivo**: cerrar parcial IOG1/IOG38 sin tocar gate.

##### Orden de ejecución recomendado

1. ~~**Exp. 2a (sigmoide)** primero.~~ **✓ cerrado**: +0.0017 vs hard
  gate (0.7822 en producción). Script `scripts/ensemble_gated.py`
   ahora soporta `--gate-mode sigmoid`, `--tau`, `--use-confidence`,
   `--conf-power`. Sin GPU, reproducible.
2. **Exp. 2b (augment inv)** ← **siguiente**. Ataca IOG38 (0.232 →
  objetivo 0.50+) y posiblemente IOG1 (0.142). Requiere 2h dev + 2h
   sanity + 10h 5-fold si pasa.
3. **Exp. 2c (case-aware)** solo si 2a+2b no llegan a 0.79. Es el más
  intrusivo y el más propenso a regresiones invisibles.

#### Exp. 3-B1 — `oversample=1.0 + GT dilate 1 vox` (ejecutado, ARCHIVADO)

**Fecha de cierre**: 2026-04-23.

**Hipótesis**: antes de saltar a cascada formal detector→seg, una
variante barata del 3D branch — forzar `oversample_foreground_percent = 1.0`
(todos los patches contienen foreground) y dilatar el GT 1 vóxel
durante entrenamiento — produciría un "detector de alta recall" que
rescataría los hard-fails cuando 2D predice empty, bajo gating.

**Setup**:

- Trainer: `custom_trainers/nnUNetTrainerALT_os1_dilate1.py`
(`nnUNetTrainerALT_os1_dilate1_250epochs`), con:
  - `oversample_foreground_percent = 1.0`
  - Custom `DilateForegroundSegTransform` (1-vox 6-connectivity) en
  pipeline de training (no en val).
- Plans: `nnUNetTSMRIPlans` (3D, baseline reusado).
- Pretrained weights: baseline TS-MRI `Dataset850` fold_0.
- 5-fold full entrenado en remoto 2× RTX 5080 (4 folds paralelos 2/GPU
  - fold_4 DDP). Cada fold `batch_size=4`, 250 epochs, `--npz` habilitado.
- Ensemble: `scripts/wake/03_run_gated.sh` ejecuta 3 variantes
(`hard`, `sigmoid`, `sigmoid+conf`) con 2D = baseline
`nnUNetTrainerALT_os033_250epochs` sin cambios.

**Resultados standalone 3D** (46 casos val cross-fold):


| métrica   | base 3D (`os033`) | Exp. 3-B1  | Δ           |
| --------- | ----------------- | ---------- | ----------- |
| mean Dice | 0.7076            | **0.5580** | **−0.1496** |
| fold_0    | ~0.65             | 0.4412     | −0.21       |
| fold_1    | ~0.68             | 0.5706     | −0.11       |
| fold_2    | ~0.70             | 0.5275     | −0.17       |
| fold_3    | ~0.72             | 0.6222     | −0.10       |
| fold_4    | ~0.75             | 0.6414     | −0.11       |


Winners vs base (n=6): IOG1 (+0.12), IOG9 (+0.05), IOG10 (+0.04),
IOG12 (+0.03), IOG31 (+0.02), IOG28 (+0.00). Losers (n=36) incluyen
colapsos catastróficos: IOG24 0.612 → 0, IOG36 0.772 → 0.184,
IOG4 0.486 → 0.005, IOG20 0.809 → 0.416, IOG29 0.380 → 0.

**Resultados gated** (vs v2c sigmoid baseline 0.7822):


| variante gated | mean Dice  | Δ vs v2c    | wins | regs |
| -------------- | ---------- | ----------- | ---- | ---- |
| exp3b1_hard    | 0.6558     | −0.1263     | 4    | 30   |
| exp3b1_sigmoid | 0.6798     | −0.1024     | 6    | 29   |
| exp3b1_sigconf | **0.7086** | **−0.0736** | 5    | 24   |


Mejor variante (sigconf, v_min=50, τ=200, conf_power=2) confirma
hipótesis parcial:

- **Big wins** (+≥0.10): IOG10 (0.404 → 0.668, +0.264), IOG1
(0.142 → 0.306, +0.163). Son exactamente los targets blandos que
se esperaba rescatar.
- **Big regs** (−≥0.10): IOG36 (−0.608), IOG4 (−0.666), IOG48 (−0.633),
IOG54 (−0.306), IOG15 (−0.284), IOG26 (−0.214), IOG2 (−0.183),
IOG35 (−0.171), IOG47 (−0.124). 9 casos con pérdidas masivas.
- **Hard-fails absolutos no resueltos**: IOG38 0.233 → 0.211 (−0.02),
IOG40 sigue 0, IOG45 sigue 0. La dilatación de 1 vóxel no rescata
los casos extremos.

**Diagnóstico**: la combinación `os=1.0` (todos los patches con FG,
sin contexto "vacío") + `dilate 1` infla la predicción 3D de forma
generalizada. El gating por confianza 2D funciona bien cuando el 2D
está casi-seguro, pero en el "borderline" (p_fg 2D modesta, IOG4
IOG36 IOG48 IOG54), el sigmoid le pasa peso al 3D y el 3D contamina
con FPs masivos. Los +0.42 gated mean sobre los 2 wins no compensan
los −2.6 acumulados en las 9 regresiones.

**Veredicto**: **ARCHIVADO**. El 3D modificado es demasiado eager;
reemplazar el 3D del gate por este empeora en 0.074. Los 2 wins
(IOG1, IOG10) son consistentes con la teoría pero la degradación
colateral los sepulta.

**Insights salvables**:

1. `oversample=1.0` solo (sin dilate) probablemente tiene menos
  overfit al FG pero mantiene patches de contexto insuficientes.
   Podría probarse como `Exp. 3-B0`.
2. La dilatación de 1 vóxel *sí* rescata IOG1 / IOG10 sin reentrenar
  el 2D. Si se pudiese aplicar selectivamente (solo cuando 2D vote
   empty con alta confianza), habría +0.009 mean en esos 2 casos sin
   regresiones. Esto pide un **gate triplicado**: 2D → si empty →
   3D baseline → si también empty → 3D modificado. Costo bajo, lo
   registro como idea para Semana 3.
3. Los 3 hard-fails absolutos (IOG38 / IOG40 / IOG45) **no se
  rescatan con dilate simple**. Requieren cascada detect→seg con
   detector a menor resolución (Variant A: `detector_lowres`) o
   nnDetection.

**Siguiente paso recomendado**: en vez de Variant A inmediatamente,
escribir el **gate triplicado** (1 día) y medir si recupera
IOG1+IOG10 sin perder los 9 casos que colapsan con sigconf naive.
Si lo logra, gated mejoraría +0.009 mean (a ~0.791) sin más training.
Si no, proseguir con Variant A lowres.

**Archivos**:

- Training: `custom_trainers/nnUNetTrainerALT_os1_dilate1.py`
- Resultados 3D: `nnunet_env/nnUNet_results/.../nnUNetTrainerALT_os1_dilate1_250epochs__nnUNetTSMRIPlans__3d_fullres/` (1.2 GB)
- Gated summaries: `.../gated_ensemble_exp3b1_{hard,sigmoid,sigconf}/summary.json`
- Scripts wake: `scripts/wake/{01..04}_*.sh` + `README.md`

#### Exp. 3 — Cascada detect → seg (cierra IOG40, IOG45)

**Hipótesis**: los dos fails duros son lesiones muy pequeñas (405
y ≤ 5 k voxels, 4–5 slices). Una cascada donde un detector ligero
propone ROIs y luego un segmentador fino opera dentro del ROI los
rescata.

**Opciones**:

- a. **nnDet** (`MIC-DKFZ/nnDetection`). Más trabajo de infra pero
  con soporte oficial.
- b. **UNet 3D ligero + heatmap de centros** a spacing nativo, crop
  64³ en cada pico, y pasar al `ALT_os033_250ep` actual como
  segmentador.

Recomiendo (b): 1 script de entrenamiento del detector + 1 script
de inferencia + wrapper que llama `nnUNetv2_predict` en cada ROI.
~3–4 días de trabajo.

**Costo**: alto relativo al beneficio numérico (2 casos → +0.04
promedio si ambos pasan de 0 a 0.5). Se justifica si se necesita
cerrar los fails absolutos para el entregable.

#### Exp. 4 — Sweep de `oversample` (marginal, barato)

Falta añadir 2 variantes (`_os025_250epochs`, `_os040_250epochs`)
en `custom_trainers/nnUNetTrainerALT.py` (~5 min). Luego:

```bash
for TR in nnUNetTrainerALT_os025_250epochs \
          nnUNetTrainerALT_os040_250epochs; do
  for C in 2d 3d_fullres; do
    [[ "$C" == "3d_fullres" ]] && P=nnUNetTSMRIPlans || P=nnUNetPlans
    nnUNetv2_train 501 "$C" 1 -tr "$TR" -p "$P" --npz
  done
done
```

**Costo**: ~3h. **Retorno**: +0.01 o descartar.

### 8.3. Orden sugerido para Semana 2 (actualizado post Exp. 2a)

Con Exp. 1 + 1a archivados y Exp. 2a cerrado (+0.0017 → gated v2c =
0.7822), el plan pivota a 2b (augment inversión) y los fails duros:


| Día                         | Tarea                                                                                                                                                                           | Output esperado                                                                                                                                                                                                                                                                | Bloqueador |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| ~~1 (mañana)~~              | ~~**Exp. 2a (sigmoide)**~~                                                                                                                                                      | ✓ 0.7822                                                                                                                                                                                                                                                                       | —          |
| 1 (tarde)                   | Inspección visual IOG10 / IOG12 / IOG35 en ITK-SNAP para confirmar si son "desplazamiento" (arreglable con augment mejor) o "error estructural" (no arreglable sin reentrenar). | Decisión: seguir con 2b/2c o saltar a 3.                                                                                                                                                                                                                                       | —          |
| ~~2 (mañana)~~              | ~~Escribir trainers de Exp. 2b y 2b'~~                                                                                                                                          | ✓ `custom_trainers/nnUNetTrainerALT_inv.py` con `nnUNetTrainerALT_os033_inv_250epochs` (2b: InvertImage) y `nnUNetTrainerALT_os033_invgamma_250epochs` (2b': DA5 gamma-inv boost). Expuestos en `run_training.sh` como `TRAINER=ALT_OS033_INV` / `TRAINER=ALT_OS033_INVGAMMA`. | —          |
| ~~2 (tarde)~~               | ~~Sanity fold 2 Exp. 2b~~                                                                                                                                                       | **FALLA** ambos criterios (fold 2 mean 0.590 vs 0.627; IOG38 0.394 < 0.45). Insight salvable: IOG38 sí sube +0.161, el problema es que p=0.15 regresa los casos grandes.                                                                                                       | —          |
| ~~2 (tarde, continuación)~~ | ~~Sanity fold 2 Exp. 2b' (invgamma)~~                                                                                                                                           | Pasa fold 2 mean +0.044; rescate de IOG28.                                                                                                                                                                                                                                     | —          |
| ~~2 (noche)~~               | ~~Sanity fold 0 Exp. 2b'~~                                                                                                                                                      | **Falla** mean (−0.021); regresa IOG10, IOG4.                                                                                                                                                                                                                                  | —          |
| ~~2 (noche + CPU)~~         | ~~Medición SNR + gated test cruzado~~                                                                                                                                           | **Gated test −0.0177**; IOG28 NO es contraste atípico (SNR_shell=+1.57); rescates B-regime no aportan al gated. Rama 2b cerrada definitivamente.                                                                                                                               | —          |
| **3**                       | **Exp. 3 — cascada detect→seg**. Targets duros: IOG40, IOG45, IOG1, IOG10, IOG38.                                                                                               | 2D + 3D baseline + detector → gated. Target gated = 0.79.                                                                                                                                                                                                                      | —          |
| 4                           | Decidir: ¿Exp. 2c (case-aware sampler) o Exp. 3 (cascada detect→seg)?                                                                                                           | Plan Semana 3.                                                                                                                                                                                                                                                                 | —          |
| 4–5                         | Ejecutar la rama elegida.                                                                                                                                                       | —                                                                                                                                                                                                                                                                              | —          |


Key difference vs plan Semana 1: **Exp. 2a agotó los rescates sin
reentrenar; quedan 0.0111 pts hasta el oracle y esos sí requieren
modificar el pipeline de entrenamiento (2b) o el modelo (3).**

### 8.4. Objetivos numéricos Semana 2 (actualizado post Exp. 2a)


| Escenario                                                     | Dice gated esperado    | Estado                                                         |
| ------------------------------------------------------------- | ---------------------- | -------------------------------------------------------------- |
| Cierre Semana 1 (gated v2, ρ=0.10)                            | 0.7730                 | hecho                                                          |
| Gate retuneado (gated v2b, hard ρ=0.40)                       | 0.7805                 | reemplazado                                                    |
| **Gate sigmoide+conf (gated v2c, v_min=1000, τ=20) ← actual** | **0.7822**             | **actual**                                                     |
| Exp. 1 full (pretrain 2D)                                     | 0.7749                 | archivado                                                      |
| Exp. 1a K=1 (shallow)                                         | 0.7155 proy.           | archivado                                                      |
| Exp. 1a mix50 (A=0.5)                                         | 0.7220 proy.           | archivado                                                      |
| Exp. 2b (InvertImage p=0.15, +gate v2c)                       | — (no promoción)       | **archivado** — regresión fold 2                               |
| Exp. 2b' (DA5 gamma-inv boost, +gate v2c)                     | — (gated test −0.0177) | **archivado** — gated test definitivo                          |
| Exp. 2c (case-aware sampler)                                  | —                      | **descartado** — ganancia esperada +0.003, no vale el esfuerzo |
| Exp. 3 (cascada detect→seg)                                   | 0.79 – 0.80 (target)   | **próximo**                                                    |
| Exp. 2b + Exp. 2c (case-aware)                                | 0.79 – 0.80            | pending                                                        |
| Exp. 2 stack + Exp. 3 (cascada IOG40/45)                      | 0.80 – 0.82            | pending                                                        |
| Exp. 3-B1 (os=1.0 + GT dilate 1 vox, sigconf gate)            | 0.7086 medido          | **archivado** — regresión −0.0736 vs v2c                       |
| **Oracle `max(2D, 3D)` (techo sin re-entrenar)**              | **0.7933**             | referencia                                                     |


Lectura: el techo sin reentrenar pesos es 0.7933 (brecha restante
0.0111 desde v2c). Exp. 2a agotó los rescates de gate post-hoc;
pasar de ahí requiere tocar el entrenamiento (2b/2c) o añadir un
modelo nuevo (3).

### 8.5. Para el próximo chat

Punto de partida recomendado para retomar:

> *"Lee §8 del ANALYSIS.md. Producción actual: gated v2c (sigmoid
> v_min=1000, τ=20, use-conf) = **0.7822**.
>
> **Rama Exp. 2 (augment de intensidad) CERRADA**:
>
> - 2b (`inv` p=0.15): fold 2 mean −0.037, archivado.
> - 2b' (`invgamma`): fold 2 +0.044, fold 0 −0.021, gated test
> (invgamma 2D + baseline 3D) −0.0177 sobre 19 casos compartidos.
> Único win gated: IOG38 +0.155. Rescates IOG28/IOG36 caen en
> B-regime y el gated ya los cubría. Archivado.
> - 2c (case-aware): descartado — ganancia esperada +0.003.
>
> Insight nuevo: `GammaTransform(p_invert=1, p_retain_stats=1)` no
> es un flip de signo sino un remapeo del histograma preservando
> mean/std. Ver §8.2.2 para el mecanismo y §8.2.3 para razones de
> descarte. IOG28 NO es contraste atípico (SNR_shell=+1.57 medido).
>
> **Exp. 3 — cascada detect→seg. Estado 2026-04-23 al cerrar sesión**:
>
> **Exp. 3-B1 (os=1.0 + GT dilate 1 vóxel) ARCHIVADO**. Trainer
> `nnUNetTrainerALT_os1_dilate1_250epochs` entrenado 5-fold en 2×
> RTX 5080 (~7.7 h wall-clock). Standalone 3D = 0.558 mean (−0.15 vs
> 0.708). Mejor variante gated (sigconf v_min=50 τ=200 conf²) =
> **0.7086 → −0.0736 vs v2c** (las otras dos peores: hard −0.126,
> sigmoid −0.102). Wins en IOG1 (+0.163) e IOG10 (+0.264) pero 9
> regresiones ≥0.10 (IOG4 −0.67, IOG36 −0.61, IOG48 −0.63, IOG54
> −0.31, IOG15 −0.28, IOG26 −0.21, IOG2 −0.18, IOG35 −0.17, IOG47
> −0.12). IOG38/40/45 siguen en 0 — dilatación 1-vox no rescata los
> hard-fails absolutos. Ver §8.2 'Exp. 3-B1' para detalles completos.
>
> **Opciones abiertas (ranking recomendado)**:
>
> 1. **Gate triplicado (barato, sin reentrenar, 1 día)**. Flujo:
>   '2D → si empty o v<50 → 3D baseline os033 → si también empty
>    → 3D Exp. 3-B1 (os1_dilate1)'. El Exp. 3-B1 solo actúa cuando
>    ambos modelos confiables se rinden, así sus wins (IOG1, IOG10)
>    llegan sin contaminar IOG4/IOG36/IOG48/etc. donde el 2D-baseline
>    sí vota fuerte. Techo estimado: +0.009 a +0.015 mean (nuevo
>    gated ~0.791–0.797). Implementación: extender
>    `scripts/ensemble_gated.py` con modo 'triple' o crear
>    `ensemble_triple.py`. Data local ya lista
>    (`gated_ensemble_exp3b1_`* + validation npz de los 3 modelos).
> 2. **Exp. 3-B0 (os=1.0 SIN dilate)**. Entrenar solo el oversample
>   puro (quitar `DilateForegroundSegTransform`) para separar las
>    contribuciones. Si standalone recupera a ~0.65-0.68, confirma
>    que el daño viene del dilate; si sigue ~0.55, el daño es del
>    os=1.0 y ambos son malos por sí solos. Costo: ~7-8 h GPU
>    2× RTX 5080 (re-rentar nodo).
> 3. **Variant A (`detector_lowres`)**. Detector 3D a spacing 4× más
>   grueso (por ej. 4×1.2×1.2) + segmentador fino dentro del ROI.
>    Apunta a IOG38/40/45 que son lesiones <5k vóxels. Costo alto
>    (~2 días dev + 10 h GPU) pero es la única ruta a los 3
>    hard-fails absolutos.
> 4. **Congelar en v2c (0.7822)** y documentar límite. Oracle 0.7933
>   da techo teórico ~1 pt arriba; ninguna ruta estudiada lo cierra
>    barato.
>
> **Recomendación**: empezar por (1). Si el gate triplicado sube
> mean ≥ +0.005 sin regresión >0.05 en ningún caso ya ganado, se
> promueve como v3. Si no, (3) Variant A es la próxima inversión.
>
> **Estado de infraestructura al cerrar**:
>
> - Nodo remoto 74.48.140.178:52571: **a apagar por el usuario**
> (GPUs idle). Resultados Exp. 3-B1 ya rsync-eados a local
> (1.2 GB en
> `nnunet_env/nnUNet_results/Dataset501_ALT_T1/nnUNetTrainerALT_os1_dilate1_250epochs__nnUNetTSMRIPlans__3d_fullres/`).
> - Local: ANALYSIS.md sincronizado; scripts `scripts/wake/0{1..4}_*.sh`
> listos para reuso en próxima iteración.
> - Git: sin commits de Exp. 3-B1 aún — pendiente commit + push del
> trainer nuevo y ANALYSIS.md antes de cerrar."*

Recursos al cierre de Exp. 2a:

- **Ganador actual**: gated v2c, **0.7822** 5-fold, con
`scripts/ensemble_gated.py --gate-mode sigmoid --min-fg-voxels 1000 --tau 20 --use-confidence`
sobre nopretrain 2D + TS-MRI pretrain 3D.
- **Ganador previo (hard gate)**: gated v2b = 0.7805 con
`scripts/ensemble_gated.py --gate-mode hard --min-fg-voxels 50 --min-fg-ratio 0.40`.
Queda disponible como fallback si el sigmoid no tiene los
softmax `.npz` en disco.
- **Oracle (techo sin reentrenar)**: 0.7933 (brecha restante 0.0111).
- **Fails absolutos residuales**: IOG40, IOG45 (ambas configs en 0;
sólo Exp. 3 los rescata).
- **Casos con Dice bajo que aún pueden subir** (objetivos de Exp. 2b+):
IOG10 (0.41, 3D mejor, Δ a max = +0.180), IOG35 (0.80, ya rescatado
por v2c), IOG19 (0.81), IOG12 (0.88, ya rescatado por v2c),
IOG38 (0.23, candidato claro para augment inv), IOG1 (0.14).
- **Scripts extendidos**: `scripts/ensemble_gated.py` con
`--gate-mode {hard, sigmoid}`, `--tau`, `--use-confidence`,
`--conf-power`. Soporta ausencia del ensemble dir (sintetiza
soft-avg desde los softmax).
- **Scripts a extender**: `custom_trainers/nnUNetTrainerALT.py`
(Exp. 2b/2c).
- **Scripts archivados (quedan por trazabilidad)**:
`scripts/convert_tsmri_weights_3d_to_2d.py` (con flags
`--max-transfer-encoder-stage` y `--mix-ratio`; pretrain de TS-MRI
confirmado como negativo en este dataset).
- **Artefactos en disco**:
  - `nnunet_env/nnUNet_results/.../gated_ensemble_v2c_sigmoid/` (ganador actual).
  - `nnunet_env/nnUNet_results/.../gated_ensemble_v2b_hard/` (fallback).
  - `nnunet_env/nnUNet_results/...__2d/` (nopretrain, ganador base).
  - `nnunet_env/nnUNet_results/...__2d.pretrain/` (full, archivado).
  - `nnunet_env/nnUNet_results_k1/` (shallow K=1, archivado).
  - `nnunet_env/nnUNet_results_mix50/` (mix A=0.5, archivado).


## 9. Dual-channel T1+T2 — `Dataset503_ALT_T1T2`

Nueva rama (post-Semana-2) que consume T1 y T2 como **dos canales de
entrada** en un solo nnU-Net en vez de dos modelos independientes + gate.
Construida por `scripts/build_t1t2_dataset.py`, disparada desde
`run_training.sh` con `T1T2=1`.

### 9.1 Geometría intra-paciente (46 comunes)

T1 y T2 ya viven en **coordenadas de escáner / paciente compartidas**, así
que basta un resampleo por world-coords (sin registration) para alinearlos.

| Métrica intra-paciente | Conteo |
| ---------------------- | ------ |
| Pacientes comunes T1∩T2 | 46 |
| Grids idénticos (tamaño+spacing+origin+dir) | 5 / 46 |
| Direcciones idénticas (axial/coronal/sagital) | 34 / 46 |
| Direcciones discordantes (reformats) | 12 / 46 |
| Nº de slices distinto entre T1 y T2 | 22 / 46 |

Dice inter-rater (T2-mask resampleado al grid T1, por world-coords):

- 34 casos same-direction: **0.80–0.96**.
- 12 casos direction-mismatch: **0.69–0.93** (peores: `IOG45` 0.70,
  `IOG35` 0.69, `IOG28` 0.76, `IOG47` 0.79).

No hace falta image registration v1; los artefactos catastróficos los
captura el guard `MIN_FOREGROUND_RETAINED = 0.5` del pipeline existente.

### 9.2 Decisiones de diseño

- **Reference grid = T1** por paciente. `imagesTr/<case>_0000.nii.gz` es
  idéntico al de `Dataset501`; `_0001.nii.gz` es T2 resampleado (linear)
  al grid T1. Pierde resolución in-plane cuando T2 es más fino que T1
  (ej. `IOG15` T2=0.49 mm vs T1=0.76 mm) — trade consciente de recall
  vs detail. Reversible (`--src` y edit menor al builder).
- **Fusión de máscaras = union (OR)** por defecto. Alternativas vía
  `--fusion {union, intersection, staple}`:
  - *Union*: máximo recall, coincide con la definición clínica de ALT
    (T1 muestra grasa; T2 muestra edema/periferia). Inflá volúmenes
    10–30 % en casos de desacuerdo alto.
  - *Intersection*: sólo consenso estricto. Mata recall en los tumores
    pequeños (405–5 k vox) que ya fallan a Dice 0.
  - *STAPLE*: con sólo 2 raters colapsa a OR/AND ponderado; útil si
    hubiera más raters.
- **Sin TotalSegmentator pretrain** en Dataset503 (TS es 1-canal; mover
  plans rompe la conv-0). Ambas configs (`2d`, `3d_fullres`) entrenan
  con `nnUNetPlans` desde cero.

### 9.3 Pérdidas esperadas por el guard

El builder descarta el caso cuando el T2-mask retiene menos del 50 % de
sus voxels foreground al resamplearlo al grid T1. En la corrida de
validación (46 comunes, fusion=union) caen 3 casos: `IOG15`, `IOG26`,
`IOG50` (T2 con in-plane muy fino + tumor lineal, el NN downsample los
adelgaza). Final: **43 / 46 casos** escritos en `Dataset503_ALT_T1T2`.
Reentrenar con `--fusion intersection` o `--strict` es una corrida
aparte (no rompe 501/502).

### 9.4 Cómo lanzarlo

```bash
# Build + 5-fold, 2 configs, 2 GPUs, trainer ALT_OS033 250ep:
T1T2=1 TRAINER=ALT_OS033 EPOCHS=250 NUM_GPUS=2 bash run_training.sh

# Intersección en lugar de unión (más conservador):
T1T2=1 FUSION=intersection TRAINER=ALT_OS033 bash run_training.sh

# Inferencia (imagesTs = <case>_0000.nii.gz T1 + <case>_0001.nii.gz T2
# ya resampleado al grid T1):
nnUNetv2_predict -i <T1T2_images_dir> -o <out_dir> \
  -d 503 -c 3d_fullres -tr nnUNetTrainerALT_os033_250epochs \
  -p nnUNetPlans -f 0 1 2 3 4
```

### 9.5 Qué comparar contra el actual ganador (v2c gated)

El GT de 503 es el **fusionado** (distinto del GT de 501/502), así que
el Dice 5-fold no es directamente comparable con la tabla TL;DR. Dos
comparaciones honestas al cerrar:

1. Dice de 503 (5-fold) contra el **max oracle(2D 501, 3D 501)** en los
   mismos 43 casos → mide cuánto cierra la brecha al techo single-modality.
2. Dice de 503 contra el **gated v2c** evaluado en los mismos 43 casos
   usando el GT fusionado (requiere reprocesar `ensemble_gated.py` con el
   GT de 503, o comparar contra ambos GTs single-modality con un "max").

Hasta que 503 pase a v2c en (2), **v2c gated sigue en producción**
(§8, 0.7822).

### 9.6 Riesgos / caveats

- **Sin image registration** en 12 casos direction-mismatch. El guard
  MIN_FOREGROUND_RETAINED captura los peores, pero `IOG45` (Dice 0.70)
  y `IOG47` (header roto + wraparound int16 ya tratado) merecen QC visual.
- **TS-MRI pretrain perdido**: ~5 pp de mejora en 3d_fullres que daba a
  Dataset501. Compensar con más épocas o con un futuro script que clone
  la conv-0 al canal 1 en los pesos TS-MRI.
- **Union inflata volúmenes 10-30 %**: si precision cae vs recall, re-run
  con `FUSION=intersection`. La métrica a vigilar es Dice en los
  pequeños (`IOG45`, `IOG29`, `IOG38`, `IOG40`, `IOG1`, `IOG4`, `IOG43`,
  `IOG35`) — ahí está el techo del oracle.
- **Desalineo de channels** si el usuario hace inferencia con un par
  T1/T2 sin resamplear T2 al grid T1: el trainer espera
  `<case>_0000.nii.gz` y `<case>_0001.nii.gz` en el **mismo grid**. Doc
  en README y prompt final del script lo recuerdan.

### 9.7 Código y artefactos

- **Scripts nuevos**: `scripts/build_t1t2_dataset.py`,
  `scripts/inspect_fusion.py`.
- **Scripts modificados**: `scripts/prepare_pretrain_plans.py` (skip TS
  transfer para datasets multichannel), `run_training.sh` (modo `T1T2=1`,
  variable `FUSION`, rama de inferencia 2-canal).
- **Artefactos en disco**:
  - `nnunet_env_T1T2/nnUNet_raw/Dataset503_ALT_T1T2/`
    (`imagesTr/<case>_0000.nii.gz`, `imagesTr/<case>_0001.nii.gz`,
    `labelsTr/<case>.nii.gz`, `dataset.json`, `fusion_report.json`).
  - `nnunet_env_T1T2/nnUNet_preprocessed/Dataset503_ALT_T1T2/` (sólo
    `nnUNetPlans*`; no hay `nnUNetTSMRIPlans`).
  - `nnunet_env_T1T2/nnUNet_results/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/`
    (5 folds, 43 casos).

### 9.8 Resultados 5-fold medidos — T2-only y T1+T2 (2026-04-24)

#### 9.8.1 T2-only (`Dataset502_ALT_T2`, ALT_os033_250ep, 3d_fullres, TS-MRI pretrain)

Entrenado en `nnunet_env/` con el mismo trainer ganador de T1. GT =
máscara T2 nativa (no fusionada).

| Fold | n | mean Dice | min | max |
| ---- | - | --------- | --- | --- |
| 0    | 10 | 0.6085   | 0.000 | 0.951 |
| 1    | 9  | 0.6675   | 0.000 | 0.944 |
| 2    | 9  | 0.5031   | 0.000 | 0.935 |
| 3    | 9  | 0.5894   | 0.000 | 0.931 |
| 4    | 9  | 0.7317   | 0.294 | 0.903 |
| agg  | 46 | **0.6198** | 0.000 | 0.951 |

- **Fails (Dice=0)**: 8 casos (IOG4, IOG9, IOG29, IOG31, IOG35, IOG40,
  IOG43, IOG45). Los 6 ya documentados como "fallos duros" en T1
  (§1.2, §5.2) reaparecen — el cambio de modalidad no rescata ninguno.
- **Nuevos fails únicos de T2**: IOG15 (0.023), IOG50 (0.091). Ambos
  tienen T2 in-plane fino (IOG15 0.45 mm; IOG50 fino también). El
  resampling al target spacing del plan los difumina — mismo mecanismo
  que §1.2 #3, pero del lado T2.
- **Ganancias reales sobre T1**: IOG38 (**0.000 → 0.944**, +0.94, el
  caso hipointenso en T1 que §1.2 señalaba como candidato a augment de
  inversión). IOG28 queda comparable (T1/3D 0.810 vs T2 0.865).
- **Varianza fold-to-fold** mayor que T1 (0.23 Dice de spread,
  fold 2=0.503 vs fold 4=0.732) → splits de 502 no están
  estratificados; re-estratificar sube la media.

**Veredicto T2-only**: peor que T1-only en media, peor que T1/DA5
baseline (0.637). **Archivado como config independiente**. Único valor
rescatable: IOG38 es una ganancia enorme que sugiere ensemblar T2
con T1 (ver 9.8.3).

#### 9.8.2 T1+T2 union (`Dataset503_ALT_T1T2`, ALT_os033_250ep, 3d_fullres, sin pretrain)

Entrenado en `nnunet_env_T1T2/`, 43 casos (guard descarta IOG15, IOG26,
IOG50 — §9.3). GT = unión T1 ∪ T2 resampleado.

| Fold | n | mean Dice | min | max |
| ---- | - | --------- | --- | --- |
| 0    | 9  | 0.4799   | 0.000 | 0.958 |
| 1    | 9  | 0.6270   | 0.000 | 0.946 |
| 2    | 9  | 0.7747   | 0.000 | 0.972 |
| 3    | 8  | 0.6256   | 0.000 | 0.951 |
| 4    | 8  | 0.5030   | 0.000 | 0.947 |
| agg  | 43 | **0.6038** | 0.000 | 0.972 |

- **Fails (Dice=0)**: 8 (IOG7, IOG9, IOG10, IOG28, IOG31, IOG38,
  IOG40, IOG45). Tres ya eran fails duros (IOG40, IOG45, IOG38).
  Los nuevos fails unions-inducidos son **IOG7, IOG9, IOG10, IOG28**
  — casos que **en T1/3D single-channel funcionaban** (IOG7 0.849,
  IOG28 0.810).
- **Bajos (0 < D < 0.5)**: 7 (IOG1, IOG2, IOG4, IOG35, IOG43, IOG46,
  IOG54). Nuevas regresiones serias: IOG2 (0.93 → 0.38), IOG46
  (0.90 → 0.50), IOG54 (0.86 → 0.32).
- **Ganancias reales por fusión**: IOG35 (0.000 → 0.460, +0.46),
  IOG29 (0.380 → 0.514, +0.13), IOG48 (0.000 → 0.723, +0.72).

##### Correlación con `fusion_report.json`

Cruzando los 8 fails + 7 bajos con las métricas de fusión:

| Predictor de regresión                            | Incidencia en fails | Incidencia global |
| ------------------------------------------------- | -------------------:| -----------------:|
| `direction_mismatch=true`                         | 6 / 8 (75 %)        | 13 / 43 (30 %)    |
| `retained > 1.3` (T2-fg infla al resamplear)      | 4 / 8 (50 %)        | 9 / 43 (21 %)     |
| `fused_fg / t1_fg > 1.15` (union amplía GT ≥15 %) | 5 / 8 (63 %)        | 11 / 43 (26 %)    |

Los tres predictores correlacionan entre sí, pero
`direction_mismatch=true` es el más fuerte: **el 75 % de los fails**
son casos donde T1 y T2 vienen en orientaciones distintas y fueron
resampleados world-coords sin registration. Los casos more-or-less
alineados (34/43) rinden bien — fold 2 completo es 0.77 porque
concentra same-direction.

##### Factores confounders

1. **No hay TS-MRI pretrain** en 503 por diseño (§9.2). La conv-0 no
   se puede clonar al canal 1 con la implementación actual. Contra el
   T1 3D equivalente (0.7076 **con** pretrain), la caída por falta de
   pretrain sola vale ~0.03–0.05 Dice (§1 tabla: DA5 baseline 3D
   0.637 → ALT 3D con pretrain 0.708, ≈ +0.07 del pretrain — pero eso
   mezcla trainer y pretrain; aislar requiere un run de control).
2. **Inflación del GT** en 11/43 casos (`fused_fg / t1_fg > 1.15`).
   El modelo intenta ajustar regiones que **sólo aparecen en T2** y
   no tiene canal suficiente (o confiable) para localizarlas en
   casos direction-mismatch → sobre-predice (IOG46: n_pred 91 434 vs
   n_ref 32 482) o colapsa a cero.
3. **Splits 503 no estratificados**. Fold 0 y fold 4 concentran los
   direction-mismatch peores; fold 2 los fáciles. El spread 0.48–0.77
   es 1.5× más ancho que el T1 (§6.2).

**Veredicto T1+T2 union**: archivado como config de producción. La
hipótesis "dos canales le dan al modelo más información" sólo se
sostiene en casos same-direction; en reformats la resampling sin
registration destruye alineación voxel-a-voxel y el modelo no puede
usar ambos canales. Empíricamente: **gana 3 casos (+1.3 pts
agregados), pierde 6–7 casos (−8+ pts agregados)**.

##### Update 2026-04-27 — **baseline Dataset503 = `fusion_union_v2_503_gated_v2c`** (sobre GT `union_v2`, `3d_fullres`)

Se entrenó de nuevo `Dataset503_ALT_T1T2` con el setting de fusión nuevo
(`union_v2`) en `3d_fullres`. Artefactos:

- `nnUNet_results_union_V2/Dataset503_ALT_T1T2/nnUNetTrainerALT_os033_250epochs__nnUNetPlans__3d_fullres/`

Desde este punto, distinguimos dos cosas:

1. **Definición de GT / setting de fusión**: `union_v2` (cómo se construye la máscara objetivo en 503).
2. **Baseline de rendimiento (lo que se toma como referencia en 503)**: **`fusion_union_v2_503_gated_v2c`**
   (porque es lo mejor que reporta el CSV para ese mismo GT).

En otras palabras: `union_v2` fija *qué* se evalúa; y **`gated_v2c` fija el baseline**
de *qué tan bien podemos hacerlo* sin re-entrenar un modelo multichannel específico.

Todas las métricas de esta subsección quedan fijadas por el CSV humano
`reports/per_case_baseline_and_mixed.csv` (regenerado tras correr el script “fusion union version”).

Métrica agregada (promedio por **paciente**; si el reporte trae `*_T1` y `*_T2`,
se promedia primero dentro de paciente y luego sobre pacientes):

 - **Baseline (503) = `fusion_union_v2_503_gated_v2c`**:
  - **Mean Dice = 0.8032** (n=46)
  - **Global Dice = 0.8815**
  - **Zeros (Dice=0)**: **1 / 46** → `IOG45`
  - **Lows (0 < Dice < 0.5)**: **2 / 46** → `IOG1` (0.017), `IOG31` (0.004)

 - **Referencia multichannel pura (no-baseline) = `fusion_union_v2_503_3d`**:
  - **Mean Dice = 0.7268** (n=46)
  - **Global Dice = 0.8327**
- **Δ vs union “viejo” (0.6038, n=43)**: **+0.1230** *(indicativo; no es comparación justa por n distinto y setting de fusión distinto)*.

Comparación justa contra la corrida `intersection` (misma lista de pacientes con valor en ambos;
en el CSV `intersection` tiene n=45 por 1 caso faltante).

Como el baseline ahora es `fusion_union_v2_503_gated_v2c`, hay **dos** comparaciones útiles:

1. **Efecto del setting de fusión** (manteniendo un “modelo” comparable): `union_v2_503_3d` vs `intersection_503_3d`.
2. **Efecto de la receta baseline** (qué referencia operacional conviene usar en 503): baseline `gated_v2c` vs alternativas.

- `fusion_intersection_503_3d` mean = **0.6705** (n=45)
- **Δ (union_v2 − intersection) = +0.0564** (sobre esos mismos 45)

###### Qué cambió realmente (por caso)

La mejora de `union_v2` viene de **matar menos casos a 0** comparado con `intersection`,
pero deja **regresiones puntuales grandes**.

- **Zeros (Dice=0)**:
  - `intersection_503_3d`: **6 / 45** → `IOG1, IOG10, IOG28, IOG31, IOG38, IOG9`
  - `union_v2_503_3d`: **2 / 46** → `IOG31, IOG45`
- **“Lows” (0 < Dice < 0.5) en `union_v2_503_3d`**: **6 / 46**
  - peores: `IOG40` 0.087, `IOG35` 0.217, `IOG54` 0.271, `IOG1` 0.280,
    `IOG28` 0.466, `IOG38` 0.482

Top deltas **union_v2 − intersection** (45 comunes):

- **Mayores rescates**:
  - `IOG10`: 0.000 → 0.727 (**+0.727**)
  - `IOG9`: 0.000 → 0.564 (**+0.564**)
  - `IOG38`: 0.000 → 0.482 (**+0.482**)
  - `IOG28`: 0.000 → 0.466 (**+0.466**)
  - `IOG43`: 0.588 → 0.872 (**+0.284**)
- **Mayores regresiones**:
  - `IOG45`: 0.648 → 0.000 (**−0.648**) *(nuevo hard-fail; requiere QC)*
  - `IOG55`: 0.933 → 0.626 (**−0.307**)
  - `IOG42`: 0.905 → 0.622 (**−0.283**)
  - `IOG40`: 0.294 → 0.087 (**−0.207**)

###### Baseline “completo” sobre `union_v2` (lo que reporta el CSV)

Sobre el mismo GT fusionado `union_v2`, el CSV computa cómo se comportan varias recetas.
**La regla de baseline en 503** queda entonces explícita:

- **Baseline (503)** = la mejor receta disponible en el CSV para GT `union_v2`
  ⇒ **`fusion_union_v2_503_gated_v2c`** (mean **0.8032**, global **0.8815**).

Para contexto, el “escalón” de recetas sobre ese mismo GT es:

- **`fusion_union_v2_503_2d`**: mean **0.7712**, global **0.8650**, zeros **2/46** (`IOG29, IOG45`)
- **`fusion_union_v2_503_softavg_2d3d_tta`**: mean **0.7692**, global **0.8676**, zeros **2/46** (`IOG29, IOG45`)
- **`fusion_union_v2_503_gated_v2b`**: mean **0.7890**, global **0.8714**, zeros **1/46** (`IOG45`)
- **`fusion_union_v2_503_gated_v2c`**: mean **0.8032**, global **0.8815**, zeros **1/46** (`IOG45`)

**Δ vs baseline (503 = gated v2c)**, medido sobre los mismos 46 pacientes:

- `fusion_union_v2_503_gated_v2b`: **−0.0142 mean** (21 wins / 21 losses / 4 ties).
- `fusion_union_v2_503_2d`: **−0.0320 mean** (7 wins / 36 losses / 3 ties).
- `fusion_union_v2_503_softavg_2d3d_tta`: **−0.0340 mean** (23 wins / 22 losses / 1 tie).
- `fusion_union_v2_503_3d` (multichannel puro): **−0.0764 mean** (15 wins / 30 losses / 1 tie).

Interpretación (baselineado):

1. **El baseline correcto de 503 no es “el modelo multichannel”** sino el
   **gate v2c** sobre ramas fuertes evaluado en ese GT: es el máximo en mean/global,
   y también es robusto en el tail al rescatar outliers (ver abajo).
2. **IOG45 define el “cero residual” del setting `union_v2`**: bajo el baseline
   (y también bajo 2D/softavg/gated/3D multichannel) queda en **Dice = 0**,
   sugiriendo un problema de (i) GT fusionado específico de ese caso, (ii) preproc/alineación,
   o (iii) distribución extrema de tamaño/contraste.
3. **Los “lows residuales” del baseline** son `IOG1` y `IOG31` (ambos casi 0),
   pero son *pocos* (2/46). En la práctica, esto es mucho mejor que el tail del multichannel puro,
   que deja varios casos <0.5.

###### Dónde el gate gana/perde contra el 3D multichannel

Si tomamos `union_v2_503_3d` como “multichannel puro”, y lo comparamos con
`fusion_union_v2_503_gated_v2c` (gate estilo 501 evaluado en el mismo GT):

- **Mejores rescates del gate**:
  - `IOG40`: 0.087 → 0.958 (**+0.871**)
  - `IOG54`: 0.271 → 0.933 (**+0.662**)
  - `IOG35`: 0.217 → 0.819 (**+0.602**)
  - `IOG38`: 0.482 → 0.847 (**+0.365**)
  - `IOG42`: 0.622 → 0.951 (**+0.329**)
- **Regresiones del gate**:
  - `IOG36`: 0.891 → 0.624 (**−0.267**)
  - `IOG1`: 0.280 → 0.017 (**−0.263**)
  - `IOG48`: 0.754 → 0.534 (**−0.220**)

Esto re-sugiere una lectura práctica: si el objetivo es **robustez per-caso**,
un gate tipo v2c sobre ramas fuertes tiende a “tapar” outliers (IOG40/54/35/38),
pero puede **sobre-castigar** casos donde el 3D multichannel ya estaba bien
(IOG36/48) o donde el 2D colapsa (IOG1).

#### 9.8.3 Próximos pasos propuestos

En orden de costo/beneficio:

1. **Gated T2-as-third-branch** *(coste: medio día, sin entrenar)*.
   La única señal nueva fuerte que T2 aporta es IOG38
   (0.000 → 0.944). Usar las predicciones 502 existentes como tercera
   rama del gated v2c con una regla simple: si las dos ramas T1 (2D y
   3D) predicen `n_pred < 500` (≈ el umbral que ya dispara el gate
   hard), probar T2. Impacto esperado 5-fold: +0.94 / 46 = **+0.020
   Dice** sólo por IOG38, con riesgo de regresión en los 8 fails de
   T2. Medible con un script corto sobre los `summary.json`
   existentes, sin reentrenar. Si sale positivo, promoción a v3.

2. **T1+T2 intersection, 1 fold** *(coste: ~10h GPU)*. El verdadero
   test de "la fusión pedagógica es el problema, no el
   multichannel". Entrenar fold 0 con `FUSION=intersection`; si sube
   de 0.48 a ≥0.65 con GT reducido, el problema es union-inflation y
   vale un 5-fold completo. Si no sube, archivar definitivamente la
   rama multichannel hasta que haya image registration.

3. **T1+T2 con TS-MRI transfer al canal 1** *(coste: ~2h dev + 50h
   GPU)*. Extender `scripts/prepare_pretrain_plans.py` para
   inicializar la conv-0 con `[w_TS, w_TS]` (clonar el canal T1 al
   canal T2) en vez de saltarse el transfer. Recupera ~0.03–0.05
   Dice del pretrain. Vale la pena **sólo si (2) sale positivo**.

4. **Image registration rígida T1↔T2 antes del builder** *(coste:
   ~1 día dev)*. Para los 12 casos direction-mismatch usar SimpleITK
   con rigid+mutualinfo antes de resamplear al grid T1. Reduce el
   `retained` errático y el `fused_fg/t1_fg` → 1. Es la corrección
   "correcta" pero altera 503 estructuralmente; ejecutar como
   `Dataset504_ALT_T1T2_reg` para no romper la comparación.

5. **NO hacer** más épocas / más augment en 503: el problema no es
   underfitting, es que la GT en direction-mismatch casos no
   describe una región coherente en ambos canales. Ver `IOG43`
   (same-grid, `retained=1.00`, rater-Dice 0.94 → Dice 0.013): ni
   siquiera los casos "bien alineados" garantizan que la unión sea
   aprendible. Ese es inspección visual, no entrenamiento.

La recomendación concreta: **(1) primero**, porque es gratis, cierra
una ganancia garantizada sobre IOG38, y nos dice sin reentrenar si T2
aporta **algo**. Si (1) sale < +0.01, enterrar 502 y 503 completos.
Si sale ≥ +0.01, entonces (2) para decidir si la rama multichannel
vale el esfuerzo de (3)+(4).

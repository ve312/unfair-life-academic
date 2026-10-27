# Evaluación Avanzada del Modelo

> **Ámbito.** Los pasos 03 y 04 del *pipeline* seleccionan y evalúan el modelo con una única partición de validación y una única partición de prueba. Este documento añade las comprobaciones estadísticas que un revisor exigiría a continuación: discriminación con intervalos de confianza, validación cruzada repetida con pruebas de significación, capacidad del modelo frente a datos e hiperparámetros, análisis de residuos y calibración por subgrupos, el techo de exactitud que imponen los datos y el grado de confianza en la garantía del 95 %.
>
> Todas las cifras y figuras proceden de `src/advanced_metrics.py`, que escribe `training-pipeline/artifacts/advanced_metrics_report.json` y seis figuras en `artifacts/figures/`. El *script* importa la configuración de los modelos de `03_trainer.py` y la política de decisión de `04_evaluator.py`, y comprueba al arrancar que `best_model.joblib` reproduce exactamente `test_predictions.parquet`. El conjunto de prueba solo se puntúa con artefactos ya ajustados: ninguna decisión de este documento se toma sobre él. La semilla es 42 en todas las secciones.

---

## 1. Discriminación: curvas ROC y precisión-exhaustividad, Gini y Kolmogorov-Smirnov

**Figura:** `discrimination_curves.png`

### 1.1. Métricas

Además del ROC-AUC y de la precisión promedio (AP), se reportan dos estadísticos habituales en la evaluación de modelos de riesgo:

$$
\text{Gini} = 2\,\text{AUC} - 1, \qquad
\text{KS} = \max_{t}\ \bigl|\,F_1(t) - F_0(t)\,\bigr|
$$

donde $F_1$ y $F_0$ son las funciones de distribución acumulada de la probabilidad predicha para las personas con y sin movilidad. El Gini reescala el AUC al intervalo $[-1, 1]$; el KS mide la máxima separación entre las dos clases a lo largo de la escala de probabilidad.

### 1.2. Validación (once candidatos, distribución natural)

| Modelo | ROC-AUC | Gini | AP | KS | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Stacking (HGB + LR → LR)** | **0,9280** | **0,8559** | **0,8303** | **0,6888** | **0,0879** | **0,2765** | **0,0073** |
| HistGradientBoosting | 0,9278 | 0,8556 | 0,8303 | 0,6868 | 0,0881 | 0,2766 | 0,0096 |
| HGB con pesos de clase | 0,9277 | 0,8554 | 0,8296 | 0,6846 | 0,1095 | 0,3317 | 0,1003 |
| Regresión logística | 0,9058 | 0,8116 | 0,7676 | 0,6434 | 0,1018 | 0,3198 | 0,0080 |
| Bosque aleatorio | 0,9030 | 0,8061 | 0,7660 | 0,6407 | 0,1029 | 0,3304 | 0,0193 |
| SVM RBF calibrada | 0,9013 | 0,8027 | 0,7573 | 0,6306 | 0,1049 | 0,3297 | 0,0234 |
| k-NN | 0,8467 | 0,6934 | 0,6305 | 0,5683 | 0,1288 | 0,8565 | 0,0442 |

El ensamble obtiene el mejor valor en las siete métricas. La variante del *gradient boosting* con pesos de clase iguala la discriminación (AUC 0,9277) con un Brier un 25 % peor y un ECE catorce veces mayor: la misma conclusión que la ablación de SMOTE, ahora con otra forma de reequilibrio.

### 1.3. Comparación pareada por *bootstrap*

Para saber si las diferencias son reales, se remuestrearon con reemplazo las 9758 personas de validación 2000 veces y, en cada remuestreo, se evaluaron **todos** los modelos sobre las mismas personas. La diferencia es así pareada y el intervalo refleja solo la incertidumbre de la comparación.

| Ensamble frente a… | ΔAUC (IC 95 %) | ΔAP (IC 95 %) | ΔBrier (IC 95 %) |
|---|---|---|---|
| Regresión logística | +0,0222 (0,0193; 0,0252) | +0,0627 (0,0545; 0,0714) | −0,0139 (−0,0157; −0,0121) |
| Bosque aleatorio | +0,0250 (0,0215; 0,0284) | +0,0644 (0,0555; 0,0731) | −0,0151 (−0,0170; −0,0131) |
| SVM RBF calibrada | +0,0266 (0,0231; 0,0299) | +0,0730 (0,0639; 0,0823) | −0,0170 (−0,0191; −0,0149) |
| k-NN | +0,0813 (0,0744; 0,0887) | +0,1995 (0,1824; 0,2171) | −0,0409 (−0,0444; −0,0376) |
| HistGradientBoosting | +0,00015 (−0,00012; 0,00040) | +0,00005 (−0,00065; 0,00066) | **−0,00019 (−0,00034; −0,00005)** |

Frente a los cuatro paradigmas alternativos, la superioridad del ensamble es inequívoca: ningún intervalo contiene el cero. Frente al *gradient boosting* solo, la discriminación es estadísticamente indistinguible, pero el Brier es mejor en el 99,5 % de los remuestreos. Es exactamente lo que se buscaba: el ensamble no pretendía discriminar mejor, sino conservar la discriminación del *gradient boosting* con mejores probabilidades.

### 1.4. Prueba (modelo desplegado)

| Métrica | Valor | IC 95 % *bootstrap* (2000 remuestreos) |
|---|---:|---|
| ROC-AUC | 0,9313 | 0,9259 – 0,9362 |
| Gini | 0,8625 | 0,8518 – 0,8724 |
| Precisión promedio | 0,8374 | 0,8251 – 0,8492 |
| KS | 0,6938 | 0,6805 – 0,7107 |
| Brier | 0,0863 | 0,0830 – 0,0897 |

Las cifras de prueba están dentro o por encima de las de validación, sin rastro de sobreajuste a la partición de selección. El panel derecho de la figura muestra las dos distribuciones acumuladas: la máxima separación (KS = 0,694) se alcanza en $p = 0{,}22$, junto al borde inferior de la banda de indecisión, lo que confirma que ese borde se sitúa donde las dos clases empiezan a mezclarse.

---

## 2. Validación cruzada estratificada repetida

**Figura:** `cross_validation.png`

### 2.1. Diseño

Una única partición de validación puede favorecer a un modelo por azar. Para descartarlo, las seis arquitecturas (las cinco familias y el ensamble, con su configuración exacta de `03_trainer.py`) se evaluaron con **validación cruzada estratificada 5 × 2**: cinco pliegues repetidos dos veces con particiones distintas, diez ajustes por modelo, todo ello **solo sobre el conjunto de entrenamiento** (29 274 filas). El ensamble repite internamente su propia validación cruzada de cinco pliegues en cada ajuste.

### 2.2. Resultados (media ± desviación típica sobre 10 pliegues)

| Modelo | ROC-AUC | AP | Brier | ECE | Rango medio (AP) |
|---|---|---|---|---|---:|
| **Stacking (HGB + LR → LR)** | **0,9268 ± 0,0027** | **0,8232 ± 0,0051** | **0,0892 ± 0,0015** | **0,0092 ± 0,0020** | **1,1** |
| HistGradientBoosting | 0,9266 ± 0,0024 | 0,8227 ± 0,0048 | 0,0894 ± 0,0014 | 0,0100 ± 0,0025 | 1,9 |
| Regresión logística | 0,9041 ± 0,0038 | 0,7604 ± 0,0062 | 0,1032 ± 0,0017 | 0,0114 ± 0,0014 | 3,6 |
| Bosque aleatorio | 0,9029 ± 0,0030 | 0,7619 ± 0,0072 | 0,1030 ± 0,0018 | 0,0175 ± 0,0039 | 3,8 |
| SVM RBF calibrada | 0,8992 ± 0,0038 | 0,7531 ± 0,0055 | 0,1061 ± 0,0016 | 0,0258 ± 0,0033 | 4,6 |
| k-NN | 0,8480 ± 0,0059 | 0,6261 ± 0,0134 | 0,1281 ± 0,0019 | 0,0412 ± 0,0045 | 6,0 |

El orden de los modelos es el mismo que en la partición de validación, y las desviaciones típicas (del orden de 0,005 en AP) son mucho menores que las distancias entre familias (más de 0,06). La selección no depende de la partición.

### 2.3. Pruebas de significación

**Prueba de Friedman.** Contrasta si los seis modelos son intercambiables comparando sus rangos en cada pliegue (Demšar, 2006). Con $\chi^2 = 45{,}37$ y $p = 1{,}2 \times 10^{-8}$, se rechaza con holgura que lo sean. El ensamble ocupa el primer puesto en nueve de los diez pliegues (rango medio 1,1).

**Prueba *t* corregida de Nadeau y Bengio.** Los pliegues comparten la mayor parte de sus datos de entrenamiento, de modo que la prueba *t* pareada habitual subestima la varianza y declara significativas diferencias que no lo son. Nadeau y Bengio (2003) corrigen la varianza con el cociente entre los tamaños de prueba y de entrenamiento:

$$
t = \frac{\bar{d}}{\sqrt{\left(\dfrac{1}{k} + \dfrac{n_{\text{test}}}{n_{\text{train}}}\right)\hat{\sigma}_d^2}}, \qquad k = 10,\ \ \frac{n_{\text{test}}}{n_{\text{train}}} = \frac{1}{4}
$$

| Ensamble − HistGradientBoosting | Diferencia media | *t* | *p* |
|---|---:|---:|---:|
| ROC-AUC | +0,00025 | 1,42 | 0,191 |
| Precisión promedio | +0,00049 | 1,53 | 0,161 |
| **Brier** | **−0,00020** | **−2,55** | **0,031** |
| ECE | −0,00081 | −0,79 | 0,451 |

La validación cruzada confirma con una prueba conservadora lo que mostró el *bootstrap* pareado: frente al *gradient boosting* solo, el ensamble no cambia la discriminación y mejora de forma significativa el error probabilístico. La mejora del ECE va en la misma dirección, pero con diez pliegues no alcanza significación.

---

## 3. Capacidad: curvas de aprendizaje, hiperparámetros y parada temprana

**Figura:** `capacity_analysis.png`

### 3.1. Curvas de aprendizaje

El *gradient boosting* y la regresión logística se entrenaron con submuestras estratificadas del 5 % al 100 % del conjunto de entrenamiento (tres semillas por tamaño) y se evaluaron siempre sobre la misma validación.

| Filas de entrenamiento | 1464 | 2927 | 5855 | 10 246 | 14 637 | 21 956 | 29 274 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP validación, HGB | 0,763 | 0,785 | 0,807 | 0,815 | 0,821 | 0,824 | 0,830 |
| AP entrenamiento, HGB | 0,939 | 0,921 | 0,901 | 0,883 | 0,869 | 0,866 | 0,864 |
| AP validación, regresión logística | 0,713 | 0,742 | 0,756 | 0,761 | 0,766 | 0,767 | 0,768 |

**Lectura.** La regresión logística se estabiliza a partir de unas 10 000 filas: su límite es estructural (no representa interacciones). El *gradient boosting* sigue mejorando con rendimientos decrecientes: duplicar los datos de 14 637 a 29 274 filas añade 0,0096 de AP. La distancia entre entrenamiento y validación se reduce de 0,18 a 0,03 a medida que crecen los datos, de modo que el modelo desplegado no está en régimen de sobreajuste. El Brier en validación desciende de 0,108 a 0,088, frente a 0,182 de un modelo que predijera siempre la tasa base.

### 3.2. Sensibilidad a los hiperparámetros

`03_trainer.py` declara que los hiperparámetros del *gradient boosting* proceden de una búsqueda aleatoria previa. Para documentar su robustez, se reentrenó el modelo base con una rejilla de 4 × 4 configuraciones de la tasa de aprendizaje y del número máximo de hojas, con parada temprana en todas:

| `learning_rate` \ `max_leaf_nodes` | 8 | 15 | **31** | 63 |
|---|---:|---:|---:|---:|
| 0,02 | 0,8286 | 0,8307 | 0,8312 | 0,8292 |
| **0,05** | 0,8306 | 0,8312 | **0,8303** | 0,8287 |
| 0,10 | 0,8306 | 0,8307 | 0,8295 | 0,8271 |
| 0,20 | 0,8303 | 0,8313 | 0,8272 | 0,8237 |

La precisión promedio en validación varía **menos de 0,008** en las dieciséis configuraciones, y la desplegada está a 0,001 de la mejor. La parada temprana compensa la tasa de aprendizaje: con 0,02 se necesitan entre 413 y 1500 iteraciones, y con 0,20, entre 96 y 447. Solo los árboles más grandes (63 hojas) con tasas altas empiezan a degradarse, la señal típica de sobreajuste. El rendimiento no depende de un ajuste fino de hiperparámetros: es una propiedad de los datos.

### 3.3. Parada temprana del modelo desplegado

El panel derecho muestra la pérdida logarítmica del *gradient boosting* desplegado sobre su entrenamiento y sobre la porción interna reservada del 10 %. La pérdida reservada alcanza su mínimo en la iteración 271 y el entrenamiento se detiene en la 321, tras 50 iteraciones sin mejora, como fija `n_iter_no_change = 50`. La pérdida final es 0,241 en entrenamiento y 0,282 en la porción reservada.

---

## 4. Residuos, calibración y subgrupos

**Figura:** `residual_analysis.png`

### 4.1. Residuos agrupados

En clasificación binaria, el residuo individual $y_i - \hat{p}_i$ solo toma dos valores y no es interpretable por sí solo. El diagnóstico estándar es el **gráfico de residuos agrupados** (Gelman y Hill, 2007): se ordenan las personas por probabilidad predicha, se agrupan en $\sqrt{N} = 98$ tramos de igual tamaño y se representa el residuo medio de cada tramo frente a su banda de ±2 errores típicos, $2\sqrt{\bar{p}(1-\bar{p})/n_b}$. Si el modelo está bien calibrado, alrededor del 95 % de los tramos deben quedar dentro de la banda.

En prueba, **el 5,1 % de los tramos** queda fuera, exactamente la proporción esperable por azar. Los residuos de desviación tienen media −0,08 y solo el 1,8 % supera el valor absoluto de 2.

### 4.2. Pendiente de calibración y descomposición de Murphy

La **pendiente y el intercepto de calibración** (Cox, 1958; Van Calster et al., 2019) se obtienen ajustando $\text{logit}\,P(y = 1) = a + b \cdot \text{logit}(\hat{p})$ sobre el conjunto de prueba. Un modelo perfectamente calibrado tiene $b = 1$ y $a = 0$; una pendiente menor que 1 indica sobreajuste (predicciones demasiado extremas). El ensamble obtiene **$b = 1{,}018$ y $a = −0{,}038$**: no es ni excesivamente confiado ni excesivamente prudente.

La **descomposición de Murphy** (1973) separa la puntuación de Brier en tres términos:

$$
\text{Brier} = \underbrace{\frac{1}{N}\sum_{k} n_k\,(\bar{p}_k - \bar{y}_k)^2}_{\text{fiabilidad (REL)}} \;-\; \underbrace{\frac{1}{N}\sum_{k} n_k\,(\bar{y}_k - \bar{y})^2}_{\text{resolución (RES)}} \;+\; \underbrace{\bar{y}(1 - \bar{y})}_{\text{incertidumbre (UNC)}}
$$

| Término | Valor (prueba) | Significado |
|---|---:|---|
| Incertidumbre | 0,1821 | Dificultad intrínseca del problema: el Brier de predecir siempre la tasa base |
| Resolución | 0,0958 | Cuánto separa el modelo a las personas de distinto destino |
| Fiabilidad | **0,0007** | Error de calibración: distancia entre probabilidad y frecuencia |
| **Brier** | **0,0863** | Índice de habilidad de Brier: $1 - 0{,}0863/0{,}1821 = 0{,}526$ |

La fiabilidad es unas 145 veces menor que la resolución: casi todo el error que queda no se debe a probabilidades mal calibradas, sino a la incertidumbre que el modelo no puede resolver con los rasgos disponibles. Es el mismo diagnóstico de la sección 5, obtenido por otra vía.

### 4.3. Calibración y precisión por subgrupos (prueba, censo de 1994)

| Subgrupo | n | Tasa real − predicción media | ROC-AUC | Precisión «movilidad» | Precisión «sin movilidad» |
|---|---:|---:|---:|---|---:|
| Mujeres | 3326 | +0,9 pp | 0,951 | **97,69 %** (n = 130) | 97,14 % |
| Hombres | 6432 | −1,1 pp | 0,913 | 95,36 % (n = 798) | 93,98 % |
| Raza blanca | 8353 | −0,3 pp | 0,930 | 95,86 % (n = 846) | 95,24 % |
| Raza negra | 916 | −0,0 pp | 0,943 | 97,44 % (n = 39) | 96,34 % |
| Asiático-Pacífico | 311 | −4,1 pp | 0,901 | 88,24 % (n = 34) | 95,77 % |
| 26–35 años | 2475 | −2,1 pp | 0,927 | 95,60 % (n = 159) | 95,90 % |
| 46–55 años | 1676 | −0,9 pp | 0,903 | 94,95 % (n = 277) | 92,67 % |
| Posgrado | 817 | +1,7 pp | 0,918 | 95,22 % (n = 356) | 89,23 % |

**Lectura.** En la población de 1994, la calibración por subgrupo se desvía como mucho unos pocos puntos y la discriminación es alta en todos. El dato más relevante para la sección 5 de `06_METHODOLOGICAL_DEFENSE.md` es el de las mujeres: en su propia población, el modelo acierta el **97,69 %** de sus decisiones «movilidad» para ellas, más que para los hombres. La caída al 59 % en la ACS 2018 no es, por tanto, un sesgo de origen del modelo, sino un cambio de la relación entre perfil e ingreso.

Hay dos limitaciones que deben declararse. La precisión del 95 % se garantiza para cada decisión sobre el conjunto de la población, no dentro de cada subgrupo. Por eso algunos subgrupos quedan por debajo: Asiático-Pacífico en «movilidad» (88,2 %, con solo 34 decisiones), los hombres en «sin movilidad» (93,98 %) y, sobre todo, los posgraduados en «sin movilidad» (89,2 %), cuya tasa de movilidad es muy alta. En un grupo con alta tasa base, la decisión «sin movilidad» es intrínsecamente más arriesgada. Exigir el 95 % por subgrupo requeriría bandas específicas por grupo, una decisión de equidad con implicaciones propias que queda fuera del alcance de este hito.

---

## 5. El techo de exactitud

**Figura:** `accuracy_ceiling.png`

### 5.1. Perfiles idénticos con destinos distintos

| Indicador (48 790 registros limpios, 13 rasgos) | Valor |
|---|---:|
| Perfiles distintos | 41 439 |
| Perfiles repetidos (dos o más personas) | 3746, con 11 097 personas |
| Perfiles repetidos con ambos destinos | **1029**, con 3864 personas (**35 %** de los repetidos) |
| Errores inevitables de cualquier regla determinista en los perfiles repetidos | 1378 (**12,4 %**) |
| Error mínimo de cualquier regla sobre 10 890 celdas narrativas (franja de edad, educación, ocupación, sexo, estado civil, franja de horas, capital) | **12,5 %** |

Los perfiles repetidos son, por construcción, los más comunes del censo, y no una muestra aleatoria. Aun así, el resultado es una cota directa e independiente de cualquier modelo: sobre ellos, ninguna función de los 13 rasgos puede bajar del 12,4 % de error.

### 5.2. El techo estimado por el modelo calibrado

Para un modelo calibrado, la exactitud esperada de la mejor regla que puede construirse con sus probabilidades es $\mathbb{E}[\max(\hat{p}, 1 - \hat{p})]$, y su complemento estima el error de Bayes:

$$
\hat{R}^* = \frac{1}{N}\sum_{i=1}^{N} \min(\hat{p}_i,\, 1 - \hat{p}_i) = 12{,}6\,\%
$$

La exactitud esperada resultante (87,41 %) coincide con la exactitud real del ensamble con umbral 0,5 (87,67 %): el modelo opera en su techo. Las dos estimaciones del error irreducible, la empírica de los perfiles idénticos (12,4 %) y la del modelo calibrado (12,6 %), concuerdan sin haberse calculado una a partir de la otra.

### 5.3. Curva riesgo-cobertura

La curva riesgo-cobertura (El-Yaniv y Wiener, 2010) ordena a las personas de mayor a menor confianza y representa la exactitud sobre la fracción decidida frente a esa fracción. Su área bajo la curva de riesgo (AURC) resume la calidad del modelo como clasificador selectivo: **0,0278**.

| Punto de la curva | Cobertura | Exactitud |
|---|---:|---:|
| Máxima cobertura con exactitud ≥ 95 % (regla simétrica) | 76,5 % | 95,0 % |
| **Banda desplegada (95 % exigido a cada decisión)** | **74,5 %** | **95,45 %** |
| Decidir sobre todas las personas | 100 % | 87,67 % |

La banda desplegada está justo por debajo de la frontera porque exige el 95 % a cada decisión por separado, no solo al conjunto. Solo el 52,1 % de las personas recibe una probabilidad con confianza de al menos el 95 %; la banda logra decidir sobre el 74,5 % porque los aciertos muy seguros compensan, dentro de cada decisión, a los menos seguros.

---

## 6. La garantía del 95 %: intervalos, probabilidad posterior y alternativa conservadora

**Figura:** `precision_guarantee.png`

### 6.1. Intervalos y probabilidad posterior

| Decisión (prueba) | Aciertos / casos | Estimación | Wilson | Clopper-Pearson | P(precisión ≥ 95 %) | Casos necesarios para límite inferior ≥ 95 % |
|---|---|---:|---|---|---:|---:|
| «Movilidad» | 888 / 928 | 95,69 % | 94,18 – 96,82 % | 94,18 – 96,90 % | 0,81 | 3332 |
| «Sin movilidad» | 6051 / 6342 | 95,41 % | **94,87** – 95,90 % | 94,87 – 95,91 % | 0,93 | 9930 |
| Todas las decididas | 6939 / 7270 | 95,45 % | 94,94 – 95,90 % | 94,94 – 95,91 % | **0,96** | 8354 |

La probabilidad posterior se calcula con una distribución *a priori* uniforme, $\theta \mid \text{datos} \sim \text{Beta}(s + 1,\ n - s + 1)$. Para la exactitud conjunta, la prueba binomial unilateral rechaza $H_0: \theta < 0{,}95$ con $p = 0{,}041$.

### 6.2. La banda conservadora

Si la Ecuación 11 exigiera que el **límite inferior** de Wilson, y no la estimación puntual, alcanzase el 95 % en validación, la banda sería **[0,21; 0,82]**. Su comportamiento en prueba:

| Política | Cobertura | Precisión «movilidad» | Precisión «sin movilidad» (Wilson) | Exactitud esperada del juego |
|---|---:|---:|---|---:|
| Banda desplegada [0,23; 0,78] | 74,50 % | 95,69 % | 95,41 % (94,87 – 95,90) | 85,22 % |
| Banda conservadora [0,21; 0,82] | 72,35 % | 96,65 % | 95,84 % (95,31 – 96,31) | 84,96 % |

La alternativa conservadora cumple el 95 % también en el límite inferior, a cambio de 2,15 puntos de cobertura y 0,26 de exactitud del juego. La discusión completa de por qué se mantiene la banda desplegada está en `06_METHODOLOGICAL_DEFENSE.md`, sección 4.

---

## 7. Síntesis

| Pregunta | Respuesta | Sección |
|---|---|---|
| ¿Es el ensamble el mejor modelo, y no por azar? | Sí: mejor en las siete métricas de validación, primero en 9 de 10 pliegues, Friedman $p = 1{,}2 \times 10^{-8}$ | 1, 2 |
| ¿Aporta algo el ensamble sobre el *gradient boosting* solo? | Igual discriminación y mejor Brier, con significación en el *bootstrap* pareado y en la prueba *t* corregida ($p = 0{,}031$) | 1.3, 2.3 |
| ¿Están sobreajustados el modelo o sus hiperparámetros? | No: la distancia entre entrenamiento y validación es de 0,03 en AP, la pendiente de calibración es 1,02 y 16 configuraciones varían menos de 0,008 | 3, 4.2 |
| ¿Son fiables las probabilidades? | Sí: el 5,1 % de los tramos de residuos queda fuera de ±2 errores típicos y la fiabilidad de Murphy es 0,0007 | 4 |
| ¿Existe un techo de exactitud? | Sí: en torno al 87,5 %, estimado de dos formas independientes | 5 |
| ¿Cuánta confianza merece la garantía del 95 %? | Probabilidad posterior de 0,96 para el conjunto de decisiones; la alternativa conservadora está calculada | 6 |

---

## Referencias

- Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika*, 45(3-4), 562-565.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- El-Yaniv, R. y Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research*, 11, 1605-1641.
- Gelman, A. y Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12(4), 595-600.
- Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L. y Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine*, 17, 230.

> **Estado del documento: borrador de referencia, no normativo.** La versión oficial del artículo se entrega por separado en PDF/Word. Este fichero se conserva bajo control de versiones como borrador de trabajo: registra la redacción, las ecuaciones y las cifras tal como estaban al cerrar el Hito 2 (octubre de 2026). Si una cifra de este borrador difiere de la de un artefacto del *pipeline*, prevalece el artefacto. La documentación técnica vigente está en `ai-service/docs/01_TRACEABILITY.md` a `06_METHODOLOGICAL_DEFENSE.md`.

# *Unfair Life*: un modelo calibrado de movilidad social con clasificación selectiva y su validación fuera de distribución en la *American Community Survey* de 2018

**Daniel Felipe Ordoñez Amaya**

**Nicolas Tamayo Melo**

**Juan David Rivera Chate**

Directora / Tutora: **Daniela Alarcon Sepulveda**

Electiva en Ingeniería Aplicada II (*English and Data Science Fundamentals*) · Programa de Ingeniería de Software · Facultad de Ingeniería

Universidad Surcolombiana · Neiva, Colombia

Octubre de 2026

---

## Resumen

*Unfair Life* es un videojuego narrativo que muestra cómo las circunstancias de origen condicionan la movilidad social. Este trabajo tuvo como objetivo construir, validar y desplegar un modelo de aprendizaje automático que estime la probabilidad de que una persona supere los 50 000 dólares anuales de 1994 a partir de su perfil censal, y medir si esa estimación conserva su validez en otra época. Se utilizaron 48 790 registros del censo *Adult* (1994). Se evaluó el balanceo con SMOTENC, que se rechazó porque degradaba la calibración. Se compararon cuatro paradigmas de clasificación y una referencia no paramétrica, y se desplegó un ensamble *Stacking* (*gradient boosting* y regresión logística combinados por un meta-modelo logístico) con una política de clasificación selectiva. En el conjunto de prueba, ambas decisiones deterministas superaron el 95 % de precisión (95,69 % y 95,41 %), con una cobertura del 74,50 % y una exactitud esperada del juego del 85,22 %. Al aplicar el modelo, sin reentrenarlo, a 298 686 personas de la ACS 2018 de California y Nueva York, la discriminación descendió (ROC-AUC de 0,833 y 0,802) y la precisión cayó de forma asimétrica. Una recalibración asimétrica del umbral restituyó la decisión «sin movilidad» al 95,34 % y al 95,81 %. La decisión «movilidad» no pudo recuperarse con ningún umbral: sus errores se concentraron en mujeres y en ocupaciones feminizadas. Se concluye que la abstención es la respuesta adecuada ante esa deriva y que la desigualdad de género aparece como su principal fuente.

**Palabras claves:** movilidad social, clasificación selectiva, calibración de probabilidades, *concept drift*, aprendizaje por ensambles, sesgo de género.

---

## *Unfair Life*: A calibrated social-mobility model with selective classification and its out-of-distribution validation on the 2018 American Community Survey

## Abstract

*Unfair Life* is a narrative video game that shows how circumstances of origin condition social mobility. The aim of this work was to build, validate and deploy a machine-learning model that estimates the probability that a person earns more than 50,000 dollars a year in 1994 terms from their census profile, and to measure whether that estimate remains valid in another era. A total of 48,790 records from the *Adult* census (1994) were used. Class balancing with SMOTENC was evaluated and rejected because it degraded calibration. Four classification paradigms and a non-parametric baseline were compared, and a *Stacking* ensemble (gradient boosting and logistic regression combined by a logistic meta-model) was deployed with a selective-classification policy. On the test set, both deterministic decisions exceeded 95% precision (95.69% and 95.41%), with a coverage of 74.50% and an expected game accuracy of 85.22%. When the model was applied, without retraining, to 298,686 people from the 2018 ACS for California and New York, discrimination decreased (ROC-AUC of 0.833 and 0.802) and precision fell asymmetrically. An asymmetric threshold recalibration restored the "no mobility" decision to 95.34% and 95.81%. The "mobility" decision could not be recovered with any threshold: its errors were concentrated among women and in feminized occupations. It is concluded that abstention is the appropriate response to this drift and that gender inequality emerges as its main source.

**Keywords:** social mobility, selective classification, probability calibration, concept drift, ensemble learning, gender bias.

---

## Introducción

### Antecedentes

La movilidad social intergeneracional, entendida como la posibilidad de que una persona alcance una posición económica distinta de la de su origen, es uno de los indicadores más estudiados de la equidad de una sociedad. En Estados Unidos, Chetty et al. (2014) mostraron que la probabilidad de ascender desde el quintil inferior de ingresos varía de forma considerable según el lugar de nacimiento, lo que indica que el origen condiciona el destino en una medida que no se explica solo por el esfuerzo individual. La literatura sobre desigualdad salarial añade que esa condición no es neutral respecto al género: aunque la brecha se redujo durante décadas, persiste y se explica en buena parte por la segregación ocupacional (Blau & Kahn, 2017; Goldin, 2014), y las ocupaciones tienden a pagar menos a medida que se feminizan (Levanon et al., 2009).

### Justificación

*Unfair Life* traslada esa evidencia a una experiencia interactiva. El jugador encarna a un personaje con un perfil socioeconómico concreto (edad, educación, ocupación, país de origen, estado civil, sexo, entre otros rasgos) y toma decisiones narrativas. Cuando una decisión pone en juego su ascenso social, el desenlace no lo escribe el guionista: lo determina un modelo estadístico entrenado con datos censales reales. El propósito sociológico del juego es que el jugador experimente que dos personas con el mismo mérito pueden obtener resultados distintos por circunstancias que no eligieron. Para que ese mensaje sea legítimo, el modelo debe cumplir tres requisitos: discriminar bien, producir probabilidades calibradas (porque el juego las convierte en una moneda ponderada) y ser explicable rasgo por rasgo.

### Estado del arte de la inteligencia artificial en la predicción socioeconómica

La predicción del ingreso a partir de variables censales es un problema de referencia en aprendizaje automático desde la publicación del conjunto *Adult* (Becker & Kohavi, 1996; Kohavi, 1996). La regresión logística (Cox, 1958) ha sido históricamente el modelo de referencia en ciencias sociales por la interpretabilidad de sus coeficientes. En datos tabulares de tamaño medio, la evidencia comparativa reciente indica que los ensambles de árboles, en especial el *gradient boosting* (Friedman, 2001; Chen & Guestrin, 2016; Ke et al., 2017), siguen superando a las redes neuronales profundas (Grinsztajn et al., 2022). La combinación de modelos con sesgos inductivos distintos mediante *stacked generalization* (Wolpert, 1992; Breiman, 1996) permite aprovechar su complementariedad.

Tres líneas de investigación complementan esa base. La primera es la calibración: una probabilidad solo es útil para la toma de decisiones si coincide con la frecuencia observada, propiedad que los modelos de margen y de *bagging* no garantizan (Niculescu-Mizil & Caruana, 2005; Platt, 1999). La segunda es la clasificación con opción de rechazo (Chow, 1970), formulada modernamente como clasificación selectiva (Geifman & El-Yaniv, 2017), que permite a un modelo abstenerse cuando su confianza no alcanza un objetivo de error. La tercera es la explicabilidad basada en valores de Shapley (Lundberg & Lee, 2017; Lundberg et al., 2020).

Dos advertencias del estado del arte orientan este trabajo. Por un lado, la predictibilidad de los resultados vitales es limitada: en un esfuerzo colaborativo con cientos de equipos, ningún modelo predijo con precisión alta los desenlaces de familias estadounidenses a partir de datos longitudinales ricos (Salganik et al., 2020). Por otro, los modelos entrenados con *Adult* reflejan una economía de 1994. Ding et al. (2021) propusieron sustituirlo por derivados de la *American Community Survey* (ACS) precisamente porque su distribución ya no representa la población actual. La adaptación a cambios de distribución (*dataset shift*) y, en particular, a cambios en la relación entre las variables y la etiqueta (*concept drift*) es un área consolidada (Gama et al., 2014; Moreno-Torres et al., 2012; Quiñonero-Candela et al., 2009). Ninguno de los trabajos revisados combina en un mismo sistema una política selectiva con garantía de precisión, un uso estocástico de la probabilidad calibrada y una validación temporal fuera de distribución, que es el aporte de este trabajo.

### Objetivos

**Objetivo general.** Construir, validar y desplegar un modelo calibrado que estime la probabilidad de movilidad social de un personaje a partir de su perfil censal, con una precisión mínima del 95 % en sus decisiones absolutas, y evaluar su validez en una población de otra época.

**Objetivos específicos.**

1. Preparar el censo *Adult* de 1994 y evaluar empíricamente si el balanceo de clases con SMOTENC mejora el modelo.
2. Comparar cuatro paradigmas de clasificación tabular y una referencia no paramétrica, y construir un ensamble que preserve la calibración.
3. Definir una política de clasificación selectiva que garantice al menos un 95 % de precisión en ambas decisiones deterministas.
4. Integrar el modelo en una arquitectura de producción (CMS, BFF y servicio de inferencia) con trazabilidad completa entre lo documentado y lo desplegado.
5. Medir el rendimiento del modelo congelado sobre la ACS 2018 de California y Nueva York, cuantificar la deriva e identificar los subgrupos en que se concentra.

---

## Metodología de análisis y recolección de datos

### Fuentes de datos

Se utilizaron dos fuentes secundarias de microdatos censales de Estados Unidos, ambas de acceso público.

**Censo *Adult* (1994).** Conjunto derivado de la *Current Population Survey* de 1994 (Becker & Kohavi, 1996), con 48 842 registros de personas y 14 atributos. La variable objetivo es binaria: $y = 1$ si el ingreso anual supera los 50 000 USD. Tras eliminar 52 duplicados exactos y conservar los valores ausentes como una categoría explícita, el conjunto final constó de **48 790 registros**, un 39,4 % por encima del mínimo de 35 000 que exige el criterio del curso. La tasa de movilidad fue del 23,94 %.

**ACS PUMS 2018 (California y Nueva York).** Microdatos de un año de la *American Community Survey* (U.S. Census Bureau, 2019), obtenidos con el paquete `folktables` (Ding et al., 2021). Se aplicó el filtro de la tarea *ACSIncome* (edad > 16, ingreso > 100, horas trabajadas > 0 y peso muestral ≥ 1). De 378 817 registros brutos de California y 196 967 de Nueva York quedaron **195 665** y **103 021** personas, respectivamente: **298 686** registros de validación externa, más de seis veces el tamaño del conjunto de entrenamiento.

La Tabla 1 resume ambas fuentes.

**Tabla 1**

*Fuentes de datos utilizadas*

---

| Fuente | Año | Ámbito | Registros brutos | Registros evaluados | Tasa de movilidad | Uso |
|:--|:--:|:--|--:|--:|--:|:--|
| *Adult* (CPS) | 1994 | Nacional | 48 842 | 48 790 | 23,94 % | Entrenamiento, validación y prueba |
| ACS PUMS | 2018 | California | 378 817 | 195 665 | 23,47 % | Prueba fuera de distribución |
| ACS PUMS | 2018 | Nueva York | 196 967 | 103 021 | 21,25 % | Prueba fuera de distribución |

---

*Nota.* La tasa de movilidad es la proporción de personas con ingreso anual superior a 50 000 USD de 1994. En la ACS, el ingreso se ajustó con el factor `ADJINC` y se deflactó a dólares de 1994 antes de calcular la etiqueta; el umbral equivale a 84 719 USD de 2018. Elaboración propia a partir de `cleaning_report.json` y `ood_results_*.json`.

### Normalización temporal y armonización de variables

Para que ambas fuentes plantearan la misma pregunta económica, toda cantidad monetaria se expresó en dólares de 1994 mediante el índice de precios al consumidor (IPC-U, serie CUUR0000SA0) de la Oficina de Estadísticas Laborales (U.S. Bureau of Labor Statistics, 2025), con el deflactor $\text{CPI}(1994)/\text{CPI}(2018) = 0{,}5902$. La tabla de IPC es la misma que utiliza el servidor del juego, leída en tiempo de ejecución, de modo que ambas implementaciones no pueden divergir. Los rasgos de la ACS se recodificaron a las 13 variables y grafías del censo de 1994: la educación (SCHL), la clase de trabajador (COW), el estado civil (MAR), la relación con el hogar, la raza (RAC1P), el país de nacimiento (POBP) y la ocupación (OCCP), esta última agrupada por rangos de código en los 14 grupos ocupacionales de 1994. La ACS no registra ganancias ni pérdidas de capital, que se fijaron en cero, el valor de más del 90 % de los registros de *Adult*.

### Partición y control de la fuga de información

El conjunto *Adult* se dividió de forma estratificada, con la semilla 42, en entrenamiento (29 274 registros), validación (9758) y prueba (9758). La desviación máxima de la tasa de movilidad entre los tres subconjuntos fue de 0,00004. La validación se reservó para seleccionar modelos y fijar la política de decisión; la prueba se utilizó una sola vez, para el informe final.

### Evaluación del balanceo de clases

La distribución natural presenta un desequilibrio moderado (76,1 % frente a 23,9 %). Se aplicó **SMOTENC** (Chawla et al., 2002), la variante de SMOTE para datos con rasgos numéricos y categóricos, a una copia del conjunto de entrenamiento. Se generaron 15 256 registros sintéticos, hasta una proporción exacta de 50/50. Los conjuntos de validación y prueba nunca se balancearon.

En lugar de adoptar el balanceo por defecto, se realizó un **estudio de ablación**: las cinco familias de modelos se entrenaron con la distribución natural y con la balanceada, y se evaluaron sobre el mismo conjunto de validación compuesto solo por personas reales (Tabla 2).

**Tabla 2**

*Estudio de ablación del balanceo: distribución natural frente a SMOTENC (conjunto de validación)*

---

| Familia | Precisión promedio (natural) | Precisión promedio (SMOTE) | ECE (natural) | ECE (SMOTE) |
|:--|--:|--:|--:|--:|
| HistGradientBoosting | **0,830** | 0,797 | **0,010** | 0,078 |
| Regresión logística | **0,768** | 0,759 | **0,008** | 0,112 |
| Bosque aleatorio | **0,766** | 0,716 | **0,019** | 0,066 |
| SVM RBF calibrada | **0,757** | 0,736 | **0,023** | 0,105 |
| k vecinos más cercanos | **0,631** | 0,597 | **0,044** | 0,121 |

---

*Nota.* La precisión promedio es el área bajo la curva precisión-exhaustividad. ECE = error de calibración esperado con diez tramos (Ecuación 6). En negrita, el mejor valor de cada par. Elaboración propia a partir de `model_comparison.json`.

En las cinco familias, sin excepción, el balanceo redujo la capacidad discriminativa y multiplicó el error de calibración, en algunos casos por más de diez. La predicción media de los modelos entrenados con SMOTE ascendió a 0,330, cuando la tasa real es 0,239. Este efecto es el esperado teóricamente: reequilibrar las clases altera la probabilidad *a priori* con la que aprende el modelo y desplaza sus probabilidades (Saerens et al., 2002). Como el juego convierte la probabilidad en una moneda ponderada, una probabilidad inflada otorgaría movilidad a personajes que en la realidad no la tendrían, y la simulación se volvería sistemáticamente optimista, en contra del propósito del proyecto. Por esta evidencia, **el modelo desplegado se entrenó con la distribución natural**. El criterio de balanceo se abordó, por tanto, de forma explícita: se aplicó, se midió y se rechazó con fundamento empírico.

### Modelos de aprendizaje automático

Se seleccionó un representante consolidado de cada uno de los cuatro paradigmas de clasificación tabular, más una referencia no paramétrica:

1. **Regresión logística** (paradigma lineal probabilístico): modelo convexo, bien calibrado por construcción e interpretable mediante razones de probabilidades.
2. **HistGradientBoosting** (paradigma de *boosting*): árboles construidos en secuencia que corrigen el gradiente de la pérdida logarítmica, con discretización en histogramas y parada temprana.
3. **Bosque aleatorio** (paradigma de *bagging*): promedio de 300 árboles descorrelacionados mediante muestras *bootstrap* y subconjuntos aleatorios de variables.
4. **Máquina de vectores de soporte con kernel RBF** (paradigma de márgenes), aproximada con el método de Nyström (Williams & Seeger, 2001) y calibrada con el método de Platt (1999).
5. **k vecinos más cercanos** (Cover & Hart, 1967), como cota inferior de referencia.

Los candidatos se ordenaron por precisión promedio, con la puntuación de Brier como criterio de desempate. Ante el dilema entre el modelo que mejor discriminaba (*gradient boosting*) y el que mejor calibraba (regresión logística), se construyó un **ensamble *Stacking*** de dos capas: los dos modelos base producen sus *log-odds* fuera de pliegue (validación cruzada estratificada de cinco pliegues) y un **meta-modelo** de regresión logística los combina. Un estudio de ablación comprobó que añadir el bosque aleatorio o los k vecinos no mejoraba ninguna métrica, rompía la explicabilidad exacta y multiplicaba por catorce el tamaño del modelo, por lo que se eligió la primera capa más simple.

### Política de decisión y validación fuera de distribución

La política de **clasificación selectiva** se fijó solo con el conjunto de validación: se buscaron los umbrales inferior y superior que maximizan la cobertura con la condición de que ambas decisiones deterministas alcancen al menos un 95 % de precisión, cada una con un mínimo de 30 casos (Ecuación 8). Dentro de la banda, el desenlace se resuelve con una moneda ponderada por la probabilidad del modelo.

La validación fuera de distribución aplicó el modelo y sus artefactos de preprocesamiento a la ACS 2018 **sin reentrenarlos ni recalibrarlos**. Para estudiar la adaptación, cada muestra estatal se dividió al 50 % de forma estratificada (semilla 42). En la mitad de adaptación se buscó una nueva banda con la misma regla del evaluador original, y ambas políticas (la congelada de 1994 y la adaptada) se evaluaron en la mitad reservada, que no participó en la búsqueda. Si ningún umbral superior alcanzaba el 95 %, se aplicaba una **abstención asimétrica**: se recalibraba solo el umbral inferior y toda probabilidad superior se resolvía con la moneda. Finalmente, los errores de la decisión «movilidad» se desglosaron por sexo y por grupo ocupacional.

---

## Resultados

### Modelo matemático general

El modelo desplegado estima $\hat{p}(x) = P(Y = 1 \mid X = x)$, donde $x$ es el perfil de 13 rasgos censales codificado en $d = 106$ columnas. La Tabla 3 resume la notación.

**Tabla 3**

*Notación del modelo matemático*

---

| Símbolo | Significado |
|:--|:--|
| $x$, $y$ | Perfil censal del personaje y etiqueta real ($y = 1$: ingreso > 50 000 USD de 1994) |
| $\hat{p}(x)$ | Probabilidad de movilidad estimada |
| $\sigma(t)$ | Función sigmoide, $\sigma(t) = 1/(1 + e^{-t})$ |
| $F(x)$ | Puntuación en escala de *log-odds*, $F = \log\bigl(\hat{p}/(1-\hat{p})\bigr)$ |
| $\ell$, $u$ | Umbrales inferior y superior de la banda de incertidumbre |
| $TP, TN, FP, FN$ | Verdaderos y falsos positivos y negativos |

---

*Nota.* Las ecuaciones se generan desde los artefactos del *pipeline* (`artifacts/latex/`) y su desarrollo completo se encuentra en `02_MATHEMATICS.md`.

**Preparación de los datos.** SMOTENC genera un individuo sintético interpolando entre un caso real de la clase minoritaria y uno de sus $k = 5$ vecinos:

$$
x_{\text{new}} = x_i + \lambda\,(x_{z_i} - x_i), \qquad \lambda \sim \mathcal{U}(0, 1),\quad x_{z_i} \in \mathrm{kNN}_5(x_i) \tag{1}
$$

**Modelos base.** La regresión logística modela los *log-odds* como una función lineal de los rasgos estandarizados $z_j$, con penalización $L_2$:

$$
F_{\text{lr}}(x) = \beta_0 + \sum_{j=1}^{d} \beta_j z_j, \qquad
\hat{\beta} = \arg\min_{\beta}\ -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\Bigr] + \frac{1}{2Cn}\lVert\beta\rVert_2^2 \tag{2}
$$

El *gradient boosting* parte de la tasa media en escala logit y añade $M = 321$ árboles con tasa de aprendizaje $\nu = 0{,}05$, cada uno ajustado al gradiente $g_i$ y a la curvatura $h_i$ de la pérdida logarítmica:

$$
F_M(x) = F_0 + \nu \sum_{m=1}^{M} f_m(x), \qquad
f_m = \arg\min_f \sum_{i=1}^{n}\Bigl(g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\Bigr),\quad g_i = \hat{p}_{m-1,i} - y_i,\ h_i = \hat{p}_{m-1,i}(1-\hat{p}_{m-1,i}) \tag{3}
$$

El bosque aleatorio promedia $T = 300$ árboles que dividen cada nodo por la máxima reducción de la impureza de Gini, y la SVM calibrada transforma su distancia a la frontera $f(x)$ en probabilidad mediante una sigmoide de Platt:

$$
\hat{p}_{\text{rf}}(x) = \frac{1}{T}\sum_{t=1}^{T}\hat{p}_t(x), \quad G(S) = 1 - \sum_{c}\pi_c(S)^2; \qquad
\hat{p}_{\text{svm}}(x) = \frac{1}{1 + \exp\bigl(A f(x) + B\bigr)} \tag{4}
$$

**Ensamble *Stacking* (ecuación general del modelo desplegado).** El meta-modelo combina linealmente los *log-odds* fuera de pliegue de los dos modelos base:

$$
F(x) = b + w_{\text{hgb}}\,F_{\text{hgb}}(x) + w_{\text{lr}}\,F_{\text{lr}}(x), \qquad \hat{p}(x) = \sigma\bigl(F(x)\bigr), \qquad
b = 0{,}0114,\ w_{\text{hgb}} = 0{,}8848,\ w_{\text{lr}} = 0{,}1013 \tag{5}
$$

La combinación se realiza en escala logit y no sobre probabilidades porque aplicar una regresión logística a probabilidades ya calibradas compone dos sigmoides y distorsiona la calibración. Que $w_{\text{hgb}} + w_{\text{lr}} \approx 0{,}99$ y $b \approx 0$ indica que la segunda capa promedia dos modelos ya calibrados sin necesidad de corregirlos.

**Evaluación.** La precisión, la puntuación de Brier y el error de calibración esperado (ECE) se definen como:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2, \qquad
\text{ECE} = \sum_{b=1}^{10}\frac{|S_b|}{N}\,\bigl|\,\overline{\hat{p}}_{S_b} - \overline{y}_{S_b}\bigr| \tag{6}
$$

Cada proporción se acompaña de su intervalo de confianza de Wilson (1927), fiable cerca de 0 y de 1:

$$
\text{CI}_{95\%}(\hat{\pi}) = \frac{\hat{\pi} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{\pi}(1-\hat{\pi})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1{,}96 \tag{7}
$$

**Clasificación selectiva.** Los umbrales maximizan la cobertura sobre el conjunto de validación $V$ con una restricción de precisión doble:

$$
(\ell^*, u^*) = \arg\max_{\ell < u}\ \frac{|\{i \in V : \hat{p}_i \le \ell \lor \hat{p}_i \ge u\}|}{|V|}
\quad \text{s.a.} \quad P(y = 1 \mid \hat{p} \ge u) \ge 0{,}95,\ \ P(y = 0 \mid \hat{p} \le \ell) \ge 0{,}95 \tag{8}
$$

con resultado $(\ell^*, u^*) = (0{,}23;\ 0{,}78)$. La decisión del juego y su exactitud esperada son:

$$
\text{decisión}(x) = \begin{cases}
\text{movilidad} & \hat{p}(x) \ge u^* \\
\text{sin movilidad} & \hat{p}(x) \le \ell^* \\
\text{movilidad si } U < \hat{p}(x),\ U \sim \mathcal{U}(0,1) & \ell^* < \hat{p}(x) < u^*
\end{cases} \tag{9}
$$

$$
\mathbb{E}[\text{Accuracy}_{\text{juego}}] = \frac{1}{N}\Biggl[\sum_{i \notin \text{banda}} \mathbb{1}(\text{decisión}_i = y_i) + \sum_{i \in \text{banda}}\Bigl(y_i\hat{p}_i + (1-y_i)(1-\hat{p}_i)\Bigr)\Biggr] \tag{10}
$$

**Explicabilidad.** El valor de Shapley reparte la predicción entre los rasgos; por su propiedad de linealidad, el ensamble se explica de forma exacta combinando las explicaciones de sus modelos base con los pesos de la Ecuación 5:

$$
\phi_j(x) = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\bigl[v_x(S \cup \{j\}) - v_x(S)\bigr], \qquad
\phi_j(F) = w_{\text{hgb}}\,\phi_j(F_{\text{hgb}}) + w_{\text{lr}}\,\phi_j(F_{\text{lr}}) \tag{11}
$$

**Aplicación en el juego.** Las cantidades monetarias del año narrado $t$ se deflactan a dólares de 1994, y los modificadores ocultos $T_d$ de cada decisión narrativa $d$ transforman el perfil base $x_0$ de la historia antes de consultarlo al modelo:

$$
x^{(1994)} = x^{(t)}\,\frac{\text{CPI}(1994)}{\text{CPI}(t)}, \qquad
x'_j = T_d(x_0)_j = \begin{cases}\min\bigl(\max(x_{0,j} + \delta_j,\ lo_j),\ hi_j\bigr) & j \text{ numérico} \\ c_j & j \text{ categórico}\end{cases} \tag{12}
$$

El peso de una decisión narrativa queda así medido como el desplazamiento en *log-odds* que produce, $F(T_d(x_0)) - F(x_0)$.

**Recalibración asimétrica ante la deriva.** Cuando el problema bilateral (8) no tiene solución en la nueva población $A$, se recalibra solo el umbral inferior:

$$
\ell^*_{\text{2018}} = \arg\max_{\ell}\ \frac{|\{i \in A : \hat{p}_i \le \ell\}|}{|A|} \quad \text{s.a.} \quad P(y = 0 \mid \hat{p} \le \ell) \ge 0{,}95,\ \ |\{i : \hat{p}_i \le \ell\}| \ge 30 \tag{13}
$$

y toda probabilidad $\hat{p} > \ell^*_{2018}$ se resuelve con la moneda ponderada.

### Trazabilidad entre lo documentado y lo desplegado

Cada etapa del *pipeline* produce un artefacto versionado que hereda del anterior, de modo que cualquier cifra de este artículo puede rastrearse hasta su origen (Tabla 4). El servicio de inferencia no reentrena ni reconstruye el modelo: carga los mismos ficheros generados por el *pipeline* y lee la banda de decisión de `traceability_report.json` en cada arranque. Antes de aceptar peticiones verifica que el modelo, el imputador, el codificador y el esquema de columnas sean coherentes entre sí; si no lo son, el servicio no arranca. Toda la cadena es reproducible con la semilla 42.

**Tabla 4**

*Cadena de evidencia del modelo*

---

| Artefacto | Etapa | Pregunta que responde |
|:--|:--|:--|
| `cleaning_report.json` | 01 · Limpieza | ¿Con qué datos se partió y qué se corrigió? |
| `balancing_report.json` | 02 · Partición y balanceo | ¿Cómo se separaron los datos sin fuga de información? |
| `model_comparison.json` | 03 · Entrenamiento | ¿Qué modelos se compararon y cómo se construyó el ensamble? |
| `traceability_report.json` | 04 · Evaluación | ¿Cuál es el rendimiento final y con qué política opera en producción? |
| `ood_results_CA.json`, `ood_results_NY.json` | 05 · Prueba fuera de distribución | ¿Cómo se comporta el modelo congelado en 2018 y qué política restituye el 95 %? |

---

*Nota.* La trazabilidad se verificó también en juego real: las probabilidades registradas por el servidor durante la historia «Turno de Noche» coincidieron hasta la tercera cifra decimal con las calculadas fuera de línea con los mismos artefactos (`01_TRACEABILITY.md`, sección 3.5).

### Comparación de modelos

La Tabla 5 presenta el rendimiento en validación de los candidatos y del ensamble, todos entrenados con la distribución natural.

**Tabla 5**

*Rendimiento de los modelos en el conjunto de validación*

---

| Modelo | Paradigma | Precisión promedio | ROC-AUC | Brier | ECE |
|:--|:--|--:|--:|--:|--:|
| Regresión logística | Lineal | 0,768 | 0,906 | 0,102 | 0,0080 |
| HistGradientBoosting | *Boosting* | 0,830 | 0,928 | 0,088 | 0,0096 |
| Bosque aleatorio | *Bagging* | 0,766 | 0,903 | 0,103 | 0,0193 |
| SVM RBF calibrada | Márgenes | 0,757 | 0,901 | 0,105 | 0,0234 |
| k vecinos más cercanos | No paramétrico | 0,631 | 0,847 | 0,129 | 0,0442 |
| **Stacking (HGB + RL → RL)** | **Ensamble** | **0,830** | — | **0,0879** | **0,0073** |

---

*Nota.* El ensamble conserva la discriminación del mejor modelo individual y reduce su ECE de 0,0096 a 0,0073 (−24 %). Elaboración propia a partir de `model_comparison.json`.

### Precisión superior al 95 % en las decisiones deterministas

Sobre el conjunto de prueba, que el modelo no vio en ninguna fase, la política selectiva decide de forma determinista el 74,50 % de los casos (Tabla 6).

**Tabla 6**

*Precisión del ensamble en el conjunto de prueba (1994) con la banda [0,23; 0,78]*

---

| Decisión | Casos | Precisión | IC 95 % (Wilson) |
|:--|--:|--:|--:|
| «Movilidad» ($\hat{p} \ge 0{,}78$) | 928 | **95,69 %** | 94,18 % – 96,82 % |
| «Sin movilidad» ($\hat{p} \le 0{,}23$) | 6342 | **95,41 %** | 94,87 % – 95,90 % |
| Exactitud sobre los casos decididos | 7270 | **95,45 %** | 94,94 % – 95,90 % |
| Exactitud esperada del juego (Ecuación 10) | 9758 | 85,22 % | 84,74 % – 85,68 %ᵃ |

---

*Nota.* ROC-AUC en prueba = 0,9313; Brier = 0,0863. Un clasificador convencional con umbral 0,5 obtendría una exactitud del 87,67 %; la diferencia con el 85,22 % es el coste deliberado de no fingir certeza dentro de la banda. Elaboración propia a partir de `traceability_report.json`.
ᵃ Rango del 95 % de 2000 partidas simuladas con la semilla 42, centrado en el valor analítico.

Ambas decisiones absolutas superan el 95 % y el *bootstrap* de 2000 remuestreos sitúa la exactitud sobre los casos decididos entre el 94,98 % y el 95,92 %. La Figura 1 muestra cómo la banda separa las zonas deterministas de la zona de suerte sistémica.

**Figura 1**

*Distribución de las probabilidades del conjunto de prueba y banda de incertidumbre*

![Histograma de probabilidades con la banda de incertidumbre](../training-pipeline/artifacts/figures/decision_policy.png)

*Nota.* Las probabilidades menores o iguales a 0,23 producen «sin movilidad» y las mayores o iguales a 0,78, «movilidad»; entre ambas, el desenlace se resuelve con una moneda ponderada por $\hat{p}$. Elaboración propia.

La Figura 2 resume la explicabilidad del ensamble. Los rasgos más influyentes son el estado civil, la edad, las ganancias de capital y la educación; `sex_Female` es la sexta columna más influyente y su efecto sobre la probabilidad es negativo, lo que evidencia que el modelo absorbió la desigualdad de género presente en el censo de 1994.

**Figura 2**

*Importancia media absoluta de cada rasgo según los valores SHAP del ensamble*

![Importancia SHAP por rasgo](../training-pipeline/artifacts/figures/shap_trait_importance.png)

*Nota.* Valores calculados de forma exacta sobre 2000 personas del conjunto de prueba; la discrepancia máxima entre la suma de contribuciones y la salida del ensamble fue de $7{,}6 \times 10^{-8}$. Elaboración propia.

### Funcionalidades del sistema

El modelo se integra en una arquitectura de tres servicios: un CMS (Strapi) que almacena las historias, un *Backend for Frontend* (NestJS) que orquesta el juego y un servicio de inferencia (FastAPI) que sirve el modelo. La Tabla 7 documenta las doce funcionalidades implementadas.

**Tabla 7**

*Funcionalidades de inteligencia artificial implementadas*

---

| N.º | Funcionalidad | Componente | Descripción |
|:--:|:--|:--|:--|
| 1 | CMS dinámico de historias | CMS y BFF | Las escenas, decisiones, perfiles base y modificadores se editan en Strapi y se validan en el BFF al cargarse, sin desplegar código |
| 2 | Inyección de contexto dinámico | BFF | Los modificadores ocultos de cada decisión transforman el perfil base, $x' = T_d(x_0)$, con proyección al dominio de entrenamiento (Ecuación 12) |
| 3 | Perfil base en el servidor | BFF | El personaje lo declara la historia y no el cliente, lo que impide enviar un perfil más favorable |
| 4 | Deflación automática por IPC | BFF | Las cantidades del año narrado se convierten a dólares de 1994 con la serie oficial y una proyección del 2,5 % anual |
| 5 | Banda de indecisión (clasificación selectiva) | Inferencia y BFF | Tres zonas de decisión con umbrales leídos de `traceability_report.json` (Ecuación 8) |
| 6 | Suerte sistémica (moneda ponderada) | BFF | Dentro de la banda, movilidad si $U < \hat{p}$ con un generador criptográfico (Ecuación 9) |
| 7 | Simulación determinista y reproducible | *Pipeline* | Semilla 42, 2000 remuestreos *bootstrap* y 2000 partidas simuladas que confirman la exactitud analítica |
| 8 | Integración BFF con predicción asegurada | Inferencia y BFF | `POST /predict` solo responde a la cabecera `X-BFF-Token` con comparación en tiempo constante |
| 9 | Validación estricta de esquemas | Inferencia y BFF | DTO en NestJS, Pydantic estricto y reglas de dominio: campos no declarados o fuera de rango se rechazan |
| 10 | Arranque *fail-fast* y monitoreo | Inferencia | El servicio no arranca con artefactos incoherentes; `GET /health` informa el modelo y la banda servidos |
| 11 | Auditoría de cada tirada | BFF | Cada prueba de movilidad registra modificadores, perfil enviado, probabilidad, zona, tirada y resultado |
| 12 | Explicabilidad serializada (SHAP) | *Pipeline* | Valores de Shapley exactos del ensamble, persistidos como tablas y figuras versionadas |

---

*Nota.* Se añaden como medidas operativas el control de orígenes cruzados (CORS) restringido al BFF y un aviso silencioso del cliente al servicio de inferencia al cargar el juego, que mitiga el arranque en frío del alojamiento. Detalle de cada funcionalidad en `03_IMPLEMENTATION_AND_BALANCE.md`, sección 3.

### Validación fuera de distribución (ACS 2018)

La Tabla 8 compara el rendimiento del modelo congelado en las tres poblaciones.

**Tabla 8**

*Rendimiento del modelo congelado dentro y fuera de distribución*

---

| Métrica | Prueba 1994 | California 2018 | Nueva York 2018 |
|:--|--:|--:|--:|
| ROC-AUC | 0,9313 | 0,8327 | 0,8022 |
| Puntuación de Brier | 0,0863 | 0,1401 | 0,1504 |
| ECE | — | 0,0663 | 0,0967 |
| Tasa real de movilidad | 23,94 % | 23,47 % | 21,25 % |
| Predicción media | — | 25,85 % | 27,28 % |
| Cobertura determinista | 74,50 % | 65,60 % | 63,47 % |
| Precisión «movilidad» | 95,69 % | 70,52 % | 63,47 % |
| Precisión «sin movilidad» | 95,41 % | 91,42 % | 91,38 % |
| Exactitud esperada del juego | 85,22 % | 77,21 % | 75,56 % |

---

*Nota.* Modelo, preprocesamiento y banda [0,23; 0,78] idénticos a los de producción, sin reentrenamiento. California: *n* = 195 665; Nueva York: *n* = 103 021. Elaboración propia a partir de `ood_results_CA.json` y `ood_results_NY.json`.

La tasa base apenas cambió (−0,5 puntos en California y −2,7 en Nueva York), mientras que el modelo sobrestima la movilidad en 2,4 y 6,0 puntos. La caída de la precisión es asimétrica: la decisión «movilidad» pierde entre 25 y 32 puntos, y «sin movilidad», solo 4.

La búsqueda bilateral de una nueva banda no tuvo solución: el mejor umbral superior de la rejilla alcanzó una precisión del 80,12 % en California y del 79,55 % en Nueva York, ambos en $\hat{p} \ge 0{,}92$. Se aplicó entonces la recalibración asimétrica de la Ecuación 13 (Tabla 9).

**Tabla 9**

*Recalibración asimétrica de la decisión «sin movilidad» (mitad reservada)*

---

| Estado | Política | Umbral | Cobertura | Precisión «sin movilidad» | IC 95 % (Wilson) | Exactitud del juego |
|:--|:--|:--:|--:|--:|--:|--:|
| California | Banda de 1994 | $\hat{p} \le 0{,}23$ | 65,57 % | 91,53 % | 91,30 % – 91,75 % | 77,27 % |
| California | Recalibración asimétrica | $\hat{p} \le 0{,}08$ | 45,01 % | **95,34 %** | 95,14 % – 95,53 % | 75,67 % |
| Nueva York | Banda de 1994 | $\hat{p} \le 0{,}23$ | 63,32 % | 91,52 % | 91,19 % – 91,83 % | 75,53 % |
| Nueva York | Recalibración asimétrica | $\hat{p} \le 0{,}07$ | 40,09 % | **95,81 %** | 95,53 % – 96,08 % | 73,89 % |

---

*Nota.* Los umbrales se estimaron en una mitad estratificada de cada muestra (semilla 42) con la regla de `04_evaluator.choose_band` y se evaluaron en la otra mitad. La recalibración asimétrica elimina la decisión determinista «movilidad»: toda probabilidad superior al umbral se resuelve con la moneda ponderada. Elaboración propia a partir de `ood_results_*.json`.

En ambos estados, el límite inferior del intervalo de confianza supera el 95 %. Para identificar el origen del fallo en la decisión «movilidad», sus resultados se desglosaron por sexo y por grupo ocupacional (Tabla 10).

**Tabla 10**

*Precisión de la decisión «movilidad» ($\hat{p} \ge 0{,}78$) por sexo y grupo ocupacional, ACS 2018*

---

| Subgrupo | Mujeres en el grupo (CA / NY) | Precisión «movilidad» CA | Precisión «movilidad» NY |
|:--|:--:|--:|--:|
| Mujeres | — | 59,32 % (*n* = 6199) | 50,90 % (*n* = 3782) |
| Hombres | — | 83,17 % (*n* = 5490) | 80,44 % (*n* = 2802) |
| *Adm-clerical* | 71,3 % / 73,1 % | 44,4 % | 41,7 % |
| *Tech-support* | 56,7 % / 62,8 % | 49,6 % | 57,7 % |
| *Prof-specialty* | 52,6 % / 59,0 % | 65,1 % | 56,8 % |
| *Exec-managerial* | 46,1 % / 46,4 % | 80,8 % | 75,3 % |

---

*Nota.* Las mujeres representan el 47,2 % (CA) y el 49,4 % (NY) de la población evaluada, pero el 73,2 % y el 77,2 % de los falsos positivos de «movilidad». La precisión de «sin movilidad» es prácticamente igual por sexo: 91,67 % frente a 91,18 % en CA y 91,69 % frente a 91,05 % en NY. *Prof-specialty* concentra el 59 % (CA) y el 62 % (NY) de las decisiones «movilidad». Desglose calculado con los mismos registros, recodificación y artefactos de `05_ood_testing.py`; aún no se persiste en `ood_results_*.json`.

---

## Discusión y conclusiones

### Sentido general de los resultados

El modelo cumple el criterio de precisión superior al 95 % allí donde toma decisiones absolutas, sobre datos no vistos y con intervalos de confianza que lo respaldan. Fuera de la población con la que se entrenó, conserva una capacidad de ordenar razonable, pero pierde la calibración y, con ella, la garantía de precisión. Esa pérdida no es uniforme: afecta con severidad a la decisión «movilidad» y de forma leve a la decisión «sin movilidad».

### El fenómeno de *concept drift*

En la taxonomía de los cambios de distribución (Moreno-Torres et al., 2012; Quiñonero-Candela et al., 2009) se distinguen el cambio en la tasa base, $P(Y)$, el cambio en la distribución de los rasgos, $P(X)$, y el cambio en la relación entre ambos, $P(Y \mid X)$, denominado *concept drift* (Gama et al., 2014). Los resultados apuntan principalmente al tercero. La tasa base de California en 2018 es casi idéntica a la de 1994 y, aun así, el modelo sobrestima la movilidad y la precisión de su cola alta cae 25 puntos. Si el problema fuera solo de tasa base, bastaría con corregir las probabilidades *a priori* (Saerens et al., 2002) o desplazar los umbrales. El hecho de que ningún umbral superior supere el 80 % de precisión indica algo más profundo: el modelo coloca en sus puntuaciones más altas a personas que en 2018 no superan el umbral. Es un error de **orden**, no de **escala**, y un umbral solo puede corregir errores de escala.

### Por qué la recalibración asimétrica salvó la decisión «sin movilidad»

En la cola baja, la precisión del modelo alcanza el 99,3 % con el umbral $\hat{p} \le 0{,}01$, de modo que existe un umbral (0,08 en California y 0,07 en Nueva York) que restituye la garantía del 95 %. La explicación es sustantiva: las predicciones bajas descansan en rasgos cuya relación con el ingreso es estable en el tiempo, como la juventud, la jornada parcial o la escolaridad baja. Una persona de 19 años que trabaja 20 horas semanales no superaba el umbral en 1994 y tampoco lo supera en 2018. La recalibración tiene un coste en cobertura (el modelo decide por sí solo en el 40–45 % de los casos), pero apenas reduce la exactitud del juego (−1,6 puntos en ambos estados), porque los casos que salen de la zona determinista pasan a una moneda que sigue estando razonablemente ponderada.

### Por qué falló la decisión «movilidad»: el sesgo hacia las profesiones feminizadas

El desglose de la Tabla 10 sitúa el error en un subgrupo identificable. La precisión de «movilidad» es entre 24 y 30 puntos menor para las mujeres que para los hombres, y ellas acumulan tres de cada cuatro falsos positivos pese a ser la mitad de la población. El error crece con la feminización del grupo ocupacional: es máximo en *Adm-clerical* (más del 70 % de mujeres) y mínimo en *Exec-managerial*. *Prof-specialty*, que concentra cerca del 60 % de las decisiones «movilidad», agrupa en 2018 la docencia, la enfermería, el trabajo social y otras profesiones de alta escolaridad con fuerte presencia femenina.

La interpretación que proponemos es coherente con la literatura sobre la devaluación del trabajo feminizado. Levanon et al. (2009) encontraron, con datos censales de 1950 a 2000, que las ocupaciones reducen su salario a medida que aumenta la proporción de mujeres, incluso controlando por cualificación. Blau y Kahn (2017) atribuyen una parte sustancial de la brecha salarial residual a la ocupación y al sector, y Goldin (2014) la vincula con la penalización de las ocupaciones que no remuneran de forma lineal las horas y la flexibilidad. El modelo de 1994 aprendió que un título universitario, una ocupación profesional y una jornada completa conducen a superar el umbral, porque en ese censo esos rasgos correspondían mayoritariamente a hombres. En 2018, las mujeres alcanzaron esos mismos rasgos de capital humano, pero lo hicieron en buena medida en profesiones cuyo salario no acompaña a su nivel educativo. El modelo les asigna puntuaciones altas por su currículum; el mercado les paga por debajo del umbral. En cambio, el lado bajo apenas distingue por sexo, lo que explica la asimetría del fallo.

Este hallazgo no debilita el marco teórico del juego, sino que lo refuerza. La explicabilidad ya mostraba que el modelo absorbió la penalización por ser mujer del censo de 1994. La prueba de 2018 muestra que, fuera de ese censo, la desigualdad de género reaparece por otra vía: ya no como un coeficiente negativo, sino como la promesa de una movilidad que no llega.

### Limitaciones

Los resultados deben leerse con cautela por cinco razones. Primero, la recodificación de la ocupación por rangos de código agrupa profesiones heterogéneas, por lo que el análisis no puede separar, por ejemplo, la enfermería de la medicina dentro de *Prof-specialty*. Segundo, la ACS no registra ganancias de capital, que se fijaron en cero; esto reduce la información disponible en la cola alta. Tercero, *Adult* es una muestra nacional y la ACS se evaluó en dos estados de ingresos altos, de modo que parte de la deriva puede ser geográfica y no solo temporal. Cuarto, el desglose por sexo y ocupación es observacional y no permite afirmar causalidad; es una explicación coherente con la evidencia, no una demostración. Quinto, ese desglose aún no se persiste en los artefactos versionados del *pipeline*.

### Conclusiones

1. El balanceo con SMOTENC se evaluó y se rechazó con evidencia: en las cinco familias de modelos redujo la discriminación y multiplicó el error de calibración, una propiedad indispensable para un sistema que convierte probabilidades en decisiones estocásticas.
2. Un ensamble *Stacking* de *gradient boosting* y regresión logística, combinado en escala logit, conservó la discriminación del mejor modelo individual y mejoró su calibración, y una política de clasificación selectiva garantizó más del 95 % de precisión en ambas decisiones deterministas sobre datos de 1994 no vistos.
3. En la ACS 2018 de California y Nueva York, el modelo congelado sufrió una deriva que afectó sobre todo a su decisión «movilidad». Una recalibración asimétrica del umbral inferior restituyó la decisión «sin movilidad» por encima del 95 %; la decisión «movilidad» no pudo recuperarse con ningún umbral y la abstención fue la respuesta coherente con el criterio de precisión.
4. Los errores de la decisión «movilidad» se concentraron en mujeres y en ocupaciones feminizadas, un patrón compatible con la literatura sobre la devaluación salarial del trabajo feminizado.

Estas conclusiones se limitan al modelo, a los datos y a las dos poblaciones estudiadas. No permiten generalizar sobre la predictibilidad de la movilidad social en otros contextos ni sustituyen un análisis causal de la brecha de género. Su aporte es más modesto: muestran que un modelo que se abstiene cuando no sabe es más honesto que uno que decide siempre, y que medir dónde falla un modelo puede revelar la misma desigualdad que pretende representar. Como trabajo futuro se propone persistir el desglose por subgrupos en el *pipeline*, utilizar una recodificación ocupacional más fina y evaluar el reentrenamiento con datos contemporáneos.

---

## Referencias

Becker, B., & Kohavi, R. (1996). *Adult* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Blau, F. D., & Kahn, L. M. (2017). The gender wage gap: Extent, trends, and explanations. *Journal of Economic Literature, 55*(3), 789–865. https://doi.org/10.1257/jel.20160995

Breiman, L. (1996). Stacked regressions. *Machine Learning, 24*(1), 49–64. https://doi.org/10.1007/BF00117832

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Chetty, R., Hendren, N., Kline, P., & Saez, E. (2014). Where is the land of opportunity? The geography of intergenerational mobility in the United States. *The Quarterly Journal of Economics, 129*(4), 1553–1623. https://doi.org/10.1093/qje/qju022

Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory, 16*(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological), 20*(2), 215–232. https://doi.org/10.1111/j.2517-6161.1958.tb00292.x

Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring Adult: New datasets for fair machine learning. En *Advances in Neural Information Processing Systems* (Vol. 34, pp. 6478–6490). Curran Associates.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), Artículo 44. https://doi.org/10.1145/2523813

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. En *Advances in Neural Information Processing Systems* (Vol. 30). Curran Associates.

Goldin, C. (2014). A grand gender convergence: Its last chapter. *American Economic Review, 104*(4), 1091–1119. https://doi.org/10.1257/aer.104.4.1091

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? En *Advances in Neural Information Processing Systems* (Vol. 35, Datasets and Benchmarks Track). Curran Associates.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. En *Advances in Neural Information Processing Systems* (Vol. 30). Curran Associates.

Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid. En *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining* (pp. 202–207). AAAI Press.

Levanon, A., England, P., & Allison, P. (2009). Occupational feminization and pay: Assessing causal dynamics using 1950–2000 U.S. Census data. *Social Forces, 88*(2), 865–891. https://doi.org/10.1353/sof.0.0264

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems* (Vol. 30). Curran Associates.

Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition, 45*(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. En *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.

Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset shift in machine learning*. MIT Press.

Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation, 14*(1), 21–41. https://doi.org/10.1162/089976602753284446

Salganik, M. J., Lundberg, I., Kindel, A. T., Ahearn, C. E., Al-Ghoneim, K., Almaatouq, A., Altschul, D. M., Brand, J. E., Carnegie, N. B., Compton, R. J., Datta, D., Davidson, T., Filippova, A., Gilroy, C., Goode, B. J., Jahani, E., Kashyap, R., Kirchner, A., McKay, S., … McLanahan, S. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences, 117*(15), 8398–8403. https://doi.org/10.1073/pnas.1915006117

U.S. Bureau of Labor Statistics. (2025). *Consumer Price Index for All Urban Consumers (CPI-U): U.S. city average, all items (CUUR0000SA0)* [Conjunto de datos]. https://www.bls.gov/cpi/

U.S. Census Bureau. (2019). *American Community Survey 2018 1-year Public Use Microdata Sample (PUMS)* [Conjunto de datos]. https://www.census.gov/programs-surveys/acs/microdata.html

Williams, C. K. I., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. En *Advances in Neural Information Processing Systems* (Vol. 13, pp. 682–688). MIT Press.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1

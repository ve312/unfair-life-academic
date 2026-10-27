# Modelo Matemático y Análisis Descriptivo

> **Ámbito.** Este documento expone el fundamento matemático del modelo de movilidad social de *Unfair Life* y el análisis descriptivo que respalda su validez. Diez de las catorce ecuaciones proceden de los ficheros de `ai-service/training-pipeline/artifacts/latex/`. Las otras cuatro (Ecuaciones 4 a 7) formalizan los modelos base y el meta-learner con la configuración exacta de `src/03_trainer.py`. Cada figura procede de `ai-service/training-pipeline/artifacts/figures/`. Las ecuaciones y las Figuras 1 a 11 las genera el notebook `notebooks/01_math_and_eda.ipynb`; las Figuras 12 a 17, que visualizan la banda de indecisión, la suerte sistémica, el ajuste por inflación y el *concept drift*, las genera `src/generate_extra_plots.py`, que persiste cada cifra que dibuja en `artifacts/extra_figures_report.json`. Ninguno de los dos reentrena nada: ambos leen los artefactos del *pipeline*. La procedencia de cada cifra se documenta en [`01_TRACEABILITY.md`](01_TRACEABILITY.md).
>
> El objetivo es justificar el criterio de evaluación según el cual **el modelo matemático cuenta con las ecuaciones explicadas, gráficos descriptivos y alta claridad**.

---

## 1. Notación común

| Símbolo | Significado |
|---|---|
| $x$ | Perfil de un personaje: sus rasgos censales (edad, educación, ocupación, etc.) |
| $y \in \{0, 1\}$ | Etiqueta real: $1$ si la persona supera los 50 000 USD anuales de 1994 (movilidad), $0$ en caso contrario |
| $\hat{p}(x)$ | Probabilidad de movilidad estimada por el modelo |
| $N$, $n$ | Número de casos del conjunto evaluado o de entrenamiento |
| $TP, TN, FP, FN$ | Verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos |
| $\sigma(t)$ | Función sigmoide, $\sigma(t) = 1 / (1 + e^{-t})$: convierte *log-odds* en probabilidad |
| $F(x)$ | Puntuación de un modelo en escala de *log-odds*, $F = \log\bigl(\hat{p} / (1 - \hat{p})\bigr)$ |
| $z_j$ | Rasgo $j$ estandarizado con la media y la desviación típica del entrenamiento |
| $d$ | Número de variables del modelo tras la codificación *one-hot*: $d = 106$ |

---

## 2. Las catorce ecuaciones del modelo

Las ecuaciones se presentan en el orden en que intervienen en el ciclo de vida del modelo, en cuatro bloques: la preparación de los datos (2.1), los modelos de aprendizaje, primero uno a uno y después combinados en el ensamble (2.2), la evaluación y la explicación (2.3), y su aplicación en el juego (2.4).

### 2.1. Preparación de los datos

#### Ecuación 1 — Sobremuestreo sintético SMOTE

*Fichero de origen: `eq_smote.tex`*

$$
x_{\text{new}} = x_i + \lambda\,\bigl(x_{z_i} - x_i\bigr), \qquad \lambda \sim \mathcal{U}(0, 1),\quad x_{z_i} \in \mathrm{kNN}_k(x_i),\ k = 5
$$

**Interpretación.** SMOTE crea un individuo sintético de la clase minoritaria (personas con movilidad) sin duplicar registros existentes. Para ello toma un individuo real $x_i$, elige al azar uno de sus cinco vecinos más cercanos de la misma clase, $x_{z_i}$, y sitúa el nuevo punto en algún lugar del segmento que los une. El factor aleatorio $\lambda$ decide a qué distancia de cada uno queda. En los rasgos categóricos, la variante SMOTENC utilizada en el proyecto no interpola: asigna la categoría mayoritaria entre los vecinos.

**Uso en el proyecto.** Se aplicó únicamente a una copia del conjunto de entrenamiento y generó 15 256 registros sintéticos, hasta lograr una proporción exacta de 50/50 entre clases. Los conjuntos de validación y prueba nunca se balancearon.

**Conclusión y regla derivada.** El balanceo se evaluó de forma empírica y **se rechazó** para el modelo desplegado. En las cinco familias de algoritmos, entrenar con datos sintéticos redujo la precisión promedio en validación, con pérdidas de entre 0,008 y 0,050. También infló las probabilidades: la predicción media pasó de 0,241 a 0,330, cuando la tasa real de movilidad es 0,239. Como el juego convierte la probabilidad en una moneda ponderada (Ecuación 13), una probabilidad inflada otorgaría movilidad a personajes que en la realidad no la tendrían. **Regla:** el modelo en producción se entrena con la distribución natural de los datos.

### 2.2. Modelos de aprendizaje: modelos base, meta-learner y ensamble

Antes de la ecuación general del ensamble (Ecuación 8), cada modelo se presenta por separado, con su formulación matemática, su configuración exacta en `03_trainer.py` y su resultado en validación. El estado del arte que justifica esta selección se desarrolla en [`04_MODEL_SELECTION.md`](04_MODEL_SELECTION.md).

| Ecuación | Modelo | Paradigma | Papel en la solución | Precisión promedio (validación) | ECE (validación) |
|---|---|---|---|---:|---:|
| 2 | Regresión logística | Lineal probabilístico | Capa 1 del ensamble desplegado | 0,768 | 0,0080 |
| 3 | HistGradientBoosting | *Boosting* de árboles | Capa 1 del ensamble desplegado (dominante) | 0,830 | 0,0096 |
| 4 | Bosque aleatorio | *Bagging* de árboles | Capa 1 evaluada en la ablación, descartada | 0,766 | 0,0193 |
| 5 | SVM RBF calibrada | Márgenes y kernels | Candidato comparado, descartado | 0,757 | 0,0234 |
| 6 | k vecinos más cercanos | No paramétrico | Referencia, peso nulo en la ablación | 0,631 | 0,0442 |
| 7 | Regresión logística (meta) | Combinación lineal de *log-odds* | Capa 2 del ensamble desplegado | — | — |
| 8 | **Stacking (HGB + LR → LR)** | **Ensamble de dos capas** | **Modelo desplegado** | **0,830** | **0,0073** |

#### Ecuación 2 — Regresión logística (modelo base 1)

*Configuración: `03_trainer.py`, `StandardScaler` + `LogisticRegression(max_iter=3000)`.*

$$
\begin{aligned}
z_j &= \frac{x_j - \mu_j}{s_j}, \qquad
F_{\text{lr}}(x) = \beta_0 + \sum_{j=1}^{d} \beta_j\, z_j, \qquad
\hat{p}_{\text{lr}}(x) = \sigma\bigl(F_{\text{lr}}(x)\bigr), \qquad d = 106 \\
\hat{\beta} &= \arg\min_{\beta}\ -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log \hat{p}_{\text{lr}}(x_i) + (1-y_i)\log\bigl(1-\hat{p}_{\text{lr}}(x_i)\bigr)\Bigr] + \frac{1}{2Cn}\lVert \beta \rVert_2^2, \qquad C = 1
\end{aligned}
$$

**Interpretación.** La regresión logística modela el logaritmo de la razón de probabilidades (*log-odds*) como una función **lineal** de los rasgos. Antes, cada rasgo se estandariza ($z_j$): se le resta la media de entrenamiento $\mu_j$ y se divide por su desviación típica $s_j$, de modo que todos los coeficientes quedan en la misma escala. Cada coeficiente $\beta_j$ es el cambio en *log-odds* por cada desviación típica del rasgo, y $e^{\beta_j}$ es la razón de probabilidades (*odds ratio*) correspondiente. Los coeficientes se estiman por máxima verosimilitud penalizada: la pérdida logarítmica más un término $L_2$ que limita su magnitud. El problema es **convexo**, por lo que tiene un único óptimo global.

**Uso en el proyecto.** En validación obtuvo una precisión promedio de 0,768 y un ROC-AUC de 0,906. Su error de calibración esperado (0,0080) fue **el más bajo de todos los candidatos individuales**, por debajo incluso del *gradient boosting* (0,0096): minimiza una regla de puntuación propia y tiene muy poca varianza. Forma parte de la primera capa del ensamble desplegado (Ecuación 8), con un peso de 0,101.

**Conclusión y regla derivada.** La distancia entre su precisión promedio (0,768) y la del *gradient boosting* (0,830) cuantifica cuánto valen las interacciones no lineales que un modelo lineal no puede representar. A cambio, aporta al ensamble una señal suave, bien calibrada y explicable en forma cerrada (Ecuación 12). **Regla:** todo modelo complejo se compara contra una referencia lineal, y solo se justifica si la supera.

#### Ecuación 3 — Gradient Boosting por histogramas (modelo base 2)

*Fichero de origen: `eq_gradient_boosting.tex`*

$$
\begin{aligned}
F_0(x) &= \log\frac{\bar{y}}{1-\bar{y}}, \qquad
F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} f_m(x), \qquad \nu = 0.05,\ M = 321 \\
\hat{p}(x) &= \sigma\bigl(F_M(x)\bigr) = \frac{1}{1 + e^{-F_M(x)}} \\
\mathcal{L} &= -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log \hat{p}(x_i) + (1-y_i)\log\bigl(1-\hat{p}(x_i)\bigr)\Bigr] + \lambda \lVert w \rVert_2^2,
\qquad \lambda = 1.0 \\
f_m &= \arg\min_{f} \sum_{i=1}^{n} \Bigl(g_i\, f(x_i) + \tfrac{1}{2} h_i\, f(x_i)^2\Bigr), \quad
g_i = \hat{p}_{m-1}(x_i) - y_i,\ \ h_i = \hat{p}_{m-1}(x_i)\bigl(1-\hat{p}_{m-1}(x_i)\bigr)
\end{aligned}
$$

**Interpretación.** El *gradient boosting* es un conjunto de árboles de decisión construido de forma aditiva:

- **Punto de partida.** El modelo arranca de $F_0$, que equivale a la tasa media de movilidad expresada en escala logarítmica de razón de probabilidades (*log-odds*).
- **Corrección progresiva.** En cada una de las $M = 321$ iteraciones se añade un árbol pequeño, $f_m$, que corrige los errores acumulados hasta ese momento. Para ello usa el gradiente $g_i$ (cuánto se equivoca la predicción actual) y la curvatura $h_i$ (cuán segura es) de la función de pérdida.
- **Prudencia.** La tasa de aprendizaje $\nu = 0{,}05$ hace que cada árbol aporte solo una fracción pequeña de su corrección, y el término de regularización $\lambda = 1{,}0$ penaliza los árboles excesivamente complejos. El número de iteraciones no se fijó de antemano: lo determinó la parada temprana, que detuvo el entrenamiento cuando dejó de mejorar sobre una porción reservada de los datos.
- **Probabilidad final.** La función sigmoide $\sigma$ transforma la puntuación acumulada en una probabilidad entre 0 y 1.

**Uso en el proyecto.** Fue el mejor de los once candidatos individuales, con una precisión promedio en validación de 0,830 frente a 0,768 del mejor modelo lineal. Hoy es el modelo base dominante del ensamble desplegado (Ecuación 8), que le asigna un peso de 0,885 frente a 0,101 de la regresión logística.

**Conclusión y regla derivada.** Minimizar la pérdida logarítmica, y no la tasa de aciertos, produce probabilidades **calibradas**: cuando el modelo estima un 60 %, aproximadamente el 60 % de esas personas tiene movilidad real. **Regla:** la salida del modelo se interpreta y se utiliza como una probabilidad genuina, no solo como una puntuación para ordenar casos. Esto es lo que legitima la moneda ponderada del juego, y lo que el ensamble de la Ecuación 8 se cuida de preservar.

#### Ecuación 4 — Bosque aleatorio (modelo base 3, evaluado en la ablación del ensamble)

*Configuración: `RandomForestClassifier(n_estimators=300, max_depth=50, min_samples_split=5, max_features="sqrt")`.*

$$
\begin{aligned}
\hat{p}_{\text{rf}}(x) &= \frac{1}{T}\sum_{t=1}^{T} \hat{p}_t(x), \qquad \hat{p}_t(x) = \frac{1}{\lvert L_t(x) \rvert}\sum_{i \in L_t(x)} y_i, \qquad T = 300 \\
G(S) &= 1 - \sum_{c \in \{0, 1\}} \pi_c(S)^2, \qquad
\Delta G = G(S) - \frac{\lvert S_L \rvert}{\lvert S \rvert}\, G(S_L) - \frac{\lvert S_R \rvert}{\lvert S \rvert}\, G(S_R), \qquad m = \lfloor \sqrt{d} \rfloor = 10
\end{aligned}
$$

**Interpretación.** El bosque aleatorio promedia las estimaciones de $T = 300$ árboles de decisión. Cada árbol $t$ se entrena con una muestra *bootstrap* (extraída con reemplazo) del conjunto de entrenamiento. En cada nodo elige, entre solo $m = 10$ rasgos sorteados, la división que más reduce la **impureza de Gini** $G$, donde $\pi_c(S)$ es la proporción de la clase $c$ en el nodo $S$. La estimación de un árbol es la frecuencia de movilidad en la hoja $L_t(x)$ donde cae la persona. El doble azar (de filas y de rasgos) descorrelaciona los árboles. Por eso promediarlos reduce la varianza: si cada árbol tiene varianza $\sigma^2$ y la correlación entre árboles es $\rho$, la varianza del promedio es $\rho\sigma^2 + (1-\rho)\,\sigma^2 / T$.

**Uso en el proyecto.** Obtuvo una precisión promedio de 0,766 y un ECE de 0,019 en validación. Sus árboles muy profundos (hojas de un solo caso) producen probabilidades extremas y peor calibradas. En la ablación del ensamble se probó como tercer modelo base, en una variante con hojas de al menos 10 casos: recibió un peso de solo 0,078 y no mejoró ninguna métrica de forma apreciable. Además rompía la explicabilidad exacta, porque el logit de un promedio de probabilidades no es aditivo en los rasgos, y multiplicaba el tamaño del modelo de 0,43 MB a 5,96 MB.

**Conclusión y regla derivada.** En este problema, reducir la varianza promediando árboles (*bagging*) rinde menos que reducir el sesgo corrigiendo errores en secuencia (*boosting*, Ecuación 3). **Regla:** un modelo entra al ensamble solo si su información no está ya contenida en los demás.

#### Ecuación 5 — Máquina de vectores de soporte con kernel RBF y calibración de Platt (modelo candidato 4)

*Configuración: `Nystroem(kernel="rbf", n_components=800)` + `LinearSVC(C=0.5)`, calibrado con `CalibratedClassifierCV(method="sigmoid", cv=3)`.*

$$
\begin{aligned}
K(x, x') &= \exp\bigl(-\gamma\, \lVert z - z' \rVert_2^2\bigr), \qquad
\tilde{\varphi}(x) = W^{-1/2}\bigl[K(x, l_1), \dots, K(x, l_m)\bigr]^{\top}, \qquad m = 800,\ \gamma = 1/d \\
(w, b) &= \arg\min_{w,\,b}\ \frac{1}{2}\lVert w \rVert_2^2 + C\sum_{i=1}^{n} \max\Bigl(0,\ 1 - \tilde{y}_i\bigl(w^{\top}\tilde{\varphi}(x_i) + b\bigr)\Bigr)^2, \qquad \tilde{y}_i \in \{-1, +1\},\ C = 0.5 \\
\hat{p}_{\text{svm}}(x) &= \frac{1}{1 + \exp\bigl(A\, f(x) + B\bigr)}, \qquad f(x) = w^{\top}\tilde{\varphi}(x) + b
\end{aligned}
$$

**Interpretación.** La SVM busca la frontera que separa las dos clases con el **mayor margen** posible. El kernel RBF $K$ mide la similitud entre dos personas y permite fronteras curvas. Calcularlo entre todos los pares de filas tiene un coste cuadrático en $n$, así que se usa la **aproximación de Nyström**: con $m = 800$ personas de referencia $l_1, \dots, l_m$ y la matriz $W$ de sus similitudes entre sí, se construye una representación $\tilde{\varphi}(x)$ sobre la que una SVM lineal equivale aproximadamente a la SVM con kernel. La SVM minimiza la pérdida de bisagra al cuadrado más un término que maximiza el margen. Su salida $f(x)$ es una distancia a la frontera, no una probabilidad. Por eso se aplica la **calibración de Platt**: una sigmoide cuyos parámetros $A$ y $B$ se ajustan por máxima verosimilitud sobre pliegues no vistos (tres en total), promediando los tres modelos resultantes.

**Uso en el proyecto.** Obtuvo una precisión promedio de 0,757 y un ECE de 0,023 en validación, y fue con diferencia el candidato más lento (17,5 s frente a 3,9 s del *gradient boosting*).

**Conclusión y regla derivada.** Los modelos de margen necesitan una calibración explícita para producir probabilidades, y aun calibrada la SVM quedó por debajo del *gradient boosting* en discriminación y en calibración, con un coste de entrenamiento muy superior. **Regla:** la probabilidad debe ser nativa del modelo o calibrarse de forma verificable; nunca se usa una puntuación como si fuera una probabilidad.

#### Ecuación 6 — k vecinos más cercanos (referencia no paramétrica)

*Configuración: `StandardScaler` + `KNeighborsClassifier(n_neighbors=25, weights="distance")`.*

$$
\hat{p}_{\text{knn}}(x) = \frac{\displaystyle\sum_{i \in N_k(x)} w_i\, y_i}{\displaystyle\sum_{i \in N_k(x)} w_i}, \qquad w_i = \frac{1}{\lVert z - z_i \rVert_2}, \qquad k = 25
$$

**Interpretación.** k-NN no aprende parámetros: estima la probabilidad de movilidad de una persona como la frecuencia de movilidad entre sus $k = 25$ vecinos más parecidos del conjunto de entrenamiento, $N_k(x)$. Los vecinos más cercanos pesan más ($w_i$ es inversamente proporcional a la distancia). Es la formalización más directa de la intuición «dime con quién andas».

**Uso en el proyecto.** Fue el candidato más débil: precisión promedio de 0,631 y ECE de 0,044 en validación. En la ablación del ensamble, el meta-learner le asignó un peso prácticamente nulo (−0,0005).

**Conclusión y regla derivada.** En un espacio de 106 dimensiones dominado por indicadores binarios, las distancias tienden a concentrarse (la *maldición de la dimensionalidad*) y la noción de vecindad pierde poder informativo. k-NN se conserva como **cota inferior de referencia**: cualquier modelo que no la supere no aprende nada útil. **Regla:** la elección del modelo debe respetar la geometría de los datos.

#### Ecuación 7 — Meta-learner: regresión logística sobre log-odds fuera de pliegue (capa 2)

*Configuración: `StackingClassifier(cv=StratifiedKFold(5), stack_method="decision_function", final_estimator=LogisticRegression())`.*

$$
\begin{aligned}
u(x_i) &= \Bigl(F^{(-k(i))}_{\text{hgb}}(x_i),\ F^{(-k(i))}_{\text{lr}}(x_i)\Bigr), \qquad k(i) \in \{1, \dots, 5\} \\
\hat{p}_{\text{meta}}(u) &= \sigma\bigl(b + w^{\top} u\bigr), \qquad
(b, w) = \arg\min_{b,\,w}\ -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log \hat{p}_{\text{meta}}(u_i) + (1-y_i)\log\bigl(1-\hat{p}_{\text{meta}}(u_i)\bigr)\Bigr] + \frac{1}{2Cn}\lVert w \rVert_2^2
\end{aligned}
$$

**Interpretación.** El meta-learner aprende cuánto confiar en cada modelo base. Sus variables de entrada $u(x_i)$ son los *log-odds* que producen los dos modelos base para la persona $i$. Cada uno se calcula con un modelo $F^{(-k(i))}$ entrenado **sin** el pliegue $k(i)$ al que pertenece esa persona: son predicciones fuera de pliegue (*out-of-fold*). Si el meta-learner se entrenara con predicciones de modelos que ya vieron esas filas, premiaría su sobreajuste. Sobre esas entradas ajusta una regresión logística (Ecuación 2) de solo tres parámetros: un intercepto $b$ y un peso por modelo base.

**Uso en el proyecto.** Los parámetros aprendidos son $b = 0{,}0114$, $w_{\text{hgb}} = 0{,}8848$ y $w_{\text{lr}} = 0{,}1013$.

**Conclusión y regla derivada.** Se eligió una regresión logística como meta-learner por cuatro razones:

- **Calibración.** Minimiza una regla de puntuación propia, así que su salida es una probabilidad calibrada.
- **Estabilidad.** Con tres parámetros, no puede sobreajustar.
- **Interpretabilidad.** Sus pesos dicen directamente cuánto aporta cada modelo.
- **Explicabilidad.** Al combinar linealmente *log-odds* aditivos, conserva la explicabilidad exacta del ensamble (Ecuación 12).

Que los pesos sumen casi uno y el intercepto sea casi cero indica que los modelos base ya estaban calibrados: el meta-learner los promedia, no los corrige. **Regla:** la capa de combinación debe ser más simple que los modelos que combina.

#### Ecuación 8 — Ensamble Stacking: ecuación general (modelo desplegado)

*Fichero de origen: `eq_stacking.tex`*

$$
\begin{aligned}
F(x) &= b + w_{\text{hgb}}\, F_{\text{hgb}}(x) + w_{\text{lr}}\, F_{\text{lr}}(x), \qquad
F_{\text{lr}}(x) = \beta_0 + \sum_{j} \beta_j\, \frac{x_j - \mu_j}{s_j}, \qquad
\hat{p}(x) = \sigma\bigl(F(x)\bigr) \\
(b, w) &= \arg\min_{b,\,w}\ -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log \sigma\bigl(F^{\text{oof}}(x_i)\bigr)
+ (1-y_i)\log\Bigl(1-\sigma\bigl(F^{\text{oof}}(x_i)\bigr)\Bigr)\Bigr] + \frac{1}{2C}\lVert w \rVert_2^2,
\qquad K = 5 \text{ pliegues} \\
b &= 0.0114, \qquad w_{\text{hgb}} = 0.8848, \qquad w_{\text{lr}} = 0.1013
\end{aligned}
$$

**Interpretación.** El modelo desplegado es un ensamble de dos capas (*stacked generalization*):

- **Primera capa.** Dos modelos base producen sus log-odds: el *gradient boosting* de la Ecuación 3, $F_{\text{hgb}}$, y la regresión logística de la Ecuación 2 sobre los rasgos estandarizados, $F_{\text{lr}}$. Sus sesgos inductivos son complementarios: el primero es flexible y no lineal, y captura interacciones y umbrales; el segundo es rígido y lineal, con poca varianza. Sus errores solo están parcialmente correlacionados.
- **Segunda capa.** El *meta-learner* de la Ecuación 7, una regresión logística, aprende a combinar ambos log-odds en una única puntuación $F(x)$, que la sigmoide convierte en probabilidad.
- **Predicciones fuera de pliegue.** Los pesos $(b, w)$ se ajustan sobre predicciones $F^{\text{oof}}$ obtenidas con validación cruzada estratificada de cinco pliegues. Cada modelo base predice filas que no vio durante su entrenamiento, lo que impide que la segunda capa premie el sobreajuste de la primera.

**Por qué en escala logit.** La combinación se realiza sobre log-odds y no sobre probabilidades. Durante el diseño se comprobó que una regresión logística aplicada a probabilidades ya calibradas compone dos sigmoides y distorsiona la calibración: su ECE fue aproximadamente tres veces el del *gradient boosting* solo. Combinar log-odds equivale a un *pool* logarítmico de opiniones y conserva la calibración.

**Uso en el proyecto.** Es el modelo que sirve la API de inferencia (`best_model.joblib`). Los pesos aprendidos son reveladores: $w_{\text{hgb}} + w_{\text{lr}} \approx 0{,}99$ y $b \approx 0$. La segunda capa no necesitó recalibrar nada; solo promedió dos modelos que ya estaban calibrados, con pesos de 0,885 para el *gradient boosting* y 0,101 para la regresión logística.

**Conclusión y regla derivada.** Frente al mejor modelo individual, el ensamble mantiene la precisión promedio en validación (0,830) y reduce el error de calibración esperado de 0,0096 a **0,0073**, un 24 % menos. La mejora en discriminación es marginal; la mejora en calibración es pequeña pero real, y es precisamente la propiedad que más importa a un juego que convierte la probabilidad en una moneda. Además, como $F(x)$ es una combinación afín de dos puntuaciones aditivas, el ensamble sigue siendo **exactamente explicable** (Ecuación 12). El estudio de ablación confirmó la composición elegida: añadir un bosque aleatorio o un modelo de vecinos no mejora ninguna métrica de forma apreciable, rompe la explicabilidad exacta y multiplica por catorce o más el tamaño del modelo. **Regla:** se combinan solo los modelos que aportan información, y la combinación se hace en la escala que preserva la calibración.

**Ejemplo numérico.** Para la enfermera de 31 años de «Turno de Noche» (`01_TRACEABILITY.md`, sección 4.3), los modelos base devuelven $F_{\text{hgb}} = -0{,}2579$ y $F_{\text{lr}} = -0{,}4234$. El ensamble calcula $F = 0{,}0114 + 0{,}8848 \cdot (-0{,}2579) + 0{,}1013 \cdot (-0{,}4234) = -0{,}2597$ y $\hat{p} = \sigma(-0{,}2597) = 0{,}4354$, exactamente lo que devuelve `predict_proba` y lo que registró el BFF durante el juego. Los dos modelos base coinciden en el signo (ambos sitúan el caso algo por debajo del equilibrio) y el meta-learner los promedia con sus pesos.

### 2.3. Evaluación, incertidumbre y explicabilidad

#### Ecuación 9 — Métricas de evaluación

*Fichero de origen: `eq_metrics.tex`*

$$
\begin{aligned}
\text{Precision} &= \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}, \qquad
\text{Accuracy} = \frac{TP + TN}{N} \\
\text{Brier} &= \frac{1}{N}\sum_{i=1}^{N}\bigl(\hat{p}_i - y_i\bigr)^2, \qquad
\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{N}\,\Bigl|\,\overline{\hat{p}}_{S_b} - \overline{y}_{S_b}\Bigr|,\ B = 10
\end{aligned}
$$

**Interpretación.** La primera línea agrupa las métricas de clasificación, que evalúan decisiones de sí o no:

- La **precisión** responde a «de las personas a quienes el modelo otorga movilidad, ¿cuántas la tienen realmente?».
- La **exhaustividad** (*recall*) responde a «de las personas con movilidad real, ¿a cuántas detecta el modelo?».
- $F_1$ equilibra ambas, y la **exactitud** (*accuracy*) mide la proporción global de aciertos.

La segunda línea agrupa las métricas probabilísticas, que evalúan la calidad de la probabilidad misma:

- La **puntuación de Brier** es el error cuadrático medio entre la probabilidad estimada y el resultado real.
- El **error de calibración esperado** (ECE) divide las predicciones en diez tramos y mide cuánto se aleja, en cada tramo, la probabilidad media predicha de la frecuencia real observada.

**Uso en el proyecto.** Todas estas métricas se calcularon para los once candidatos y para el ensamble sobre el conjunto de validación, y se registraron en `model_comparison.json`. Los candidatos se ordenaron por precisión promedio (el área bajo la curva precisión-exhaustividad), con la puntuación de Brier como criterio de desempate.

**Conclusión y regla derivada.** El ensamble desplegado obtuvo un Brier de 0,0879 y un ECE de 0,0073 en validación, y un Brier de 0,0863 en prueba. El *gradient boosting* entrenado con SMOTE, en cambio, alcanzó un ECE de 0,078, más de diez veces peor. **Regla:** no basta con que un modelo acierte; debe acertar con probabilidades fiables. Por eso las métricas probabilísticas son decisivas en la selección, y una exactitud aislada no es criterio suficiente.

#### Ecuación 10 — Intervalo de confianza de Wilson

*Fichero de origen: `eq_wilson.tex`*

$$
\text{CI}_{95\%}(\hat{\pi}) = \frac{\hat{\pi} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{\pi}(1-\hat{\pi})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},
\qquad z = 1.96
$$

**Interpretación.** Toda proporción medida sobre una muestra finita, como una precisión del 95,9 %, es una estimación sujeta a error. El intervalo de Wilson delimita el rango en que se encuentra el valor verdadero con un 95 % de confianza. Se prefirió al intervalo normal clásico porque sigue siendo fiable cuando la proporción está próxima a 0 o a 1, que es exactamente el caso de precisiones cercanas al 95 %, y porque nunca produce límites fuera del rango entre 0 y 1.

**Uso en el proyecto.** `04_evaluator.py` acompaña con un intervalo de Wilson cada precisión y cada exactitud que reporta en `traceability_report.json`. Este cálculo es independiente del remuestreo de 2000 iteraciones descrito en el documento de trazabilidad.

**Conclusión y regla derivada.** En el conjunto de prueba, la precisión de la decisión «movilidad» es del 95,69 %, con un intervalo de 94,18 % a 96,82 %. La de «sin movilidad» es del 95,41 %, con un intervalo de 94,87 % a 95,90 %. Ambas estimaciones puntuales superan el 95 %, y los intervalos muestran que el verdadero valor se mantiene en torno a ese nivel. El intervalo de «movilidad» es el más ancho porque descansa sobre menos casos (928 decisiones frente a 6342), una diferencia que el intervalo hace explícita en lugar de ocultarla. **Regla:** ninguna cifra de rendimiento se comunica sin su intervalo de confianza.

#### Ecuación 11 — Clasificación selectiva

*Fichero de origen: `eq_selective_classification.tex`*

$$
\begin{aligned}
(\ell^*, u^*) &= \arg\max_{\ell < u}\ \frac{\bigl|\{i \in V : \hat{p}_i \le \ell \ \lor\ \hat{p}_i \ge u\}\bigr|}{|V|}
\quad \text{s.t.}\quad
\Pr\bigl(y = 1 \mid \hat{p} \ge u\bigr) \ge 0.95,\ \ \Pr\bigl(y = 0 \mid \hat{p} \le \ell\bigr) \ge 0.95 \\
(\ell^*, u^*) &= (0.23,\ 0.78) \\
\text{decision}(x) &= \begin{cases}
\text{mobility} & \hat{p}(x) \ge u^* \\
\text{no mobility} & \hat{p}(x) \le \ell^* \\
\text{mobility with probability } \hat{p}(x) & \ell^* < \hat{p}(x) < u^* \quad \text{(systemic luck)}
\end{cases}
\end{aligned}
$$

**Interpretación.** El modelo solo emite un veredicto cuando tiene la confianza suficiente. Se buscan dos umbrales, uno inferior $\ell$ y otro superior $u$, que maximicen la proporción de casos decididos (la *cobertura*). La búsqueda está sujeta a una restricción doble: cuando el modelo afirma «movilidad», debe acertar al menos en el 95 % de los casos, y lo mismo cuando afirma «sin movilidad». Además, cada decisión debe estar respaldada por al menos 30 casos. La búsqueda se realizó exclusivamente sobre el conjunto de validación $V$.

**Uso en el proyecto.** El resultado es la banda $[0{,}23;\ 0{,}78]$, que la API lee de `traceability_report.json` en cada arranque. Fuera de la banda, el desenlace del personaje es determinista. Dentro de ella, el BFF resuelve el destino con una moneda ponderada por $\hat{p}(x)$, mecanismo que el juego denomina **suerte sistémica**.

**Conclusión y regla derivada.** En el conjunto de prueba, el ensamble decide de forma determinista el **74,50 %** de los casos, con una exactitud del **95,45 %** sobre ellos, y deja el 25,50 % restante al azar ponderado. Esta es la regla de negocio central del proyecto: **el modelo no inventa certezas que no tiene**. Cuando el perfil censal no determina el destino de una persona con alta confianza, el juego lo reconoce explícitamente y lo convierte en una tirada de dados, cuyas probabilidades siguen siendo las del modelo.

**Lectura intuitiva (Figura 13).** Cada borde de la banda es el punto donde su decisión deja de alcanzar el 95 %. Al subir el corte inferior $\ell$, la decisión «sin movilidad» abarca a más personas pero acierta menos, y cruza el 95 % justo en 0,23. Al bajar el corte superior $u$, la decisión «movilidad» cruza el 95 % justo en 0,78. De las 2450 parejas $(\ell, u)$ de la rejilla, solo 506 cumplen ambas restricciones en validación, y la Ecuación 11 elige la de mayor cobertura (74,50 %). En prueba, el 65,0 % de las personas cae en «sin movilidad», donde la tasa real es del 4,6 %; el 25,5 % cae en la banda, donde es del 46,5 %, casi una moneda al aire; y el 9,5 % cae en «movilidad», donde es del 95,7 %. La banda no es un intervalo arbitrario: es la región donde el propio censo no distingue entre los dos destinos.

#### Ecuación 12 — Valores de Shapley (SHAP)

*Fichero de origen: `eq_shapley.tex`*

$$
\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,\bigl(|F| - |S| - 1\bigr)!}{|F|!}\Bigl[v_x\bigl(S \cup \{j\}\bigr) - v_x(S)\Bigr],
\qquad F(x) = \phi_0 + \sum_{j \in F} \phi_j(x)
$$

**Interpretación.** Procedente de la teoría de juegos cooperativos, el valor de Shapley $\phi_j(x)$ reparte la predicción de un individuo entre sus rasgos de forma justa. Considera todas las coaliciones posibles $S$ de los demás rasgos, mide cuánto cambia la predicción al añadir el rasgo $j$ a cada coalición y promedia esas contribuciones con pesos que equilibran coaliciones de distinto tamaño. La segunda igualdad expresa la propiedad de **exactitud local**: la predicción del modelo, en *log-odds*, es exactamente la suma de un valor base $\phi_0$ más las contribuciones de todos los rasgos. Una tercera propiedad, la **linealidad**, es la que permite explicar el ensamble: si $F = b + w_{\text{hgb}} F_{\text{hgb}} + w_{\text{lr}} F_{\text{lr}}$, entonces $\phi_j(F) = w_{\text{hgb}}\,\phi_j(F_{\text{hgb}}) + w_{\text{lr}}\,\phi_j(F_{\text{lr}})$.

**Uso en el proyecto.** Se explicó el **ensamble completo** para una muestra de 2000 personas del conjunto de prueba, seleccionada con la semilla 42. La contribución del *gradient boosting* se calculó de forma exacta con `TreeExplainer` intervencional; la de la regresión logística, con su fórmula cerrada, $\phi_j = (\beta_j / s_j)(x_j - \mathbb{E}[x_j])$. Ambas se calcularon frente a la misma muestra de referencia de 100 personas de entrenamiento y se combinaron con los pesos de la segunda capa. La exactitud local se verificó numéricamente: la discrepancia máxima entre la suma de contribuciones y la salida del ensamble fue de $7{,}6 \times 10^{-8}$, un error numérico despreciable frente a contribuciones del orden de la unidad.

**Conclusión y regla derivada.** El modelo deja de ser una caja negra. Los rasgos más influyentes son el estado civil (0,93 de contribución media absoluta), la edad (0,72), las ganancias de capital (0,68) y la educación (0,44). El tipo de empleo (0,079), el país de origen (0,080) y la raza (0,049) son los de menor peso. **Regla:** cada predicción es descomponible y auditable rasgo por rasgo, y las relaciones aprendidas son coherentes con la literatura socioeconómica.

### 2.4. Aplicación en el juego

#### Ecuación 13 — Exactitud global esperada del juego

*Fichero de origen: `eq_game_accuracy.tex`*

$$
\mathbb{E}[\text{Accuracy}_{\text{game}}] = \frac{1}{N}\Biggl[\sum_{i \notin \text{band}} \mathbb{1}\bigl(\text{decision}_i = y_i\bigr)
+ \sum_{i \in \text{band}} \Bigl(y_i\,\hat{p}_i + (1-y_i)(1-\hat{p}_i)\Bigr)\Biggr] = 85.22\%
$$

**Interpretación.** Mide con qué frecuencia el desenlace que vive el jugador coincide con la realidad censal del personaje, considerando a **todos** los jugadores y no solo a los decididos de forma determinista. El primer sumando cuenta los aciertos fuera de la banda, donde la decisión es fija. El segundo suma, para cada caso dentro de la banda, la probabilidad de que la moneda caiga del lado correcto: $\hat{p}_i$ si la persona tenía movilidad y $1 - \hat{p}_i$ si no la tenía.

**Uso en el proyecto.** Se evaluó analíticamente sobre el conjunto de prueba y se contrastó con 2000 partidas simuladas, cuyos resultados se encuentran entre el 84,74 % y el 85,68 % (véase `01_TRACEABILITY.md`, sección 3.3). Como cada moneda es una Bernoulli independiente, la desviación típica de la exactitud de una partida es $\sqrt{\sum_{i \in \text{band}} \hat{p}_i(1-\hat{p}_i)}\,/\,N = 0{,}00242$; la de las 2000 partidas simuladas es 0,00244. La Figura 14 superpone esa normal analítica al histograma de las simulaciones.

**Conclusión y regla derivada.** La exactitud esperada del juego es del **85,22 %**. Es inferior al 87,67 % que obtendría un clasificador convencional con umbral 0,5 sobre el mismo conjunto, y esa diferencia es intencional: es el precio de no fingir certeza donde no la hay. **Regla:** la mecánica de suerte sistémica es un compromiso deliberado y cuantificado entre fidelidad estadística y honestidad sobre la incertidumbre, no una pérdida de calidad accidental.

#### Ecuación 14 — Ajuste por inflación (IPC)

*Fichero de origen: `eq_inflation.tex`*

$$
\begin{aligned}
\text{CPI}(t) &= \begin{cases} \text{CPI}^{\text{BLS}}(t) & t \le 2024 \\ \text{CPI}^{\text{BLS}}(2024)\,(1 + 0.025)^{\,t - 2024} & t > 2024 \end{cases} \\
x^{(1994)} &= x^{(t)} \cdot \frac{\text{CPI}(1994)}{\text{CPI}(t)}, \qquad x \in \{\text{capital gain},\ \text{capital loss}\} \\
\text{Threshold}^{(t)} &= 50\,000 \cdot \frac{\text{CPI}(t)}{\text{CPI}(1994)} \qquad \text{e.g. } t = 2026:\ 111\,191\ \text{USD}
\end{aligned}
$$

**Interpretación.** El modelo fue entrenado con dólares de 1994 y, por tanto, está «congelado» en esa economía. Las historias del juego, sin embargo, transcurren en otros años. La primera línea define el índice de precios al consumidor (IPC-U, serie CUUR0000SA0 de la Oficina de Estadísticas Laborales de EE. UU.): se usa el valor oficial hasta 2024, último año publicado, y a partir de ahí se proyecta con una inflación documentada del 2,5 % anual. La segunda línea convierte las cantidades monetarias del año de la historia a dólares de 1994 antes de enviarlas al modelo. La tercera realiza la conversión inversa del umbral de 50 000 USD, para expresarlo en dólares del año narrado.

**Uso en el proyecto.** El ajuste lo aplica el BFF (`bff-api/src/core/domain/services/consumer-price-index.ts`) a las ganancias y pérdidas de capital, que son los únicos rasgos monetarios del modelo.

**Conclusión y regla derivada.** Sin este ajuste, una ganancia de capital de 2026 se interpretaría con el poder adquisitivo de 1994 y sobrestimaría sistemáticamente la riqueza del personaje. Con él, el umbral de movilidad equivale a 73 568 USD en 2010, 84 719 USD en 2018, 105 833 USD en 2024, 111 191 USD en 2026 y 132 171 USD en 2033. **Ejemplo numérico.** En 2026, $\mathrm{CPI}(1994)/\mathrm{CPI}(2026) = 0{,}4497$: una ganancia de capital de 10 000 USD de 2026 llega al modelo como 4497 USD de 1994. Para un mismo perfil de referencia, la Figura 15 muestra que sin deflación la predicción entra en la zona «movilidad» con solo 5250 USD declarados, y con deflación hacen falta 11 500 USD; el cociente (2,19) reproduce, salvo la resolución de la rejilla de 250 USD, el factor $1/0{,}4497 = 2{,}22$. **Regla:** toda cantidad monetaria se normaliza a dólares de 1994 antes de llegar al modelo, y el umbral de movilidad que ve el jugador se expresa en la moneda del año que está viviendo.

---

## 3. Índice de gráficos descriptivos

Las figuras se encuentran en `ai-service/training-pipeline/artifacts/figures/`. Son dieciséis ficheros. Los diez primeros los genera el notebook; en ellos, el histograma con la banda de incertidumbre y la curva precisión-exhaustividad comparten el fichero `decision_policy.png` como dos paneles de una misma figura. Los seis restantes (Figuras 12 a 17, sección 3.5) los genera `src/generate_extra_plots.py`. Las seis figuras de evaluación estadística (curvas ROC y precisión-exhaustividad, validación cruzada, capacidad, residuos, techo de exactitud y garantía del 95 %) se describen en [`05_ADVANCED_EVALUATION.md`](05_ADVANCED_EVALUATION.md).

### 3.1. Análisis exploratorio de datos

#### Figura 1 — Distribuciones numéricas por clase de ingreso

**Fichero:** `eda_numeric_distributions.png`

Muestra, en forma de densidad, la distribución de la edad, las horas semanales de trabajo y los años de educación, comparando a quienes superan los 50 000 USD con quienes no.

**Qué demuestra.** Las clases están claramente separadas en los tres rasgos. Las personas con movilidad se concentran entre los 35 y los 55 años, mientras que la clase sin movilidad domina por debajo de los 25. También trabajan con más frecuencia jornadas superiores a las 40 horas y se acumulan en los niveles educativos universitarios (13 años o más). Esto confirma que los datos contienen **señal predictiva real**: un modelo construido sobre ellos tiene información genuina que aprender y no solo ruido.

#### Figura 2 — Tasa de movilidad por categoría

**Fichero:** `eda_rate_by_category.png`

Representa, para cada nivel educativo y cada estado civil con al menos 100 registros, la proporción de personas que superan los 50 000 USD, con la tasa global del 23,9 % como referencia.

**Qué demuestra.** La educación produce un gradiente casi monótono: desde tasas inferiores al 5 % en la educación primaria hasta más del 70 % en doctorados y escuelas profesionales. El estado civil marca una diferencia aún más abrupta, ya que las personas casadas superan el 40 %, frente a menos del 10 % en el resto de categorías. Estos patrones descriptivos anticipan exactamente lo que el análisis SHAP confirmará después (Figuras 8 a 10). La coincidencia entre lo que muestran los datos y lo que aprende el modelo es una evidencia de validez.

### 3.2. Balanceo y calibración

#### Figura 3 — Balance de clases: natural frente a SMOTE

**Fichero:** `smote_class_balance.png`

Compara el número de registros de cada clase en el conjunto de entrenamiento antes y después de aplicar SMOTENC: de 22 265 frente a 7009 se pasa a 22 265 frente a 22 265, con 15 256 registros sintéticos añadidos.

**Qué demuestra.** Documenta visualmente que el balanceo de clases **se aplicó y se cuantificó**, como exige la metodología. Su rechazo posterior no se debe a una omisión, sino a una decisión basada en evidencia (Ecuación 1 y Figura 4).

#### Figura 4 — Curvas de calibración: SMOTE frente a natural

**Fichero:** `calibration_smote_vs_natural.png`

Diagrama de fiabilidad sobre el conjunto de validación que compara el modelo elegido (HistGradientBoosting natural) con dos variantes entrenadas con SMOTE. Cada punto enfrenta la probabilidad media predicha en un tramo con la frecuencia real observada; la diagonal representa la calibración perfecta.

**Qué demuestra.** El modelo natural sigue la diagonal de cerca, mientras que las variantes con SMOTE se desplazan por debajo de ella, es decir, **sobrestiman** sistemáticamente la probabilidad de movilidad. Es la evidencia gráfica que justifica el rechazo de SMOTE: en un juego donde la probabilidad se convierte en una moneda, un modelo descalibrado otorgaría movilidad de forma injustificada.

### 3.3. Rendimiento y política de decisión (conjunto de prueba)

#### Figura 5 — Matrices de confusión y matriz de decisión selectiva

**Fichero:** `confusion_matrices.png`

Presenta tres paneles sobre los 9758 casos de prueba:

1. **Matriz de confusión con el umbral convencional de 0,5:** 468 falsos positivos y 1601 verdaderos positivos.
2. **Matriz de confusión con el umbral determinista de 0,7782:** los falsos positivos caen a 42 y los verdaderos positivos son 889.
3. **Matriz de decisión selectiva** con tres columnas: «sin movilidad», «suerte» y «movilidad».

**Qué demuestra.** Hace visible el compromiso entre precisión y cobertura. Elevar el umbral reduce drásticamente los errores en la decisión «movilidad», a costa de detectar menos casos. La matriz selectiva muestra cómo la política resuelve ese dilema: los errores deterministas son escasos (40 y 291 casos), y los casos ambiguos, 1331 y 1157, se derivan a la columna de suerte en lugar de decidirse incorrectamente.

#### Figura 6 — Histograma de probabilidades con la banda de incertidumbre

**Fichero:** `decision_policy.png` (panel izquierdo)

Histograma apilado, en escala logarítmica, de las probabilidades que el modelo asigna a los casos de prueba, coloreado por clase real y con la banda $[0{,}23;\ 0{,}78]$ sombreada.

**Qué demuestra.** Fuera de la banda, cada extremo está dominado por una sola clase: a la izquierda casi exclusivamente personas sin movilidad, y a la derecha casi exclusivamente personas con movilidad. Dentro de la banda, las dos clases se mezclan en proporciones comparables. Esto justifica visualmente que la región central sea, de verdad, una zona de incertidumbre, donde decidir de forma determinista sería arbitrario.

#### Figura 7 — Curva precisión-exhaustividad

**Fichero:** `decision_policy.png` (panel derecho)

Traza la precisión frente a la exhaustividad para todos los umbrales posibles sobre el conjunto de prueba, con la línea de referencia del 95 % de precisión y el punto de operación elegido marcado.

**Qué demuestra.** El umbral de 0,778, fijado únicamente con datos de validación, alcanza en prueba una precisión del 95,5 % con una exhaustividad del 38,1 %. Que el punto caiga sobre la línea del 95 % en un conjunto nunca utilizado para elegirlo demuestra que **la política generaliza** y no está sobreajustada. El área bajo esta curva es, además, la métrica con la que se ordenaron los modelos candidatos (Ecuación 9).

#### Figura 8 — Diagrama de fiabilidad en prueba

**Fichero:** `calibration_test.png`

Diagrama de fiabilidad del ensamble desplegado sobre el conjunto de prueba, con la banda de incertidumbre sombreada y la puntuación de Brier (0,0863) en el título.

**Qué demuestra.** Los puntos siguen la diagonal en todo el rango de probabilidades, con desviaciones moderadas en el interior de la banda; la mayor aparece en torno a 0,45. Confirma, sobre datos nunca vistos, que **las probabilidades del modelo son fiables**. Esta es la condición matemática que hace legítima la moneda ponderada de la suerte sistémica, porque la probabilidad con que se lanza la moneda corresponde de forma aproximada a la frecuencia real del desenlace.

### 3.4. Explicabilidad (SHAP)

#### Figura 9 — Resumen SHAP (*beeswarm*)

**Fichero:** `shap_beeswarm.png`

Representa, para 2000 personas del conjunto de prueba, la contribución SHAP exacta del ensamble para las catorce columnas más influyentes, más una fila que agrega las 92 columnas restantes. Cada valor de una variable categórica aparece como una columna binaria propia, por ejemplo `marital_status_Married-civ-spouse`. Cada punto es una persona; su posición horizontal indica cuánto empuja esa columna la predicción hacia la movilidad (derecha) o en contra (izquierda), y su color indica si el valor es alto o bajo.

**Qué demuestra.** Ofrece una visión global y a la vez individual del razonamiento del modelo, y permite comprobar que la dirección de cada efecto es coherente:

- Estar casado, tener más años de educación o trabajar más horas empuja hacia la movilidad.
- Las ganancias de capital elevadas producen los efectos individuales más intensos de todo el gráfico.
- La juventud y ocupaciones como `Other-service` o `Handlers-cleaners` la reducen.

También hace visible que el modelo ha absorbido desigualdades presentes en el censo de 1994: `sex_Female` es la sexta columna más influyente y desplaza la predicción hacia abajo, y `sex_Male` aparece también entre las catorce primeras con el efecto opuesto. La aportación lineal de la regresión logística hace este efecto de género más visible en el ensamble que en el *gradient boosting* solo. Esta transparencia es coherente con el propósito del juego, que consiste precisamente en exponer esas desigualdades estructurales.

#### Figura 10 — Importancia por rasgo

**Fichero:** `shap_trait_importance.png`

Gráfico de barras con la contribución SHAP media absoluta de cada rasgo original. Las columnas binarias generadas por la codificación de una misma variable categórica se agrupan en un único rasgo.

**Qué demuestra.** Establece una jerarquía clara e interpretable: estado civil, edad, ganancias de capital y educación concentran la mayor parte de la capacidad predictiva, mientras que la raza, el tipo de empleo y el país de origen tienen el menor peso. La coincidencia con la Figura 2 refuerza la validez del modelo, porque lo que aprende coincide con lo que muestran los datos.

#### Figura 11 — Dependencia SHAP de edad y educación

**Fichero:** `shap_dependence.png`

Representa, para la edad y para los años de educación, la contribución SHAP frente al valor del rasgo.

**Qué demuestra.** Revela la **forma** de las relaciones aprendidas, más allá de su magnitud:

- **Edad:** la contribución crece con rapidez hasta los 45-50 años, se estabiliza y desciende ligeramente después, como corresponde al ciclo vital de ingresos.
- **Educación:** crece de forma casi escalonada y monótona, con un salto marcado a partir de los 13 años de estudio, que equivalen a un título universitario.

Que el modelo haya aprendido relaciones no lineales y plausibles, y no artefactos de los datos, constituye una evidencia sustantiva de su validez.

> **Nota sobre la numeración.** Las Figuras 1 a 11 corresponden a los diez ficheros PNG del notebook, porque el histograma y la curva precisión-exhaustividad comparten fichero; las Figuras 12 a 17 corresponden a los seis ficheros de `generate_extra_plots.py`.

### 3.5. Figuras complementarias: banda, suerte, inflación y deriva

Estas seis figuras hacen visibles los mecanismos menos intuitivos del modelo. Las genera `src/generate_extra_plots.py`, que importa la lógica exacta de `04_evaluator.py` (búsqueda de la banda, rejillas y simulación de monedas) y de `05_ood_testing.py` (carga, recodificación y deflación de la ACS 2018) en lugar de reimplementarla. Antes de dibujar, el *script* recalcula la banda con `choose_band` y comprueba que coincide con la de `traceability_report.json`; también recalcula las 2000 partidas y comprueba que su rango coincide con el registrado. Todas las cifras de las figuras quedan en `artifacts/extra_figures_report.json`.

#### Figura 12 — Comparación de modelos: distribución natural frente a SMOTE

**Fichero:** `model_comparison.png`

Tres paneles sobre el conjunto de validación, para las cinco familias entrenadas con la distribución natural (azul) y con SMOTE (dorado): precisión promedio, error de calibración esperado y probabilidad media predicha. Las líneas discontinuas marcan el ensamble desplegado y la tasa real de movilidad (0,239).

**Qué demuestra.** Resume en una sola imagen la decisión sobre el balanceo y la elección del ensamble. SMOTE no mejora la discriminación de ninguna familia, multiplica su error de calibración (de 0,008 a 0,112 en la regresión logística) y eleva la probabilidad media muy por encima de la tasa real (entre 0,30 y 0,35 frente a 0,239). El ensamble iguala la mejor precisión promedio (0,8303) con el menor ECE de todos (0,0073) y una probabilidad media (0,242) prácticamente igual a la real. Complementa a las Figuras 3 y 4 y a la Ecuación 1.

#### Figura 13 — Anatomía de la banda de indecisión

**Fichero:** `band_anatomy.png`

1. **Precisión de cada decisión determinista según su corte**, en validación (línea continua) y en prueba (discontinua), con el suelo del 95 % y la banda sombreada.
2. **Mapa de las bandas factibles:** cada celda es una pareja $(\ell, u)$ de la rejilla de la Ecuación 11, coloreada por su cobertura; las celdas en blanco incumplen alguna de las dos restricciones del 95 %. La estrella marca la banda elegida.
3. **Las tres zonas en prueba:** proporción de personas, tasa real de movilidad y probabilidad media predicha en cada zona.

**Qué demuestra.** Hace intuitiva la Ecuación 11. La banda $[0{,}23;\ 0{,}78]$ no se fijó a ojo: cada borde está donde la curva de su decisión corta el 95 %, y la estrella está en la esquina de la región factible que maximiza la cobertura. Que las curvas de prueba sigan a las de validación demuestra que la banda generaliza. El tercer panel da el sentido narrativo: fuera de la banda el desenlace es casi seguro (4,6 % y 95,7 % de movilidad real), y dentro de ella es casi una moneda al aire (46,5 %), que es exactamente lo que el juego hace allí.

#### Figura 14 — Simulación de la suerte sistémica

**Fichero:** `systemic_luck_simulation.png`

El panel izquierdo es el histograma de las 2000 partidas simuladas sobre los 9758 personajes de prueba (semilla 42), con el valor esperado analítico de la Ecuación 13, la normal que predice su varianza y el rango que contiene al 95 % de las partidas. El panel derecho compara tres exactitudes: la de los casos decididos (95,45 %), la de todos los jugadores con la moneda (85,22 %) y la de un clasificador convencional con umbral 0,5 (87,67 %).

**Qué demuestra.** La suerte es aleatoria para cada jugador y predecible para el conjunto: el histograma se centra en el valor analítico y su anchura es la que predice la teoría. El panel derecho muestra en qué zona se cumple el 95 % y cuánto cuesta, deliberadamente, no fingir certeza dentro de la banda (Ecuación 13).

#### Figura 15 — Ajuste por inflación

**Fichero:** `inflation_adjustment.png`

1. El IPC-U oficial hasta 2024 y su proyección al 2,5 % anual, con 1994 marcado como la moneda del modelo.
2. El umbral de movilidad (50 000 USD de 1994) expresado en dólares de cada año, con los años de las historias marcados.
3. La probabilidad que devuelve el modelo para un mismo personaje de 2026 según la ganancia de capital declarada, con deflación (como hace el BFF) y sin ella.

**Qué demuestra.** El tercer panel es el argumento de la Ecuación 14. Sin deflación, el modelo lee dólares de 2026 como si fueran de 1994, y el personaje alcanza la zona «movilidad» con una ganancia de 5250 USD; con deflación necesita 11 500 USD. La curva verde es la gris desplazada a la derecha por el factor $1/0{,}4497 = 2{,}22$. Sin el ajuste, el juego regalaría movilidad a cualquier personaje contemporáneo con algo de capital.

#### Figura 16 — *Concept drift*: el modelo de 1994 frente a la ACS 2018

**Fichero:** `concept_drift.png`

Entre dos poblaciones puede cambiar la tasa base $P(Y)$ (*prior shift*), la distribución de los perfiles $P(X)$ (*covariate shift*) o la relación entre ambos, $P(Y \mid X)$ (*concept drift*). Solo el último invalida un clasificador bien calibrado, porque cambia la respuesta correcta para un mismo perfil. La figura tiene cuatro paneles sobre 195 665 personas de California y 103 021 de Nueva York (`01_TRACEABILITY.md`, sección 3.6):

1. **Diagramas de fiabilidad** del modelo congelado en el censo de 1994 (prueba) y en la ACS 2018.
2. **Precisión de «movilidad» según el corte superior** $u$.
3. **Precisión de «sin movilidad» según el corte inferior** $\ell$, con los cortes recalibrados de 0,08 (CA) y 0,07 (NY).
4. **Precisión de «movilidad» por sexo y grupo ocupacional** con la banda congelada.

**Qué demuestra.** La tasa base apenas cambió (23,9 % en 1994 y 23,5 % en CA), pero en 2018 los puntos de fiabilidad caen por debajo de la diagonal a partir de 0,5: para un mismo perfil, la probabilidad real de superar el umbral es menor. Es la firma del *concept drift*. El segundo panel muestra que **ningún** corte superior devuelve a 2018 al 95 % (el máximo ronda el 80 %): el error es de orden, no de umbral. El tercero muestra que el corte inferior sí puede moverse para recuperar el 95 %. El cuarto localiza el error: la precisión de «movilidad» es del 59 % (CA) y del 51 % (NY) para las mujeres, frente al 83 % y al 80 % de los hombres, y cae al 44 % y al 42 % en *Adm-clerical*. El desglose queda persistido en `extra_figures_report.json` (`concept_drift.subgroups`).

#### Figura 17 — Una historia rejugada fuera de línea: «Turno de Noche»

**Fichero:** `turno_de_noche_trace.png`

Probabilidad que devuelve el modelo para el perfil base de Daniela y para los seis caminos de la historia, tras aplicar los modificadores ocultos de cada decisión (inyección de contexto) y la deflación de 2033, frente a la banda de indecisión.

**Qué demuestra.** Conecta las ecuaciones con el juego. Los modificadores de cada decisión llevan a Daniela dentro de la banda mientras la familia se mantiene unida (0,435, 0,373 y 0,256: decide la moneda) y fuera de ella cuando la migración rompe el hogar (0,082, 0,011 y 0,063: «no» determinista). Las seis probabilidades coinciden hasta la cuarta cifra con las registradas por el BFF durante el juego real (`01_TRACEABILITY.md`, secciones 3.5 y 4.3).

---

## 4. Conclusión

El modelo matemático de *Unfair Life* cumple el criterio de contar con **ecuaciones explicadas, gráficos descriptivos y alta claridad** por tres motivos:

1. **Ecuaciones explicadas.** Cada una de las catorce ecuaciones se presenta junto con su interpretación, su uso concreto en el proyecto y la regla de negocio que se deriva de ella. Cada modelo de aprendizaje tiene su propia formulación (Ecuaciones 2 a 7) antes de la ecuación general del ensamble (Ecuación 8). Ninguna es decorativa: todas condicionan una decisión verificable del sistema, desde el rechazo de SMOTE y la composición del ensamble hasta la normalización monetaria por inflación.
2. **Gráficos descriptivos.** Las diecisiete figuras cubren el ciclo completo: la existencia de señal en los datos (Figuras 1 y 2), la decisión sobre el balanceo y la selección de modelos (Figuras 3, 4 y 12), el rendimiento y la política de decisión sobre datos no vistos (Figuras 5 a 8), la explicabilidad del razonamiento del modelo (Figuras 9 a 11) y los mecanismos que el juego construye sobre él: la banda de indecisión (Figura 13), la suerte sistémica (Figura 14), el ajuste por inflación (Figura 15), la robustez ante el *concept drift* (Figura 16) y su aplicación a una historia concreta (Figura 17).
3. **Alta claridad y coherencia.** Las conclusiones de las ecuaciones y las de los gráficos se refuerzan mutuamente, y todas las cifras proceden de artefactos generados automáticamente y trazables (véase `01_TRACEABILITY.md`). Lo que este documento describe es, por tanto, exactamente el modelo que opera en producción.

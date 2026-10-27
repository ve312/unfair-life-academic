# Defensa Metodológica: las cinco objeciones previsibles

> **Ámbito.** Este documento reúne, en un solo lugar, la respuesta a las cinco objeciones que un evaluador estricto podría plantear al componente de inteligencia artificial de *Unfair Life*. Cada objeción se responde con tres tipos de argumento: la **evidencia** medida en los artefactos del proyecto, el **fundamento teórico** (estadística, teoría de la decisión, teoría de juegos) y el **estado del arte** que respalda la decisión. Al final de cada sección, una tabla indica dónde se desarrolla ese mismo argumento en los demás documentos, para que pueda trasladarse al artículo final.
>
> Las cifras nuevas de este documento proceden de `training-pipeline/artifacts/advanced_metrics_report.json` (producido por `src/advanced_metrics.py`) y de `extra_figures_report.json` (producido por `src/generate_extra_plots.py`). Las demás, de los informes descritos en [`01_TRACEABILITY.md`](01_TRACEABILITY.md).

| # | Objeción | Respuesta en una frase | Sección |
|---|---|---|---|
| 1 | «Un modelo que no acierta el 95 % en todos los casos es un mal modelo» | Con estos rasgos, ningún modelo puede superar en torno a un 87 % de exactitud: personas con perfiles censales idénticos tienen ingresos distintos | 1 |
| 2 | «La banda y la moneda son un artificio para inflar la precisión» | Son la regla de decisión óptima con opción de rechazo (Chow, 1970) y la única que respeta la probabilidad condicional de cada perfil | 2 |
| 3 | «No se balancearon las clases» | Se balancearon, se midió el efecto y se rechazó: SMOTE distorsiona la probabilidad que el juego convierte en moneda | 3 |
| 4 | «El intervalo de confianza de una decisión baja a 94,87 %» | Es la consecuencia esperada de fijar la restricción exactamente en el 95 %; la alternativa conservadora se calculó y cuesta cobertura sin mejorar el juego | 4 |
| 5 | «El modelo falla en 2018, y sobre todo con mujeres» | Es *concept drift* documentado en la literatura económica; el modelo lo detecta, lo localiza y restituye el 95 % donde es posible | 5 |

---

## 1. El techo de precisión: por qué ningún modelo puede predecir la movilidad con un 95 % de exactitud global

### 1.1. Evidencia

**Perfiles idénticos, destinos distintos.** En los 48 790 registros limpios hay 41 439 perfiles distintos de 13 rasgos. De ellos, 3746 se repiten y agrupan a 11 097 personas. En **1029** de esos perfiles repetidos conviven personas que superan el umbral con personas que no lo superan, y reúnen al **35 %** de los registros repetidos. Sobre esos 11 097 registros, cualquier función determinista de los 13 rasgos, por perfecta que sea, comete **al menos 1378 errores, un 12,4 %**, porque a un mismo perfil solo puede asignarle un destino. El perfil repetido más numeroso lo ilustra: 26 hombres blancos de 39 años, casados, con bachillerato, en *Craft-repair*, 40 horas semanales y sin capital; **9 superan el umbral y 17 no**. No existe información en el censo que los distinga.

Esta cota no depende de que los perfiles coincidan exactamente. Agrupando a las personas en 10 890 celdas descritas como lo haría un narrador (franja de edad, educación, ocupación, sexo, estado civil, franja de horas y si tiene ganancias de capital), el error mínimo de cualquier regla sobre esas celdas es del **12,5 %**.

**El modelo calibrado conoce su propio techo.** Si un modelo está calibrado, la exactitud esperada de la mejor decisión que se puede tomar con sus probabilidades es la media de $\max(\hat{p}, 1 - \hat{p})$. Para el ensamble desplegado, sobre el conjunto de prueba, esa media vale **87,41 %**, y la exactitud real del ensamble con umbral 0,5 es **87,67 %**: el modelo ya opera en su techo. Solo el **52,1 %** de las personas recibe una probabilidad con confianza de al menos el 95 %. La pendiente de calibración (1,02) y la descomposición de Murphy (fiabilidad 0,0007 frente a resolución 0,0958) confirman que esa confianza es fiable (`05_ADVANCED_EVALUATION.md`, sección 4).

**Más datos o más capacidad no cambiarían la conclusión.** Las curvas de aprendizaje muestran rendimientos decrecientes: duplicar los datos de entrenamiento del *gradient boosting*, de 14 637 a 29 274 filas, eleva su precisión promedio en validación solo de 0,821 a 0,830, y la regresión logística apenas se mueve desde las 10 000 filas (0,761 frente a 0,768). La rejilla de 16 configuraciones de hiperparámetros del *gradient boosting* varía la precisión promedio en menos de 0,008 (`05_ADVANCED_EVALUATION.md`, sección 3). Esas ganancias marginales afectan a la ordenación de los casos, no al error mínimo que impone la existencia de perfiles idénticos con destinos distintos: ese error es una propiedad de la información disponible y ningún volumen de datos con los mismos rasgos lo elimina.

### 1.2. Fundamento teórico

Para cualquier clasificador $h$ que solo observe $x$, el error está acotado inferiormente por el **error de Bayes**:

$$
R(h) \;\ge\; R^* \;=\; \mathbb{E}_x\bigl[\min\bigl(\eta(x),\, 1 - \eta(x)\bigr)\bigr], \qquad \eta(x) = P(Y = 1 \mid X = x)
$$

$R^*$ es cero solo si cada perfil determina por completo su destino, es decir, si $\eta(x) \in \{0, 1\}$ para todo $x$. Los perfiles con etiquetas mezcladas demuestran empíricamente que $\eta(x)$ toma valores intermedios. Por tanto, $R^* > 0$ con independencia del algoritmo, y una exactitud global del 95 % es inalcanzable **por definición**, no por limitación técnica.

La literatura distingue dos fuentes de incertidumbre (Hüllermeier y Waegeman, 2021). La **epistémica** procede de la falta de conocimiento del modelo y se reduce con más datos o mejores algoritmos. La **aleatoria** procede de la variabilidad intrínseca del fenómeno respecto de los rasgos observados y no se reduce con ningún modelo. Los rendimientos decrecientes de las curvas de aprendizaje, la insensibilidad a los hiperparámetros y, sobre todo, los perfiles idénticos con destinos distintos indican que la incertidumbre residual de *Unfair Life* es mayoritariamente aleatoria.

### 1.3. Estado del arte

- **Economía del ingreso.** Las ecuaciones de ingresos de Mincer (1974), basadas en educación y experiencia, explican solo una fracción minoritaria de la varianza del logaritmo del ingreso individual. Jencks et al. (1972) concluyeron que la familia de origen, la escolaridad y las capacidades medidas dejan sin explicar la mayor parte de la desigualdad de ingresos entre personas comparables, y atribuyeron un peso sustancial a la suerte. Frank (2016) desarrolla el mismo argumento para el éxito económico.
- **Predictibilidad de trayectorias vitales.** En el *Fragile Families Challenge* (Salganik et al., 2020), 160 equipos con miles de variables longitudinales no lograron predecir resultados vitales mucho mejor que una referencia simple de cuatro variables. La conclusión de los autores es que ciertos resultados sociales son poco predecibles a nivel individual aunque se disponga de datos ricos.
- **El propio censo *Adult*.** Tras casi tres décadas como banco de pruebas, la exactitud publicada en este conjunto ronda el 85–88 % para los mejores métodos (Kohavi, 1996; Ding et al., 2021). El 87,67 % del ensamble está en el extremo superior de ese rango.

### 1.4. Consecuencia para el proyecto

El criterio de precisión superior al 95 % no puede cumplirse sobre **todas** las personas; puede cumplirse sobre las personas cuyo perfil sí determina su destino. Por eso el proyecto lo formula como precisión de las decisiones deterministas, y la sección 2 justifica cómo se elige ese subconjunto. Pretender un 95 % global con estos datos solo sería posible sobreajustando o filtrando información del conjunto de prueba.

| Dónde se desarrolla | Documento y sección |
|---|---|
| Análisis de perfiles idénticos, techo de exactitud y curva riesgo-cobertura | `05_ADVANCED_EVALUATION.md`, sección 5 (figura `accuracy_ceiling.png`) |
| Curvas de aprendizaje e hiperparámetros | `05_ADVANCED_EVALUATION.md`, sección 3 (figura `capacity_analysis.png`) |
| Precisión frente a exactitud global | `03_IMPLEMENTATION_AND_BALANCE.md`, sección 2 |
| Exactitud esperada del juego (Ecuación 13) | `02_MATHEMATICS.md`, sección 2.4 |

---

## 2. La banda de indecisión y el dado ponderado como solución al techo

### 2.1. La banda es la regla óptima con opción de rechazo

Chow (1970) demostró que, si el coste de abstenerse es menor que el de equivocarse, la regla que minimiza el error para una tasa de rechazo dada es **abstenerse cuando la probabilidad posterior máxima no supera un umbral**, y decidir la clase más probable en otro caso. La banda de *Unfair Life* es exactamente esa regla, con dos refinamientos:

1. **Umbrales asimétricos por clase.** En lugar de un único umbral sobre $\max(\hat{p}, 1-\hat{p})$, la Ecuación 11 fija un corte por decisión ($\ell^* = 0{,}23$ y $u^* = 0{,}78$) y exige el 95 % a **cada** una por separado. Con un umbral simétrico, la clase mayoritaria podría compensar los errores de la minoritaria; con dos, ninguna decisión queda por debajo del objetivo.
2. **Selección en validación, verificación en prueba.** Es el protocolo de la clasificación selectiva moderna (El-Yaniv y Wiener, 2010; Geifman y El-Yaniv, 2017): se fija el riesgo objetivo, se maximiza la cobertura y se comprueba en datos no vistos.

La **curva riesgo-cobertura** (`05_ADVANCED_EVALUATION.md`, sección 5) muestra la frontera completa: ordenando a las personas por confianza, la exactitud se mantiene sobre el 95 % hasta el 76,5 % de cobertura y cae al 87,67 % si se decide sobre todas. La banda desplegada opera en el 74,5 %, justo por debajo de la frontera, porque exige el 95 % a cada decisión y no solo al conjunto.

### 2.2. El dado ponderado preserva la verdad de cada perfil

Dentro de la banda caben dos políticas: decidir por la clase más probable (umbral 0,5) o lanzar una moneda con probabilidad $\hat{p}$. La primera tiene más exactitud: en los 2488 casos de la banda acierta el 64,95 %, frente al 55,35 % esperado de la moneda. El proyecto elige la moneda deliberadamente, por tres razones:

1. **Fidelidad condicional.** Para un perfil con $\hat{p} = 0{,}40$, la regla de umbral 0,5 asigna «sin movilidad» al 100 % de las partidas, mientras que la realidad censal de ese perfil es un 40 % de movilidad. La moneda reproduce ese 40 %. El juego no es un clasificador que deba acertar a un individuo, sino un **simulador** que debe reproducir $P(Y \mid x)$. Desde la teoría de la decisión, devolver un resultado determinista equivale a informar una probabilidad de 0 o 1, que una regla de puntuación propia como la de Brier penaliza frente a informar la probabilidad verdadera (Gneiting y Raftery, 2007).
2. **Coherencia con los datos.** La sección 1 demuestra que personas con el mismo perfil tienen destinos distintos. Una regla determinista contradiría esa evidencia, porque daría siempre el mismo destino al mismo perfil; la moneda la respeta.
3. **Justicia como lotería ponderada.** En la teoría de la justicia distributiva, cuando varios candidatos tienen pretensiones legítimas pero desiguales sobre un bien indivisible y no existe información para discriminar entre ellos, la asignación justa es una **lotería ponderada** por la fuerza de cada pretensión (Broome, 1990). En términos de teoría de juegos, es una **estrategia mixta**: frente a un adversario, la naturaleza, cuyo «tipo» (los factores no observados) es desconocido, la regla aleatorizada no promete al jugador un resultado que los datos no respaldan y le da exactamente la probabilidad que su perfil merece. La probabilidad de la moneda la fija el modelo calibrado, no el diseñador.

El coste de esta elección está cuantificado y es pequeño: la exactitud esperada del juego es del 85,22 %, frente al 87,67 % de decidirlo todo con umbral 0,5 (Figura 14 de `02_MATHEMATICS.md`). A cambio, la tasa de movilidad que experimentan los jugadores (22,0 %) queda más cerca de la real (23,9 %) que la de la regla determinista (21,2 %).

### 2.3. La moneda es verificable

La moneda no es una caja negra: se lanza con `crypto.randomInt`, la regla «movilidad si tirada < $p$» está cubierta por pruebas unitarias en su valor límite, cada tirada queda en el registro de auditoría, y 2000 partidas simuladas reproducen la exactitud analítica con la varianza que predice la teoría (`01_TRACEABILITY.md`, secciones 3.3 y 4).

| Dónde se desarrolla | Documento y sección |
|---|---|
| Clasificación selectiva (Ecuación 11) y su lectura intuitiva | `02_MATHEMATICS.md`, Ecuación 11; Figura 13 |
| Exactitud esperada del juego (Ecuación 13) y simulación | `02_MATHEMATICS.md`, Ecuación 13; Figura 14; `01_TRACEABILITY.md`, sección 3.3 |
| Curva riesgo-cobertura | `05_ADVANCED_EVALUATION.md`, sección 5 |
| Estado del arte de la abstención | `04_MODEL_SELECTION.md`, sección 2.4 |

---

## 3. El rechazo del balanceo artificial (SMOTE)

### 3.1. Evidencia

SMOTENC se aplicó (15 256 registros sintéticos, 44 530 filas, 50/50) y las cinco familias se entrenaron con y sin él. En las cinco, el balanceo **redujo** la precisión promedio (entre 0,008 y 0,050), **multiplicó** el error de calibración esperado (de 0,008 a 0,112 en la regresión logística) y elevó la probabilidad media a entre 0,30 y 0,35, cuando la tasa real es 0,239 (Figura 12 de `02_MATHEMATICS.md`). La variante con pesos de clase, otra forma de reequilibrio, igualó la discriminación (0,830) con un ECE de 0,100, diez veces peor.

### 3.2. Fundamento teórico

Un clasificador probabilístico estima $P(Y \mid x) \propto P(x \mid Y)\,P(Y)$. Entrenar con una proporción artificial $\pi'$ en lugar de la real $\pi$ cambia el factor $P(Y)$ y, por tanto, los *odds* de **toda** predicción:

$$
\frac{\hat{p}'(x)}{1 - \hat{p}'(x)} \;=\; \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{\pi' / (1 - \pi')}{\pi / (1 - \pi)} \;=\; \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot 3{,}18
$$

La corrección de este sesgo exige deshacer la transformación (Saerens, Latinne y Decaestecker, 2002), y eso devuelve exactamente al modelo entrenado con la distribución natural. Elkan (2001) demostró además que reequilibrar las clases equivale a desplazar el umbral de decisión: todo lo que SMOTE podría aportar a la **decisión** se obtiene moviendo los cortes, sin tocar la **probabilidad**. *Unfair Life* hace precisamente eso con la Ecuación 11.

### 3.3. Estado del arte

Van den Goorbergh et al. (2022) evaluaron sistemáticamente las correcciones de desequilibrio (sobremuestreo aleatorio, submuestreo y SMOTE) en modelos de riesgo y concluyeron que **empeoran la calibración sin mejorar la discriminación**, y que no deben usarse cuando la probabilidad estimada es el producto final. Es exactamente el caso de *Unfair Life*, donde la probabilidad se convierte en una moneda. Además, el desequilibrio del proyecto es moderado (24/76), muy lejos de los escenarios extremos (1/100 o menos) para los que SMOTE se diseñó (Chawla et al., 2002).

### 3.4. Cómo se trata el desequilibrio sin SMOTE

La estratificación de la partición, la precisión promedio como criterio de selección y la restricción del 95 % por clase en la política de decisión equilibran el trato de ambas clases sin alterar los datos (`03_IMPLEMENTATION_AND_BALANCE.md`, sección 1.4).

| Dónde se desarrolla | Documento y sección |
|---|---|
| Ablación de SMOTE en las cinco familias | `03_IMPLEMENTATION_AND_BALANCE.md`, secciones 1.2 y 1.3; `01_TRACEABILITY.md`, sección 2.3 |
| Equilibrio en la regla de decisión en lugar de en los datos | `03_IMPLEMENTATION_AND_BALANCE.md`, sección 1.4 |
| Ecuación de SMOTE y regla derivada | `02_MATHEMATICS.md`, Ecuación 1 |
| Evidencia gráfica | `02_MATHEMATICS.md`, Figuras 3, 4 y 12 |

---

## 4. El límite inferior del 94,87 % en el intervalo de confianza

### 4.1. Qué dice exactamente la cifra

La decisión «sin movilidad» acertó en 6051 de 6342 casos de prueba: **95,41 %**. Su intervalo de Wilson al 95 % es [94,87 %; 95,90 %]. El límite inferior queda 0,13 puntos por debajo del objetivo. Los intervalos de Clopper-Pearson (exacto) y de Jeffreys coinciden hasta la segunda cifra decimal: 94,87 % y 94,88 %.

### 4.2. Por qué es lo esperado y no un fallo

1. **La restricción se fijó en el 95 %, no por encima.** La Ecuación 11 exige que la estimación puntual alcance el 95 % en validación. Si la precisión verdadera de una decisión está cerca del 95 %, su intervalo de confianza **debe** contener el 95 %, y su límite inferior quedará por debajo en aproximadamente la mitad de las muestras posibles. Un límite inferior por encima del 95 % solo sería esperable si la precisión verdadera fuera claramente superior al objetivo.
2. **La evidencia favorece con claridad el cumplimiento.** Con una distribución *a priori* uniforme, la probabilidad posterior de que la precisión verdadera de «sin movilidad» sea al menos del 95 % es **0,93**; la de «movilidad», 0,81; y la de la exactitud conjunta sobre los 7270 casos decididos, **0,96**. Para esta última, la prueba binomial unilateral rechaza la hipótesis «exactitud inferior al 95 %» con $p = 0{,}041$.
3. **Exigir el límite inferior requeriría más datos de los que existen.** Para que el límite inferior de Wilson supere el 95 % con la precisión observada (95,41 %), harían falta unas **9930 decisiones** «sin movilidad»; el conjunto de prueba completo tiene 9758 personas.
4. **La alternativa conservadora se calculó.** Si la banda se eligiera exigiendo que el **límite inferior** de Wilson, y no la estimación puntual, alcance el 95 % en validación, la banda sería [0,21; 0,82]. En prueba, su cobertura bajaría del 74,50 % al 72,35 % (−2,15 puntos) y la exactitud esperada del juego, del 85,22 % al 84,96 %. Las dos precisiones subirían (96,65 % y 95,84 %), y el límite inferior de «sin movilidad» pasaría al 95,31 %. El proyecto conserva la banda puntual porque el criterio de la rúbrica es una precisión superior al 95 %, que ambas decisiones cumplen, y porque la versión conservadora sacrifica personas decididas sin mejorar la experiencia de juego. Ambas opciones quedan documentadas y la conservadora puede activarse sin reentrenar, cambiando solo la regla de `choose_band`.

### 4.3. Estado del arte

La distinción entre garantizar la estimación puntual y garantizar una cota de confianza es explícita en la literatura de clasificación selectiva. El algoritmo SGR de Geifman y El-Yaniv (2017) fija el umbral con una cota superior del riesgo, al precio de menos cobertura; la selección por estimación puntual es la variante estándar cuando el tamaño de la muestra de validación es grande. El intervalo de Wilson se prefiere al de Wald para proporciones cercanas a los extremos (Brown, Cai y DasGupta, 2001).

| Dónde se desarrolla | Documento y sección |
|---|---|
| Intervalos de Wilson, Clopper-Pearson y Jeffreys; posterior; banda conservadora | `05_ADVANCED_EVALUATION.md`, sección 6 (figura `precision_guarantee.png`) |
| Tabla de precisión con intervalos y nota de rigor | `03_IMPLEMENTATION_AND_BALANCE.md`, sección 2.2 |
| Intervalo de Wilson (Ecuación 10) | `02_MATHEMATICS.md`, Ecuación 10 |
| *Bootstrap* de 2000 remuestreos | `01_TRACEABILITY.md`, sección 3.2 |

---

## 5. La caída de precisión en 2018 y las profesiones feminizadas

### 5.1. Evidencia

Aplicado sin reentrenar a 298 686 personas de la ACS 2018, el modelo conserva buena parte de su capacidad de ordenar (ROC-AUC 0,833 en California y 0,802 en Nueva York), pero la precisión de «movilidad» cae al 70,5 % y al 63,5 %. La caída se concentra en las mujeres (59,3 % y 50,9 %, frente al 83,2 % y al 80,4 % de los hombres), que son el 73–77 % de los falsos positivos, y en las ocupaciones feminizadas (*Adm-clerical*: 44,4 % y 41,7 %).

El contraste decisivo es con 1994. En el conjunto de prueba del censo original, la precisión de «movilidad» para las mujeres fue del **97,69 %** (130 decisiones) y para los hombres del 95,36 %; la calibración por sexo se desvía menos de 1,1 puntos (`05_ADVANCED_EVALUATION.md`, sección 4). **El modelo no estaba sesgado contra las mujeres en su propia población**: lo que cambió entre 1994 y 2018 es la relación entre el perfil de una mujer y su ingreso.

### 5.2. Fundamento teórico

Moreno-Torres et al. (2012) distinguen tres desplazamientos entre la población de entrenamiento y la de uso: de la tasa base $P(Y)$, de los perfiles $P(X)$ y de la relación $P(Y \mid X)$. Solo el tercero, el *concept drift* (Gama et al., 2014), invalida a un clasificador calibrado, y no puede corregirse moviendo umbrales si altera el **orden** de los casos. Los datos de 2018 muestran exactamente esa firma: la tasa base apenas cambió en California (23,9 % frente a 23,5 %), mientras que en la cola alta de puntuaciones ningún corte devuelve la precisión al 95 % (máximo del 80 %). En la cola baja, en cambio, el orden se conserva y un corte de 0,08 (CA) o 0,07 (NY) restituye el 95 % sobre datos no vistos.

### 5.3. Estado del arte y lectura sustantiva

La explicación del patrón por sexo y ocupación está documentada en la economía laboral:

- **Brecha salarial persistente.** La brecha de género se redujo entre 1980 y 2010, pero se mantuvo con fuerza en la parte alta de la distribución y entre trabajadores con alta cualificación (Blau y Kahn, 2017; Goldin, 2014).
- **Devaluación de las ocupaciones feminizadas.** Cuando una ocupación se feminiza, su remuneración relativa desciende, incluso controlando por cualificación (Levanon, England y Allison, 2009). En 2018, *Prof-specialty* reúne docencia, enfermería y trabajo social, con alta escolaridad y fuerte presencia femenina; el modelo de 1994 asocia ese perfil a ingresos altos, y el mercado de 2018 no lo remunera en consecuencia.
- **El conjunto ACS como referencia.** Ding et al. (2021) crearon la formulación *ACSIncome* precisamente para estudiar cómo los modelos entrenados en *Adult* se degradan entre estados y años, y documentaron que la distribución de los grupos y su relación con el ingreso varían sustancialmente.

### 5.4. Por qué refuerza el proyecto en lugar de debilitarlo

1. **El sistema detecta el problema.** El protocolo fuera de distribución mide la caída, la atribuye a un subgrupo identificable y declara `TWO_SIDED_TARGET_NOT_REACHABLE` en lugar de forzar un umbral que no cumpliría.
2. **El sistema responde con la política correcta.** Recalibra solo la decisión cuyo orden se conserva y envía el resto a la moneda: la abstención como respuesta a la incertidumbre, coherente con la sección 2.
3. **El hallazgo es el argumento del juego.** Que una mujer con título universitario reciba en 2018 menos de lo que su perfil anticipaba es la desigualdad estructural que *Unfair Life* pretende mostrar, ahora medida fuera del censo con el que se construyó el modelo.

| Dónde se desarrolla | Documento y sección |
|---|---|
| Protocolo, métricas, recalibración asimétrica y desglose por sexo y ocupación | `01_TRACEABILITY.md`, sección 3.6 y prueba 8 |
| Justificación y conclusión | `03_IMPLEMENTATION_AND_BALANCE.md`, sección 4 |
| Evidencia gráfica | `02_MATHEMATICS.md`, Figura 16 |
| Calibración y precisión por sexo en 1994 | `05_ADVANCED_EVALUATION.md`, sección 4 |
| Desglose persistido | `training-pipeline/artifacts/extra_figures_report.json`, `concept_drift.subgroups` |

---

## Referencias

- Blau, F. D. y Kahn, L. M. (2017). The gender wage gap: Extent, trends, and explanations. *Journal of Economic Literature*, 55(3), 789-865.
- Broome, J. (1990). Fairness. *Proceedings of the Aristotelian Society*, 91, 87-101.
- Brown, L. D., Cai, T. T. y DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101-133.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46.
- Ding, F., Hardt, M., Miller, J. y Schmidt, L. (2021). Retiring Adult: New datasets for fair machine learning. *Advances in Neural Information Processing Systems*, 34.
- El-Yaniv, R. y Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research*, 11, 1605-1641.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973-978.
- Frank, R. H. (2016). *Success and Luck: Good Fortune and the Myth of Meritocracy*. Princeton University Press.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. y Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1-37.
- Geifman, Y. y El-Yaniv, R. (2017). Selective classification for deep neural networks. *Advances in Neural Information Processing Systems*, 30.
- Gneiting, T. y Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
- Goldin, C. (2014). A grand gender convergence: Its last chapter. *American Economic Review*, 104(4), 1091-1119.
- Hüllermeier, E. y Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. *Machine Learning*, 110(3), 457-506.
- Jencks, C., Smith, M., Acland, H., Bane, M. J., Cohen, D., Gintis, H., Heyns, B. y Michelson, S. (1972). *Inequality: A Reassessment of the Effect of Family and Schooling in America*. Basic Books.
- Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, 202-207.
- Levanon, A., England, P. y Allison, P. (2009). Occupational feminization and pay: Assessing causal dynamics using 1950-2000 U.S. census data. *Social Forces*, 88(2), 865-891.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. National Bureau of Economic Research.
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. y Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition*, 45(1), 521-530.
- Saerens, M., Latinne, P. y Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21-41.
- Salganik, M. J., Lundberg, I., Kindel, A. T., et al. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences*, 117(15), 8398-8403.
- Van den Goorbergh, R., van Smeden, M., Timmerman, D. y Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*, 29(9), 1525-1534.

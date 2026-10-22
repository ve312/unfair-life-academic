# Selección de Modelos y Estado del Arte

> **Ámbito.** Este documento justifica la selección de los modelos de aprendizaje automático de *Unfair Life* a partir de las características del problema y del estado del arte en clasificación sobre datos tabulares. Explica por qué se eligieron cuatro modelos principales y una referencia, qué esperaba la literatura de cada uno, qué se obtuvo y cómo esa evidencia condujo al ensamble desplegado. La formulación matemática de cada modelo se desarrolla en [`02_MATHEMATICS.md`](02_MATHEMATICS.md) (Ecuaciones 2 a 8), y la procedencia de cada cifra en [`01_TRACEABILITY.md`](01_TRACEABILITY.md).
>
> El objetivo es justificar el criterio de evaluación según el cual se seleccionan **cuatro modelos de aprendizaje automático acordes a la solución, con un estado del arte claro y soportado en un marco teórico**.

---

## 1. Caracterización del problema

La elección de un modelo no es neutral: depende de la forma de los datos y de lo que el sistema necesita hacer con sus predicciones. El problema de *Unfair Life* tiene seis rasgos que condicionan la selección:

| Rasgo del problema | Valor en el proyecto | Consecuencia para la selección |
|---|---|---|
| Tipo de datos | Tabulares: 5 variables numéricas y 8 categóricas (106 columnas tras la codificación *one-hot*) | Son preferibles los modelos diseñados para datos tabulares heterogéneos |
| Tarea | Clasificación binaria: superar o no 50 000 USD anuales de 1994 | Se necesitan modelos que estimen $P(Y = 1 \mid x)$ |
| Desequilibrio | 23,9 % de casos positivos | La métrica de selección debe ser robusta al desequilibrio (precisión promedio) |
| Uso de la salida | La probabilidad se convierte en una moneda ponderada en el juego | La **calibración** es un requisito, no una mejora opcional |
| Política de decisión | Clasificación selectiva con precisión ≥ 95 % en las decisiones absolutas | Se necesitan probabilidades que ordenen bien los casos extremos |
| Propósito social | El juego expone desigualdades estructurales | El modelo debe ser **explicable** rasgo por rasgo |

El problema exige, por tanto, modelos que **discriminen bien**, estimen **probabilidades calibradas** y sean **explicables**. Ningún modelo destaca por igual en las tres propiedades, lo que justifica comparar paradigmas distintos y, finalmente, combinarlos.

---

## 2. Estado del arte en clasificación tabular

### 2.1. Los cuatro paradigmas

La literatura agrupa los clasificadores para datos tabulares en cuatro grandes paradigmas, a los que se suma una referencia no paramétrica. El proyecto seleccionó un representante consolidado de cada uno:

1. **Modelos lineales probabilísticos.** La regresión logística (Cox, 1958) es la referencia clásica en ciencias sociales y en el análisis de ingresos. Sus coeficientes son interpretables como razones de probabilidades, y al maximizar la verosimilitud produce probabilidades bien calibradas cuando el modelo está bien especificado (Niculescu-Mizil y Caruana, 2005). Su limitación es estructural: no representa interacciones ni umbrales sin ingeniería manual de variables.
2. **Modelos de margen con kernels.** Las máquinas de vectores de soporte (Cortes y Vapnik, 1995) maximizan el margen de separación y, mediante kernels como el RBF, construyen fronteras no lineales. Su salida es una distancia a la frontera, no una probabilidad, por lo que requieren calibración posterior (Platt, 1999). Para conjuntos de decenas de miles de filas, la aproximación de Nyström (Williams y Seeger, 2001) evita el coste cuadrático del kernel exacto.
3. **Ensambles por *bagging*.** El bosque aleatorio (Breiman, 2001) promedia árboles descorrelacionados mediante muestras *bootstrap* y subconjuntos aleatorios de variables. Reduce la varianza de los árboles individuales y es robusto con poca configuración. Sus probabilidades, sin embargo, tienden a estar mal calibradas en los extremos (Niculescu-Mizil y Caruana, 2005).
4. **Ensambles por *boosting*.** El *gradient boosting* (Friedman, 2001) construye árboles en secuencia, cada uno corrigiendo el error de los anteriores mediante el gradiente de la pérdida. Sus implementaciones modernas usan información de segundo orden (Chen y Guestrin, 2016) y discretizan las variables en histogramas para entrenar con rapidez (Ke et al., 2017). La implementación usada, `HistGradientBoostingClassifier` de *scikit-learn*, sigue este último diseño.
5. **Referencia no paramétrica.** Los k vecinos más cercanos (Cover y Hart, 1967) estiman la probabilidad localmente, sin aprender parámetros. Sirven como cota inferior: un modelo que no los supera no ha aprendido estructura útil.

### 2.2. Qué dice la evidencia reciente

Un estudio comparativo de referencia (Grinsztajn, Oyallon y Varoquaux, 2022) evaluó modelos sobre decenas de conjuntos tabulares y concluyó que los ensambles de árboles, y en particular el *gradient boosting*, siguen superando en general a las redes neuronales profundas en datos tabulares de tamaño medio. Las razones son tres: son robustos a variables poco informativas, capturan funciones irregulares y no necesitan grandes volúmenes de datos. El conjunto *Adult Income* (Becker y Kohavi, 1996; Kohavi, 1996), con unos 49 000 registros y variables heterogéneas, encaja exactamente en ese escenario. Por eso el proyecto no incluyó redes neuronales entre los candidatos: el estado del arte no las favorece para este tipo de datos, y su menor interpretabilidad iría en contra de un requisito central.

### 2.3. Combinación de modelos: *stacked generalization*

La *stacked generalization* (Wolpert, 1992; Breiman, 1996) combina varios modelos base mediante un meta-modelo entrenado sobre sus predicciones fuera de pliegue. Su justificación teórica es que modelos con sesgos inductivos distintos cometen errores parcialmente independientes, y una combinación aprendida puede aprovechar esa complementariedad. La literatura recomienda dos cosas: que el meta-modelo sea simple, normalmente una regresión logística regularizada, para no sobreajustar, y que se entrene con predicciones fuera de pliegue, para no premiar el sobreajuste de los modelos base.

### 2.4. Requisitos complementarios del estado del arte

La selección se apoyó además en cuatro líneas de investigación:

- **Calibración.** Una probabilidad útil debe coincidir con la frecuencia observada (Niculescu-Mizil y Caruana, 2005). El proyecto la mide con la puntuación de Brier y con el error de calibración esperado (ECE).
- **Desequilibrio de clases.** SMOTE (Chawla et al., 2002) es la técnica estándar de sobremuestreo. La literatura advierte que reequilibrar artificialmente las clases cambia la probabilidad *a priori* con la que aprende el modelo y, por tanto, distorsiona sus probabilidades. El proyecto lo comprobó empíricamente (`02_MATHEMATICS.md`, Ecuación 1).
- **Explicabilidad.** Los valores de Shapley (Lundberg y Lee, 2017), calculables de forma exacta en modelos de árboles (Lundberg et al., 2020), son el estándar para atribuir una predicción a sus variables.
- **Abstención.** La clasificación con opción de rechazo (Chow, 1970), formulada modernamente como clasificación selectiva (Geifman y El-Yaniv, 2017), permite que el modelo decida solo cuando su confianza alcanza un objetivo de error.

---

## 3. Modelos seleccionados

Se seleccionaron **cuatro modelos principales**, uno por paradigma, más la referencia no paramétrica. Para cada uno se registró antes del entrenamiento qué esperaba la literatura, y después se contrastó con lo obtenido en el conjunto de validación (`model_comparison.json`, distribución natural):

| Modelo (ecuación en 02) | Paradigma | Expectativa según la literatura | Precisión promedio | ROC-AUC | Brier | ECE | Tiempo |
|---|---|---|---:|---:|---:|---:|---:|
| Regresión logística (2) | Lineal | Buena calibración, discriminación limitada por la linealidad | 0,768 | 0,906 | 0,102 | **0,0080** | 0,4 s |
| HistGradientBoosting (3) | *Boosting* | Mejor discriminación en datos tabulares | **0,830** | **0,928** | **0,088** | 0,0096 | 3,9 s |
| Bosque aleatorio (4) | *Bagging* | Discriminación alta, calibración débil en los extremos | 0,766 | 0,903 | 0,103 | 0,0193 | 3,6 s |
| SVM RBF calibrada (5) | Márgenes | Discriminación competitiva, coste alto, calibración dependiente de Platt | 0,757 | 0,901 | 0,105 | 0,0234 | 17,5 s |
| k-NN (6), referencia | No paramétrico | Cota inferior en alta dimensión | 0,631 | 0,847 | 0,129 | 0,0442 | 0,1 s |

**Lectura de los resultados.** Las cinco expectativas se confirmaron:

- El *gradient boosting* fue el mejor en discriminación y en Brier, como predice la evidencia sobre datos tabulares.
- La regresión logística fue el modelo mejor calibrado, como predice su formulación de máxima verosimilitud, pero quedó muy por debajo en discriminación (0,768 frente a 0,830).
- El bosque aleatorio y la SVM calibrada alcanzaron una discriminación intermedia y una calibración peor que la de los dos anteriores.
- k-NN quedó último en todas las métricas, como corresponde a un método local en 106 dimensiones.

Los cinco modelos se entrenaron también con la copia balanceada por SMOTE, y en los cinco el balanceo empeoró la precisión promedio y la calibración (`01_TRACEABILITY.md`, sección 2.3). Una sexta variante, el *gradient boosting* con pesos de clase, alcanzó la misma discriminación (0,830) con una calibración diez veces peor (ECE de 0,100), lo que confirma que cualquier reequilibrio artificial de las clases distorsiona las probabilidades.

---

## 4. Del modelo individual al ensamble

Los resultados plantean un dilema que ningún modelo individual resuelve: el *gradient boosting* discrimina mejor y la regresión logística calibra mejor. La *stacked generalization* (sección 2.3) es la respuesta del estado del arte a ese dilema. El ensamble desplegado combina los dos modelos de **sesgos inductivos más complementarios**: uno flexible y no lineal, de bajo sesgo, y otro rígido y lineal, de baja varianza. La combinación la hace un meta-learner logístico sobre sus *log-odds* fuera de pliegue (`02_MATHEMATICS.md`, Ecuaciones 7 y 8).

El ensamble mantuvo la discriminación del mejor modelo (precisión promedio de 0,830) y redujo el ECE de 0,0096 a **0,0073**, un 24 % menos, sin perder la explicabilidad exacta. Para decidir qué modelos componen la primera capa se realizó un estudio de ablación:

| Primera capa | Precisión promedio | ECE | Peso del modelo añadido | Explicabilidad exacta | Tamaño |
|---|---:|---:|---|---|---:|
| **HGB + regresión logística (desplegado)** | **0,830** | **0,0073** | — | **Sí** | **0,43 MB** |
| HGB + RL + bosque aleatorio | 0,830 | 0,0079 | RF: 0,078 | No | 5,96 MB |
| HGB + RL + bosque aleatorio + k-NN | 0,830 | 0,0079 | k-NN: −0,0005 | No | 6,91 MB |

Añadir más modelos no mejoró la discriminación, empeoró ligeramente la calibración, rompió la explicabilidad exacta y multiplicó por catorce el tamaño del modelo. La SVM no se incluyó en la ablación porque, además de quedar por debajo en todas las métricas, su coste de entrenamiento se habría multiplicado por los cinco pliegues del *stacking*. Por el principio de parsimonia, se eligió la primera capa más simple entre las de rendimiento equivalente.

---

## 5. Conclusión

La selección de modelos de *Unfair Life* cumple el criterio de evaluación por tres motivos:

1. **Cuatro modelos acordes a la solución.** Se seleccionó un representante consolidado de cada uno de los cuatro paradigmas de clasificación tabular (lineal, márgenes, *bagging* y *boosting*), más una referencia no paramétrica. La selección responde a los requisitos del problema (discriminación, calibración y explicabilidad), no a una preferencia por un algoritmo concreto.
2. **Estado del arte claro.** Cada elección se apoya en la literatura de referencia, incluida la evidencia reciente de que los ensambles de árboles dominan en datos tabulares de tamaño medio, lo que también justifica no usar redes neuronales.
3. **Marco teórico verificado.** Las expectativas teóricas se registraron antes de entrenar y se contrastaron con resultados medidos sobre datos no vistos. Todas se confirmaron, y el ensamble final es la respuesta que el propio estado del arte propone al dilema entre discriminación y calibración que revelaron los datos.

---

## Referencias

- Becker, B. y Kohavi, R. (1996). *Adult* [conjunto de datos]. UCI Machine Learning Repository.
- Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49-64.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46.
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Cover, T. y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
- Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), 215-232.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Geifman, Y. y El-Yaniv, R. (2017). Selective classification for deep neural networks. *Advances in Neural Information Processing Systems*, 30.
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35 (Datasets and Benchmarks Track).
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. y Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, 202-207.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. y Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
- Niculescu-Mizil, A. y Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En *Advances in Large Margin Classifiers* (pp. 61-74). MIT Press.
- Williams, C. K. I. y Seeger, M. (2001). Using the Nyström method to speed up kernel machines. *Advances in Neural Information Processing Systems*, 13.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.

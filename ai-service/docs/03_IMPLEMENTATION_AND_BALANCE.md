# Implementación, Balanceo y Precisión del Modelo

> **Ámbito.** Este documento cierra la documentación del servicio de inteligencia artificial de *Unfair Life*. Justifica tres aspectos: el cumplimiento del criterio de volumen de datos y balanceo de clases, el cumplimiento del criterio de precisión superior al 95 %, y las diez funcionalidades implementadas entre el servicio de inferencia (FastAPI) y el *Backend for Frontend* (NestJS). La procedencia de cada cifra se documenta en [`01_TRACEABILITY.md`](01_TRACEABILITY.md) y su fundamento matemático, en [`02_MATHEMATICS.md`](02_MATHEMATICS.md).

---

## 1. Criterio de dataset y balanceo

### 1.1. Volumen de datos: 48 790 registros limpios

El criterio exige un conjunto de datos de más de 35 000 registros. El proyecto utiliza el censo *Adult Income* de 1994, cuyo fichero original contiene **48 842 registros**. Tras el proceso de limpieza documentado en `cleaning_report.json`, que eliminó 52 registros duplicados exactos y conservó los valores ausentes bajo la categoría explícita *Unknown* en lugar de descartar filas, el conjunto final consta de **48 790 registros válidos**.

Esta cifra supera el mínimo exigido en un **39,4 %**. El cumplimiento no depende de una verificación manual: el *script* `02_balancer.py` comprueba de forma automática que el número de registros no sea inferior a 35 000 y detiene el *pipeline* en caso contrario. El resultado de esa comprobación queda registrado en `balancing_report.json` (`"total_records": 48790`, `"min_required_records": 35000`).

El volumen disponible permitió además una partición estratificada en tres subconjuntos independientes: 29 274 registros de entrenamiento, 9758 de validación y 9758 de prueba. Cada uno es suficientemente grande para producir estimaciones con intervalos de confianza estrechos.

### 1.2. Balanceo aplicado, evaluado y rechazado

La distribución natural del conjunto presenta un desequilibrio moderado: el **76,1 %** de las personas no supera los 50 000 USD anuales y el **23,9 %** sí lo hace. En cumplimiento del criterio, este desequilibrio se abordó explícitamente con la técnica SMOTENC, una variante de SMOTE adecuada para datos con rasgos numéricos y categóricos. Con ella se generaron **15 256 registros sintéticos** que llevaron el conjunto de entrenamiento a una proporción exacta de 50/50.

El balanceo no se adoptó por defecto, sino que se sometió a un **estudio de ablación**. Las cinco familias de algoritmos se entrenaron dos veces, una con la distribución natural y otra con la balanceada, y todas se evaluaron sobre el mismo conjunto de validación compuesto exclusivamente por personas reales. Los resultados, registrados en `model_comparison.json`, son concluyentes:

| Familia | Precisión promedio (natural) | Precisión promedio (SMOTE) | ECE (natural) | ECE (SMOTE) |
|---|---:|---:|---:|---:|
| HistGradientBoosting | **0,830** | 0,797 | **0,010** | 0,078 |
| Regresión logística | **0,768** | 0,759 | **0,008** | 0,112 |
| Bosque aleatorio | **0,766** | 0,716 | **0,019** | 0,066 |
| SVM (RBF, calibrada) | **0,757** | 0,736 | **0,023** | 0,105 |
| K vecinos más cercanos | **0,631** | 0,597 | **0,044** | 0,121 |

La evidencia apunta en una sola dirección. En **las cinco familias sin excepción**, el entrenamiento con SMOTE redujo la capacidad discriminativa del modelo (la precisión promedio) y multiplicó su error de calibración esperado (ECE), en algunos casos por más de diez.

### 1.3. Por qué la calibración es innegociable

El deterioro de la calibración es la razón decisiva del rechazo, y se explica por la naturaleza del juego. *Unfair Life* no utiliza el modelo como un simple clasificador binario, sino como el **motor estocástico** de la suerte sistémica. Cuando el perfil de un personaje cae en la zona de incertidumbre, su destino se resuelve con una moneda ponderada exactamente por la probabilidad que estima el modelo. Para que esa mecánica sea justa y fiel a la realidad, una probabilidad del 60 % debe corresponder a personas que, en el mundo real, alcanzan la movilidad aproximadamente en el 60 % de los casos.

Forzar una proporción artificial de 50/50 rompe esa correspondencia. Al ver tantos casos positivos como negativos durante el entrenamiento, el modelo aprende que la movilidad es mucho más frecuente de lo que realmente es. La predicción media de los modelos entrenados con SMOTE ascendió a **0,330**, cuando la tasa real de movilidad es **0,239**. En el contexto del juego, esto equivaldría a otorgar movilidad social a personajes que en la realidad no la tendrían. El resultado sería una simulación sistemáticamente optimista que contradiría el propósito mismo del proyecto: exponer la desigualdad estructural.

Por estas razones, **el modelo desplegado se entrenó con la distribución natural 76/24**. Esto incluye a sus tres componentes: el modelo desplegado es un ensamble Stacking (véase `02_MATHEMATICS.md`, Ecuación 8, y la selección de sus componentes en `04_MODEL_SELECTION.md`), y tanto sus dos modelos base como su segunda capa se ajustaron exclusivamente sobre datos sin balancear. El ensamble se construyó, además, para preservar la calibración: combina sus modelos en escala logit y redujo el error de calibración esperado en validación de 0,0096 a 0,0073. SMOTE se aplicó, se midió y se documentó, de modo que el criterio de balanceo se aborda de forma rigurosa. Su rechazo es una decisión metodológica fundamentada en evidencia empírica, no una omisión.

---

## 2. Criterio de precisión (≥ 95 %) frente a exactitud global

### 2.1. Dos métricas distintas para dos preguntas distintas

Para evaluar este criterio con rigor conviene distinguir dos medidas que con frecuencia se confunden:

- La **precisión** responde a la pregunta: cuando el modelo afirma algo, ¿con qué frecuencia acierta?
- La **exactitud global** responde a otra: sobre todos los casos, incluidos aquellos en que el modelo deliberadamente no decide, ¿con qué frecuencia coincide el desenlace con la realidad?

### 2.2. Precisión superior al 95 % en la zona determinista

El ensamble Stacking opera bajo una política de **clasificación selectiva**: solo emite un veredicto absoluto cuando su confianza es suficiente, es decir, cuando la probabilidad estimada es como máximo 0,23 o al menos 0,78. Esta **zona determinista** cubre el **74,50 %** de los casos del conjunto de prueba, que el modelo no vio durante el entrenamiento ni durante la selección de sus parámetros. Los resultados en esa zona son los siguientes:

| Decisión determinista | Precisión en prueba | Intervalo de confianza al 95 % (Wilson) |
|---|---:|---:|
| «Movilidad» (probabilidad ≥ 0,78) | **95,69 %** | 94,18 % – 96,82 % |
| «Sin movilidad» (probabilidad ≤ 0,23) | **95,41 %** | 94,87 % – 95,90 % |
| Exactitud conjunta sobre los casos decididos | **95,45 %** | 94,94 % – 95,90 % |

**Ambas** decisiones absolutas del modelo superan el umbral del 95 %, no solo una de ellas, y también lo supera la exactitud combinada sobre todos los casos decididos. La banda se fijó con un criterio declarado de antemano, utilizando solo el conjunto de validación: maximizar la cobertura con la condición de que las dos decisiones alcanzaran al menos un 95 % de precisión, cada una respaldada por un mínimo de 30 casos. Que los resultados en el conjunto de prueba reproduzcan ese nivel demuestra que la política generaliza y no está sobreajustada.

### 2.3. La moneda ponderada en la zona de incertidumbre

En el 25,50 % restante de los casos, con probabilidades entre 0,23 y 0,78, el perfil censal no determina el destino de la persona con suficiente confianza. Allí, por diseño del juego, el modelo no emite un veredicto absoluto: el BFF lanza una **moneda ponderada** por la probabilidad estimada, mecanismo que el juego denomina **suerte sistémica**. Esta zona no es un fallo del modelo, sino el reconocimiento explícito de que, para una parte de la sociedad, el desenlace depende de factores que ningún dato censal recoge.

Al incorporar estos lanzamientos aleatorios, la **exactitud global esperada** sobre la totalidad de los jugadores es del **85,22 %**. El valor se ha confirmado mediante 2000 simulaciones independientes del juego, cuyos resultados se sitúan entre el 84,74 % y el 85,68 %. Esta cifra es inferior a la precisión determinista por una razón estructural: dentro de la banda, el resultado lo decide intencionadamente el azar ponderado y no el veredicto del modelo.

### 2.4. Conclusión sobre el criterio

El criterio de precisión igual o superior al 95 % **se cumple estrictamente allí donde el modelo toma decisiones absolutas**: 95,69 % en la decisión «movilidad», 95,41 % en «sin movilidad» y 95,45 % de exactitud conjunta, todo ello sobre datos no vistos y con intervalos de confianza que lo respaldan. La exactitud global del 85,22 % no contradice este resultado, sino que lo complementa. Cuantifica el coste deliberado de una decisión de diseño que se niega a fabricar certezas donde no las hay. Un modelo que forzara una decisión determinista sobre todos los casos podría exhibir una única cifra de exactitud, pero lo haría a costa de ocultar su propia incertidumbre.

---

## 3. Las diez funcionalidades de la inteligencia artificial

Las funcionalidades siguientes están implementadas y operativas en el servicio de inferencia (`ai-service/inference-api`), en el BFF (`bff-api`) o en el *pipeline* de entrenamiento (`ai-service/training-pipeline`), según se indica en cada caso.

### 3.1. Endpoint de predicción asegurada (`POST /predict` con autenticación)

**Ubicación:** servicio de inferencia (`controllers.py`, `security.py`) y BFF (`fast-api-predictor.adapter.ts`).

El servicio de inferencia expone el endpoint `POST /predict`. Recibe el perfil censal de un personaje y devuelve la probabilidad de movilidad, la zona de decisión y los límites de la banda. El endpoint solo atiende peticiones que incluyan la cabecera `X-BFF-Token` con un secreto compartido, configurado en la variable de entorno `BFF_SECRET_TOKEN`. La comparación del secreto tarda lo mismo acierte o falle, de modo que el tiempo de respuesta no revela información sobre él. Si falta la cabecera o es incorrecta, la respuesta es `403 Forbidden`. En local, el servicio lee el secreto de `inference-api/.env` (cargado por `main.py`, sin sobrescribir las variables ya definidas en el entorno), igual que el BFF y el CMS leen los suyos; el valor debe coincidir en los tres. Si el BFF recibe un `401` o `403`, lo registra indicando que el secreto no coincide, y el cliente ve un `502`. En producción (Render), si el secreto no está configurado, el servicio rechaza todas las peticiones en lugar de aceptar un valor por defecto conocido. Así, el modelo solo es accesible a través del BFF y nunca de forma directa desde Internet.

### 3.2. Endpoint de salud y monitoreo (`GET /health`)

**Ubicación:** servicio de inferencia y BFF.

El endpoint `GET /health` del servicio de inferencia confirma que el modelo está cargado y devuelve su identidad y su política de decisión: el nombre del modelo, los límites de la banda (0,23 y 0,78) y el umbral determinista (0,7782). En el despliegue actual, el nombre que informa es `Stacking (HGB + LR -> LR) [natural]`. Responde con `503` si el modelo no está disponible. Con esta información, una plataforma de monitoreo puede verificar no solo que el servicio está activo, sino también qué versión del modelo está sirviendo. El BFF dispone de su propio `GET /health`, que responde de inmediato sin consultar a sus dependencias, de forma que un servicio dependiente en reposo no haga parecer caído al BFF.

### 3.3. Validación estricta de esquemas (Pydantic y DTO)

**Ubicación:** servicio de inferencia (`schemas.py`, entidad de dominio `Character`) y BFF (`validation-pipe.ts`, DTO).

Los datos de entrada se validan en tres capas sucesivas:

1. **BFF.** La petición del cliente se valida con DTO de NestJS configurados para rechazar cualquier campo no declarado y sin conversiones implícitas de tipo.
2. **Servicio de inferencia.** El esquema Pydantic prohíbe campos adicionales y restringe los rangos numéricos (por ejemplo, edad entre 17 y 90 años y horas semanales entre 1 y 99).
3. **Dominio.** La entidad `Character` aplica sus propias reglas de negocio antes de que el perfil llegue al modelo.

Un perfil malformado se rechaza con un error descriptivo (`400` o `422`) y nunca produce una predicción silenciosamente errónea.

### 3.4. Seguridad *fail-fast* ante la falta de artefactos

**Ubicación:** servicio de inferencia (`fast_api_app.py`, `sklearn_predictor.py`).

Los artefactos del modelo se cargan una única vez, durante el arranque del servicio. Antes de aceptar peticiones, el servicio comprueba que existan todos los ficheros necesarios y que sean coherentes entre sí:

- que el modelo ofrezca probabilidades e incluya la clase positiva;
- que haya sido entrenado con las columnas exactas declaradas en `feature_schema.json`;
- que el imputador y el codificador reproduzcan ese mismo esquema;
- que la política de decisión de `traceability_report.json` sea válida.

Si cualquiera de estas comprobaciones falla, **el servicio se niega a arrancar**. Es preferible un despliegue fallido y visible a un servicio que responde con predicciones de un modelo incompleto o incoherente.

### 3.5. Alineación de la moneda ponderada en el juego (validación estadística)

**Ubicación:** BFF (`systemic-luck.ts` y sus pruebas) y *pipeline* (`04_evaluator.py`).

La moneda ponderada se resuelve con una regla simple: hay movilidad si un número aleatorio uniforme entre 0 y 1 es menor que la probabilidad del modelo. La alineación de esta mecánica con el modelo se valida en dos niveles complementarios:

- **Pruebas unitarias del BFF.** Verifican que la regla respeta exactamente su límite. Con una probabilidad de 0,6, una tirada de 0,599 otorga movilidad y una de 0,600 no. También comprueban que fuera de la banda nunca se lanza la moneda y que se rechaza cualquier valor aleatorio fuera de rango.
- **Simulación estadística del *pipeline*.** El evaluador simula 2000 partidas completas sobre el conjunto de prueba con la semilla 42 y obtiene una exactitud del juego entre el 84,74 % y el 85,68 %. El rango es coherente con el valor esperado analítico del 85,22 %.

En conjunto, ambos niveles demuestran que la frecuencia de movilidad que experimentan los jugadores reproduce la probabilidad calibrada del modelo.

### 3.6. Deflación económica histórica (ajuste por IPC a dólares de 1994)

**Ubicación:** BFF (`consumer-price-index.ts`, `inflation-adjuster.ts`).

El modelo fue entrenado con dólares de 1994, mientras que las historias del juego transcurren en otros años. Antes de consultar al modelo, el BFF convierte las ganancias y pérdidas de capital del personaje a dólares de 1994 con el índice de precios al consumidor oficial (IPC-U, serie CUUR0000SA0 de la Oficina de Estadísticas Laborales de EE. UU.). Para los años posteriores a 2024, último año publicado, aplica una proyección documentada del 2,5 % anual. La misma conversión expresa el umbral de movilidad en la moneda del año narrado: equivale a 111 191 USD en 2026. Sin este ajuste, el modelo sobrestimaría sistemáticamente la riqueza de los personajes contemporáneos.

### 3.7. Motor de decisión selectiva por banda de incertidumbre

**Ubicación:** servicio de inferencia (entidad `DecisionPolicy`) y BFF (`systemic-luck.ts`).

El servicio de inferencia clasifica cada predicción en una de tres zonas: «movilidad», «sin movilidad» o «incierta». Los límites de la banda no están escritos en el código: se leen en cada arranque de `traceability_report.json`, donde quedaron fijados por el evaluador a partir del conjunto de validación. El BFF aplica la consecuencia de cada zona. En las dos zonas deterministas, el desenlace del personaje es fijo y se garantiza una precisión superior al 95 %. En la zona incierta, el destino se resuelve con la moneda ponderada. Este motor materializa en el juego la política de clasificación selectiva descrita en la sección 2.

### 3.8. Auditoría y trazabilidad de tiradas en registros

**Ubicación:** BFF (`resolve-decision.service.ts`) y servicio de inferencia (`controllers.py`, `security.py`).

Cada comprobación de movilidad queda registrada en los *logs* del BFF con todos los datos necesarios para reconstruirla: la escena y la decisión implicadas, el año de la historia, la probabilidad devuelta por el modelo, la zona asignada, quién decidió (el modelo o la suerte), el valor exacto de la tirada aleatoria cuando la hubo y el resultado final. Desde la incorporación de la inyección de contexto dinámico, el registro incluye también los modificadores ocultos que la decisión aplicó al personaje, cada uno con su valor anterior y posterior y con la indicación de si fue acotado al dominio de entrenamiento, y el perfil exacto enviado a la IA tras la mutación y la deflación. Cada predicción del juego real puede así reproducirse fuera de línea con los artefactos del *pipeline*. El servicio de inferencia, por su parte, registra cada predicción servida y cada intento de acceso rechazado por falta del secreto compartido. Gracias a ello, cualquier desenlace de una partida puede auditarse a posteriori y verificarse que siguió la política documentada.

### 3.9. Control de acceso por orígenes cruzados (CORS estricto)

**Ubicación:** servicio de inferencia (`cors.py`) y BFF (`http.config.ts`).

Ambos servicios restringen qué orígenes de navegador pueden invocarlos. El BFF solo admite el origen del cliente del juego desplegado (`FRONTEND_URL`) y, opcionalmente, una lista explícita de orígenes adicionales. En producción se niega a arrancar si `FRONTEND_URL` no está configurado, de modo que CORS nunca puede desplegarse abierto ni vacío por error. Los orígenes de desarrollo local solo se permiten fuera de producción. El servicio de inferencia admite únicamente el origen del BFF y valida que cada entrada sea un origen sintácticamente correcto. CORS actúa como una capa de defensa en navegador que complementa, sin sustituirla, la autenticación por secreto compartido de la funcionalidad 3.1.

### 3.10. Serialización de la explicabilidad (valores SHAP)

**Ubicación:** *pipeline* de entrenamiento (`notebooks/01_math_and_eda.ipynb`) y artefactos (`artifacts/figures/`, `artifacts/latex/`).

Los valores SHAP se calculan de forma exacta sobre el **mismo fichero `best_model.joblib`** que sirve la API, para una muestra de 2000 personas del conjunto de prueba. Como el modelo es un ensamble Stacking, su explicación aprovecha el axioma de linealidad de Shapley. La contribución del *gradient boosting* se obtiene con `TreeExplainer` intervencional, la de la regresión logística con su fórmula cerrada, y ambas se combinan con los pesos de la segunda capa. Antes de serializarlos se verifica la propiedad de exactitud local: la suma de las contribuciones reproduce la salida del ensamble con una discrepancia máxima de $7{,}6 \times 10^{-8}$. Los resultados se persisten como artefactos versionados en el repositorio:

- la tabla de importancia media por rasgo (`table_trait_importance.tex`);
- el resumen global (`shap_beeswarm.png`);
- el gráfico de importancia agrupada (`shap_trait_importance.png`);
- las curvas de dependencia (`shap_dependence.png`).

La explicabilidad se genera, por tanto, fuera de línea y queda documentada junto al modelo. La API de producción no calcula valores SHAP por petición, lo que mantiene bajo su tiempo de respuesta. Estos artefactos revelan, entre otros hallazgos, que el modelo ha absorbido desigualdades presentes en el censo de 1994, como el efecto negativo de ser mujer sobre la probabilidad de movilidad: `sex_Female` es la sexta columna más influyente del ensamble. Esa evidencia sustenta el marco teórico del juego (véase `02_MATHEMATICS.md`, Figura 9).

---

## 4. Robustez ante *Concept Drift*: la ACS 2018 de California y Nueva York

### 4.1. El criterio del 95 % fuera de la distribución de entrenamiento

La sección 2 demuestra la precisión sobre personas del mismo censo de 1994. Para comprobar si la garantía sobrevive al paso del tiempo, el *script* `05_ood_testing.py` aplicó el modelo desplegado, **sin reentrenarlo ni recalibrarlo**, a 195 665 personas de California y 103 021 de Nueva York de la *American Community Survey* de 2018, con los ingresos deflactados a dólares de 1994 por la misma tabla IPC-U del BFF (funcionalidad 3.6). El protocolo completo y sus cifras se documentan en `01_TRACEABILITY.md`, sección 3.6.

El modelo conserva buena parte de su capacidad de ordenar (ROC-AUC de 0,833 en CA y 0,802 en NY, frente a 0,931 en 1994), pero la relación entre el perfil y el ingreso cambió: es un caso de *concept drift*. Con la banda congelada de 1994, ninguna de las dos decisiones deterministas mantiene el 95 %, y su caída es muy desigual:

| Decisión (banda [0,23; 0,78]) | 1994 | CA 2018 | NY 2018 |
|---|---:|---:|---:|
| «Movilidad» | 95,69 % | 70,52 % | 63,47 % |
| «Sin movilidad» | 95,41 % | 91,42 % | 91,38 % |

### 4.2. La recalibración asimétrica recupera el 95 % en «sin movilidad»

La respuesta estándar a una deriva de distribución, cuando no se dispone de etiquetas para reentrenar, es la **recalibración de umbrales**: se conserva el orden aprendido y solo se desplazan los cortes. Se aplicó con la regla exacta del evaluador (`04_evaluator.choose_band`), estimando los cortes sobre una mitad estratificada de la muestra y midiéndolos en la otra mitad, que no participó en la búsqueda.

La búsqueda bilateral fracasa: ningún corte superior alcanza el 95 % en «movilidad» (el máximo de la rejilla es 80,12 % en CA y 79,55 % en NY). El sistema adopta entonces una **política asimétrica**: mantiene la decisión determinista «sin movilidad» con un corte más estricto y envía todo lo demás a la moneda ponderada.

| Mitad reservada | Corte «sin movilidad» | Cobertura | Precisión (IC 95 % Wilson) | Exactitud esperada del juego |
|---|---|---:|---:|---:|
| California | $p \le 0{,}08$ | 45,01 % | **95,34 %** (95,14 – 95,53) | 75,67 % |
| Nueva York | $p \le 0{,}07$ | 40,09 % | **95,81 %** (95,53 – 96,08) | 73,89 % |

En ambos estados, la precisión de la decisión «sin movilidad» vuelve a superar el 95 %, con el límite inferior del intervalo de confianza también por encima. El coste es cobertura: el modelo decide por sí solo en el 40–45 % de los casos y cede el resto a la suerte sistémica. La exactitud del juego apenas varía respecto a la banda congelada (−1,6 puntos en los dos estados), porque los casos que salen de la zona determinista pasan a una moneda bien ponderada.

### 4.3. El sesgo en profesiones feminizadas justifica la caída en «movilidad»

La pérdida de precisión en «movilidad» no se reparte de forma uniforme. Al desglosar las predicciones $p \ge 0{,}78$ por sexo y por grupo ocupacional:

- La precisión de «movilidad» es del **59,3 %** para las mujeres de California y del **50,9 %** para las de Nueva York, frente al 83,2 % y al 80,4 % de los hombres.
- Las mujeres son el 47–49 % de la población evaluada, pero el **73,2 % (CA) y el 77,2 % (NY)** de los falsos positivos de «movilidad».
- En *Adm-clerical*, el grupo más feminizado (71–73 % de mujeres), la precisión cae al 44,4 % y al 41,7 %. En *Prof-specialty* (53–59 % de mujeres), que concentra alrededor del 60 % de todas las decisiones «movilidad», queda en el 65,1 % y el 56,8 %. En *Exec-managerial*, menos feminizado, se mantiene en el 80,8 % y el 75,3 %.
- La decisión «sin movilidad», en cambio, es prácticamente igual para ambos sexos (91,7 % frente a 91,2 % en CA, y 91,7 % frente a 91,1 % en NY).

La interpretación es coherente con la brecha salarial de las profesiones feminizadas. En 2018, *Prof-specialty* y *Adm-clerical* reúnen ocupaciones de escolaridad media o alta con fuerte presencia femenina (docencia, enfermería, trabajo social, administración) cuyo salario queda por debajo de los 84 719 USD que equivalen al umbral de 1994. El modelo de 1994 asigna a esos perfiles puntuaciones altas por su educación y su ocupación, pero el mercado de 2018 no los remunera en consecuencia, y menos aún si quien ocupa el puesto es una mujer. Como el error está en el **orden** de la cola alta y no en su calibración, ningún umbral lo corrige: la abstención en «movilidad» es la única política que respeta el criterio del 95 %.

Este hallazgo refuerza el marco teórico del proyecto en lugar de debilitarlo. La explicabilidad SHAP (funcionalidad 3.10) ya mostró que el modelo absorbió la penalización por ser mujer del censo de 1994; la prueba de 2018 muestra que, fuera de ese censo, la desigualdad de género reaparece como el principal motivo por el que el modelo promete movilidades que no se cumplen.

> El desglose por sexo y ocupación se calculó con los mismos registros, recodificación y artefactos que `ood_results_*.json`, pero todavía no se persiste en esos ficheros.

---

## 5. Conclusión

La implementación del servicio de inteligencia artificial de *Unfair Life* satisface los criterios evaluados con evidencia verificable:

1. **Dataset y balanceo.** El conjunto supera el mínimo de 35 000 registros con 48 790 registros limpios, verificado automáticamente. El balanceo con SMOTE se aplicó y se evaluó con un estudio de ablación en cinco familias de algoritmos. Se rechazó porque degradaba la calibración de las probabilidades, requisito indispensable del motor estocástico del juego.
2. **Precisión.** Las dos decisiones absolutas del ensamble Stacking superan el 95 % de precisión sobre datos no vistos (95,69 % y 95,41 %), con una cobertura del 74,50 %. La exactitud global del 85,22 % refleja de forma transparente la incorporación deliberada de la suerte sistémica.
3. **Funcionalidades.** Diez funcionalidades distribuidas entre el servicio de inferencia, el BFF y el *pipeline* abarcan la seguridad, la validación, la robustez, la fidelidad estadística, la corrección económica, la auditoría y la explicabilidad del modelo.
4. ***Concept drift*.** Sobre la ACS 2018, la recalibración asimétrica del corte inferior restituye el 95 % en «sin movilidad» (95,34 % en California y 95,81 % en Nueva York). La decisión «movilidad» no puede recuperarse con ningún umbral, porque su error se concentra en mujeres y profesiones feminizadas, cuyo salario de 2018 no sigue el orden aprendido en 1994; en esa zona el sistema se abstiene y decide la moneda ponderada.

En conjunto, estos tres documentos (trazabilidad, matemáticas e implementación) constituyen la documentación técnica completa del componente de inteligencia artificial del proyecto.

# Propuestas de mejoras funcionales

A partir de la revisión se identifican varias mejoras funcionales que podrían implementarse:

## Aprovechar el GPU: 
Actualmente el entrenamiento usa PyTorch y podría acelerarse notablemente con CUDA. Se sugiere asegurar que los tensores y el modelo se transfieran a GPU (.to(device)), reduciendo tiempos de entrenamiento de LSTM grandes. El uso de GPU es una práctica estándar en deep learning para series temporales
journalofbigdata.springeropen.com
.

## Robustez en la lectura de datos: 
El daemon recarga continuamente desde SQLite. Podría optimizarse usando una única conexión persistente o incorporando lógica de buffering para evitar múltiples aperturas. Además, sería útil manejar excepciones de red o API (por ejemplo, al usar CCXT) para que el sistema recupere errores sin detenerse abruptamente.

## Parámetros configurables: 
Algunas constantes (longitudes de ventana, listas de características) están codificadas. Sería beneficioso moverlas a configuraciones JSON (usando config_manager) para ajustar sin modificar el código. Esto facilitaría probar diferentes técnicas de ingeniería de características o distintos horizontes de predicción.

## Entrenamiento por lotes y validación cruzada: 

Actualmente se emplea una partición fija (por defecto 20% validación, 10% test). Se podrían incorporar métodos de cross-validation para evaluar robustez o esquemas de entrenamiento continuo (rolling window) para reflejar cambios de régimen. El uso de validación cruzada en series temporales debe hacerse respetando el orden temporal para evitar look-ahead bias
quantstart.com.

## Mejorar documentación y tipo estático: 
Aunque existen comentarios en español, se recomienda añadir docstrings para todas las funciones y clases, especificando tipos de datos en el código (por ejemplo, con typing). Esto ayudaría al mantenimiento y permitiría la generación de documentación automática. También, el uso de linters y pruebas unitarias podría detectar errores temprano.

## Simulación de mercado más realista: 
El actual backtesting es simple. Como sugieren las buenas prácticas, se podría incorporar costos de transacción, slippage y latencia para que los resultados simulados sean más conservadores
quantstart.com. Por ejemplo, descontar comisiones por operación o aplicar retardo a la ejecución simularía mejor un entorno HF real.

## Capa de trading en vivo: 
El código incluye CCXT, pero la lógica actual es principalmente simulada. Para uso real, se debería implementar un manejo seguro de excepciones de API (reconexiones, límites de tasa) y callbacks que sincronicen órdenes con los estados contables reales (inicial, órdenes abiertas, llenadas). También conviene diferenciar claramente el modo paper trading del real, evitando operaciones accidentales en vivo.

## Optimización del modelo LSTM: 
Se pueden explorar arquitecturas alternativas (p. ej. GRU para menor complejidad) o técnicas de regularización avanzadas. Dado que el modelo ya es multi-salida, se podría investigar aprendizaje multi-tarea formal, donde la predicción de volumen ayude a la de retorno. Sin embargo, estas mejoras son de investigación; se deben probar cuidadosamente para evitar overfitting
quantstart.com
.

Escalado de características más sofisticado: Se emplea escalado Z-score global. Podría evaluarse escalado Min-Max o normalización por características que cambian con el tiempo. En series financieras, a veces conviene escalado basado en percentiles o ventanas móviles para adaptarse a regímenes cambiantes. También es crítico aplicar el escalador ajustado solo con datos de entrenamiento para evitar leakage.

Paralelización y concurrencia: Para adaptarse a alta frecuencia, el daemon podría ser multi-hilo o asíncrono, separando la lectura de datos, predicción y logging en tareas paralelas. Esto previene bloqueos si alguna parte (p. ej. IO) tarda. Alternativamente, usar un motor de eventos o websocket de CCXT aceleraría la ingesta de datos en vivo.

# Trabajos realizados

Encargos que Jarvis termina. El resultado lo dice Jarvis, y sus frases no
generan memoria (ignore_roles=["assistant"]), así que sin diario el trabajo se
pierde. Debe quedar una línea compacta por trabajo terminado, no un hecho por
orden. `## Trabajo terminado` es lo que el puente del cerebro entrega.

## Conversación

U: Prepárame un informe sobre la energía solar en España. [herramientas: delegate_task]
A: Lo preparo, señor. Le aviso cuando esté.
A: El informe sobre energía solar está listo, señor. La solar ya genera cerca del diecisiete por ciento de la electricidad en España y ha abaratado el precio mayorista.
U: Y otra cosa, dame tres ideas rápidas para cenar.
A: Aquí tiene tres ideas: tortilla de calabacín, ensalada de garbanzos y salmón al horno. La nota queda en pantalla.
U: Revisa por qué falló el despliegue del agente. [herramientas: delegate_task]
A: Revisado, señor: faltaba el secreto KWAMI_SDK_TOKEN en el repositorio; con eso se arregla.

## Trabajo terminado

- 2026-11-16 | Informe sobre la energía solar en España | La solar da cerca del 17 % de la electricidad en España y ha abaratado el precio mayorista.
- 2026-11-16 | Revisión del despliegue del agente | Faltaba el secreto KWAMI_SDK_TOKEN en el repositorio; con eso se arregla.

## Debe recordar

- ¿Qué informe preparó Jarvis? :: solar
- ¿Por qué falló el despliegue del agente? :: KWAMI_SDK_TOKEN|secreto|token

## No debe recordar

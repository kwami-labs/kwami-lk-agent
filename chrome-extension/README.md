# Kwami · Jarvis en tu Chrome

La extensión que deja a Jarvis ver tus pestañas y trabajar en tu propio Chrome, con
las sesiones que ya tienes abiertas. Es la pieza de Kwami equivalente a Claude in
Chrome, pero no depende de ninguna cuenta de claude.ai: habla solo con Kwami en este
ordenador.

## Por qué

El Chrome propio de Jarvis se maneja por el puerto de depuración, y Google (y todo lo
que entra con «Continuar con Google») lo rechaza: «este navegador o aplicación puede
que no sea seguro». Tu Chrome de siempre no tiene ese puerto, así que esas webs te
dejan entrar, y Jarvis trabaja después sobre esa sesión.

## Instalarla (una vez)

1. Abre KWAMI en este ordenador (el lanzador copia la extensión a
   `%USERPROFILE%\.jarvis\chrome-extension`).
2. En Chrome, ve a `chrome://extensions` y activa **Modo de desarrollador** (arriba a
   la derecha).
3. Pulsa **Cargar descomprimida** y elige la carpeta
   `%USERPROFILE%\.jarvis\chrome-extension`.
4. Fija el icono (el anillo cian) en la barra. En menos de un minuto pone **ON**:
   conectada a Kwami.

Cuando la extensión cambie, pulsa el botón de recargar de su tarjeta en
`chrome://extensions` (o reinicia Chrome).

## Qué hace Jarvis con ella

- **Mirar tus pestañas**: «¿qué pestañas tengo abiertas?», «léeme la del banco».
- **Trabajar en páginas**: las abre en una ventana suya dentro de tu Chrome, en el grupo
  de pestañas **Jarvis**, sin quitarte el foco. El HUD de Kwami muestra lo que hace.
- **Iniciar sesión**: te trae la página al frente y entras tú. Él nunca escribe
  contraseñas, códigos ni números de tarjeta (la extensión lo impide, no solo él).

## Lo que nunca hace

- Ejecutar código que le llegue de fuera: solo corre las funciones que trae dentro.
- Cerrar una pestaña tuya: solo las suyas, y cuando se lo pides.
- Hablar con nadie que no sea Kwami en `127.0.0.1`. El puente de Kwami solo acepta esta
  extensión (por su identificador fijo, `bkgbdgdkjbfagjkghfdfeilncobkiiie`) y solo los
  procesos de Jarvis le dan órdenes; una web abierta en el navegador no puede.

## Pausarla

El botón **Pausar a Jarvis** de su ventanita corta la conexión hasta que la reanudes.
Para apagar el puente del lado de Kwami: `KWAMI_CHROME_BRIDGE=off` en
`~\.jarvis\brain-worker.env`. El puerto es el 47219 (`KWAMI_CHROME_BRIDGE_PORT` y el
apartado «Avanzado» de la ventanita, si lo cambias).

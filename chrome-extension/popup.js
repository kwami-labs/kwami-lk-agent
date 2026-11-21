const DEFAULTS = { port: 47219, paused: false };

const stateBox = document.getElementById('state');
const stateText = document.getElementById('state-text');
const help = document.getElementById('help');
const toggle = document.getElementById('toggle');
const portInput = document.getElementById('port');
const jarvisTabs = document.getElementById('jarvis-tabs');
const last = document.getElementById('last');

const NAMES = {
  tabs: 'mirar tus pestañas',
  read: 'leer una página',
  open: 'abrir una página',
  navigate: 'ir a una dirección',
  click: 'pulsar en una página',
  type: 'escribir en un campo',
  key: 'pulsar una tecla',
  scroll: 'desplazar una página',
  capture: 'mostrar la pestaña en Kwami',
  media: 'controlar un vídeo',
  activate: 'traer una pestaña al frente',
  close: 'cerrar una pestaña suya',
};

function ago(ms) {
  const seconds = Math.round((Date.now() - ms) / 1000);
  if (seconds < 60) return `hace ${seconds} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `hace ${minutes} min`;
  return `hace ${Math.round(minutes / 60)} h`;
}

async function render() {
  const [{ paused, port }, status] = await Promise.all([
    chrome.storage.local.get(DEFAULTS),
    chrome.runtime.sendMessage({ type: 'kwami-status' }).catch(() => null),
  ]);
  portInput.value = String(port);
  const connected = !!(status && status.connected);
  if (paused) {
    stateBox.dataset.state = 'paused';
    stateText.textContent = 'En pausa';
    help.textContent = 'Jarvis no puede ver ni tocar tu Chrome hasta que lo reanudes.';
    toggle.textContent = 'Reanudar';
  } else if (connected) {
    stateBox.dataset.state = 'on';
    stateText.textContent = 'Conectado a Kwami';
    help.textContent = 'Jarvis puede leer tus pestañas y trabajar en las suyas cuando se lo pidas.';
    toggle.textContent = 'Pausar a Jarvis';
  } else {
    stateBox.dataset.state = 'waiting';
    stateText.textContent = 'Esperando a Kwami';
    help.textContent =
      'Abre KWAMI en este ordenador: la extensión se conecta sola en menos de un minuto.';
    toggle.textContent = 'Pausar a Jarvis';
  }
  if (status && status.lastAt) {
    last.textContent = `${NAMES[status.lastCommand] || status.lastCommand}, ${ago(status.lastAt)}`;
  }
  const groups = await chrome.tabGroups.query({ title: 'Jarvis' });
  let count = 0;
  for (const group of groups) {
    count += (await chrome.tabs.query({ groupId: group.id })).length;
  }
  jarvisTabs.textContent = String(count);
}

toggle.addEventListener('click', async () => {
  const { paused } = await chrome.storage.local.get(DEFAULTS);
  await chrome.storage.local.set({ paused: !paused });
  setTimeout(render, 300);
});

portInput.addEventListener('change', async () => {
  const port = Number(portInput.value);
  if (Number.isInteger(port) && port >= 1024 && port <= 65535) {
    await chrome.storage.local.set({ port });
  }
});

document.getElementById('reconnect').addEventListener('click', async () => {
  await chrome.runtime.sendMessage({ type: 'kwami-reconnect' }).catch(() => null);
  setTimeout(render, 800);
});

void render();
setInterval(render, 2000);

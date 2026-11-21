// Jarvis in the user's own Chrome: the link between this browser and Kwami on this computer.
//
// The extension connects to Kwami's bridge on 127.0.0.1 (the PC's work thread hosts it) and
// answers its commands: list the tabs, read one, open a page in Jarvis's window, click,
// type, press a key, scroll, take a picture of the tab Jarvis works in. The bridge only
// accepts this extension (by its fixed id), and only Kwami's own processes may send it
// commands. Nothing here runs code that arrives over the socket: Kwami picks which of the
// functions in page-functions.js runs, never what it says.
//
// The user is in charge: the popup pauses it, and a paused extension drops the link and
// answers nothing until it is resumed. Passwords, codes and card numbers are never typed.

import {
  clickElement,
  extractPage,
  mediaAction,
  pressKey,
  probePage,
  scrollPage,
  typeInto,
} from './page-functions.js';

const DEFAULTS = { port: 47219, paused: false };
const GROUP_TITLE = 'Jarvis';
const RECONNECT_ALARM = 'kwami-reconnect';
const LOAD_TIMEOUT_MS = 15000;
/** A window behind others is not painted, and Chrome then never answers a capture. */
const CAPTURE_TIMEOUT_MS = 2500;
const MAX_TEXT = 12000;

const status = {
  connected: false,
  paused: false,
  lastCommand: '',
  lastAt: 0,
  lastError: '',
};

/** @type {WebSocket | null} */
let socket = null;
/** Set before the first await: the worker wakes with several reasons to connect at once. */
let connecting = false;

async function prefs() {
  return chrome.storage.local.get(DEFAULTS);
}

function publishStatus() {
  chrome.storage.session.set({ status: { ...status } }).catch(() => {});
  chrome.action
    .setBadgeText({ text: status.paused ? 'II' : status.connected ? 'ON' : '' })
    .catch(() => {});
  chrome.action
    .setBadgeBackgroundColor({ color: status.paused ? '#6b7280' : '#0e7490' })
    .catch(() => {});
}

async function connect() {
  if (connecting) return;
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  connecting = true;
  try {
    await open();
  } finally {
    connecting = false;
  }
}

async function open() {
  const { port, paused } = await prefs();
  status.paused = !!paused;
  if (status.paused) {
    publishStatus();
    return;
  }
  let ws;
  try {
    ws = new WebSocket(`ws://127.0.0.1:${Number(port) || DEFAULTS.port}/extension`);
  } catch (error) {
    status.lastError = String(error);
    publishStatus();
    return;
  }
  socket = ws;
  ws.onopen = () => {
    status.connected = true;
    status.lastError = '';
    publishStatus();
    ws.send(
      JSON.stringify({
        type: 'hello',
        version: chrome.runtime.getManifest().version,
        browser: navigator.userAgent,
      }),
    );
  };
  ws.onmessage = (event) => {
    void onMessage(ws, event.data);
  };
  ws.onclose = () => {
    if (socket === ws) socket = null;
    status.connected = false;
    publishStatus();
  };
  ws.onerror = () => {};
}

function disconnect() {
  const ws = socket;
  socket = null;
  if (ws) ws.close();
  status.connected = false;
  publishStatus();
}

async function onMessage(ws, raw) {
  let message;
  try {
    message = JSON.parse(raw);
  } catch {
    return;
  }
  if (message.type === 'ping') {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'pong' }));
    return;
  }
  if (message.id === undefined) return;
  let reply;
  try {
    if (status.paused) throw new Error('paused');
    const handler = COMMANDS[message.cmd];
    if (!handler) throw new Error('unknown-command');
    status.lastCommand = String(message.cmd);
    status.lastAt = Date.now();
    reply = { id: message.id, ok: true, result: await handler(message.args || {}) };
  } catch (error) {
    reply = { id: message.id, ok: false, error: String((error && error.message) || error) };
  }
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(reply));
  publishStatus();
}

// -- tabs and Jarvis's window ---------------------------------------------------------

function httpUrl(raw) {
  let url;
  try {
    url = new URL(String(raw || ''));
  } catch {
    throw new Error('bad-url');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('bad-url');
  return url.href;
}

async function jarvisGroups() {
  const groups = await chrome.tabGroups.query({ title: GROUP_TITLE });
  return new Set(groups.map((g) => g.id));
}

function describeTab(tab, focusedWindowId, groups) {
  return {
    id: tab.id,
    windowId: tab.windowId,
    index: tab.index,
    title: tab.title || '',
    url: tab.url || tab.pendingUrl || '',
    active: !!tab.active,
    focused: !!tab.active && tab.windowId === focusedWindowId,
    audible: !!tab.audible,
    pinned: !!tab.pinned,
    loading: tab.status === 'loading',
    jarvis: groups.has(tab.groupId),
  };
}

async function focusedWindowId() {
  try {
    const win = await chrome.windows.getLastFocused();
    return win.focused ? win.id : -1;
  } catch {
    return -1;
  }
}

async function tabInfo(tabId) {
  const tab = await chrome.tabs.get(Number(tabId));
  return describeTab(tab, await focusedWindowId(), await jarvisGroups());
}

async function waitLoaded(tabId) {
  const deadline = Date.now() + LOAD_TIMEOUT_MS;
  await new Promise((resolve) => setTimeout(resolve, 300));
  while (Date.now() < deadline) {
    const tab = await chrome.tabs.get(tabId);
    if (tab.status === 'complete') return;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

/** Jarvis's own window in this Chrome, if it is still open. */
async function jarvisWindow() {
  const { jarvisWindowId } = await chrome.storage.session.get('jarvisWindowId');
  if (typeof jarvisWindowId !== 'number') return null;
  try {
    const win = await chrome.windows.get(jarvisWindowId);
    return win.type === 'normal' ? win.id : null;
  } catch {
    return null;
  }
}

async function groupAsJarvis(tabId, windowId) {
  try {
    const [existing] = await chrome.tabGroups.query({ title: GROUP_TITLE, windowId });
    const groupId = await chrome.tabs.group(
      existing ? { tabIds: [tabId], groupId: existing.id } : { tabIds: [tabId], createProperties: { windowId } },
    );
    if (!existing) await chrome.tabGroups.update(groupId, { title: GROUP_TITLE, color: 'cyan' });
  } catch {
    // A tab that cannot be grouped (a popup window) still works.
  }
}

/** A page in Jarvis's window, which it opens without taking the focus from the user. */
async function openTab({ url, focus }) {
  const href = httpUrl(url);
  let windowId = await jarvisWindow();
  let tab;
  if (windowId === null) {
    const win = await chrome.windows.create({ url: href, focused: !!focus, type: 'normal' });
    windowId = win.id;
    tab = win.tabs[0];
    await chrome.storage.session.set({ jarvisWindowId: windowId });
  } else {
    tab = await chrome.tabs.create({ windowId, url: href, active: true });
    if (focus) await chrome.windows.update(windowId, { focused: true });
  }
  await groupAsJarvis(tab.id, windowId);
  await waitLoaded(tab.id);
  return tabInfo(tab.id);
}

async function inPage(tabId, func, args = []) {
  let results;
  try {
    results = await chrome.scripting.executeScript({ target: { tabId: Number(tabId) }, func, args });
  } catch (error) {
    const text = String((error && error.message) || error);
    if (/cannot (access|be scripted)|chrome:\/\/|extensions gallery|webstore/i.test(text)) {
      throw new Error('protected-page');
    }
    throw error;
  }
  const [first] = results || [];
  return first ? first.result : undefined;
}

function checked(result) {
  if (result && typeof result === 'object' && result.error) throw new Error(result.error);
  return result;
}

const COMMANDS = {
  async hello() {
    return { version: chrome.runtime.getManifest().version };
  },

  async tabs() {
    const [tabs, focused, groups] = await Promise.all([
      chrome.tabs.query({}),
      focusedWindowId(),
      jarvisGroups(),
    ]);
    return { tabs: tabs.map((tab) => describeTab(tab, focused, groups)) };
  },

  async info({ tabId }) {
    return tabInfo(tabId);
  },

  async open({ url, focus }) {
    return openTab({ url, focus });
  },

  async navigate({ tabId, url }) {
    const id = Number(tabId);
    await chrome.tabs.update(id, { url: httpUrl(url) });
    await waitLoaded(id);
    return tabInfo(id);
  },

  async back({ tabId }) {
    const id = Number(tabId);
    await chrome.tabs.goBack(id);
    await waitLoaded(id);
    return tabInfo(id);
  },

  async forward({ tabId }) {
    const id = Number(tabId);
    await chrome.tabs.goForward(id);
    await waitLoaded(id);
    return tabInfo(id);
  },

  async activate({ tabId, focus }) {
    const tab = await chrome.tabs.update(Number(tabId), { active: true });
    if (focus) {
      const win = await chrome.windows.get(tab.windowId);
      await chrome.windows.update(tab.windowId, {
        focused: true,
        ...(win.state === 'minimized' ? { state: 'normal' } : {}),
      });
    }
    return tabInfo(tab.id);
  },

  async close({ tabId }) {
    await chrome.tabs.remove(Number(tabId));
    return { closed: true };
  },

  async read({ tabId, maxText }) {
    const page = await inPage(tabId, extractPage, [Math.min(Number(maxText) || MAX_TEXT, 50000)]);
    return page || { title: '', url: '', text: '', elements: [] };
  },

  async click({ tabId, elementId }) {
    return checked(await inPage(tabId, clickElement, [String(elementId || '')]));
  },

  async type({ tabId, elementId, text, clear }) {
    return checked(
      await inPage(tabId, typeInto, [String(elementId || ''), String(text ?? ''), clear !== false]),
    );
  },

  async key({ tabId, key }) {
    return checked(await inPage(tabId, pressKey, [String(key || '')]));
  },

  async scroll({ tabId, direction }) {
    return checked(await inPage(tabId, scrollPage, [direction === 'up' ? 'up' : 'down']));
  },

  async probe({ tabId }) {
    return inPage(tabId, probePage);
  },

  async media({ tabId, action, level }) {
    return { result: await inPage(tabId, mediaAction, [String(action || ''), Number(level) || 0]) };
  },

  /** A picture of the tab, only while it is the one showing in its window. */
  async capture({ tabId, quality }) {
    const tab = await chrome.tabs.get(Number(tabId));
    if (!tab.active) throw new Error('not-visible');
    const win = await chrome.windows.get(tab.windowId);
    if (win.state === 'minimized') throw new Error('not-visible');
    const dataUrl = await Promise.race([
      chrome.tabs.captureVisibleTab(tab.windowId, {
        format: 'jpeg',
        quality: Math.max(10, Math.min(Number(quality) || 50, 90)),
      }),
      new Promise((_resolve, reject) =>
        setTimeout(() => reject(new Error('not-visible')), CAPTURE_TIMEOUT_MS),
      ),
    ]);
    return { jpeg: dataUrl.slice(dataUrl.indexOf(',') + 1) };
  },
};

// -- staying connected ----------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(RECONNECT_ALARM, { periodInMinutes: 0.5 });
  void connect();
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create(RECONNECT_ALARM, { periodInMinutes: 0.5 });
  void connect();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM) void connect();
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return;
  if (changes.paused || changes.port) {
    disconnect();
    void connect();
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.type === 'kwami-status') {
    sendResponse({ ...status });
    return false;
  }
  if (message && message.type === 'kwami-reconnect') {
    disconnect();
    void connect().then(() => sendResponse({ ok: true }));
    return true;
  }
  return false;
});

void connect();

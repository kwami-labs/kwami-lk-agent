// What runs inside a page, in the extension's own isolated world.
//
// chrome.scripting sends each function's source to the page on its own, so every one of
// them is self-contained: nothing from the module around it reaches the page. None of
// them runs text that came from Kwami as code; Kwami only picks which one runs.

/** The page as Jarvis reads it: its text, and the elements it can use, stamped with ids. */
export function extractPage(maxText) {
  const doc = document;
  const body = doc.body;
  if (!body) return { title: doc.title, url: location.href, text: '', elements: [] };

  const selector = [
    'a[href]',
    'button',
    'input:not([type=hidden])',
    'select',
    'textarea',
    '[contenteditable=""]',
    '[contenteditable="true"]',
    '[role="button"]',
    '[role="link"]',
    '[role="tab"]',
    '[role="menuitem"]',
    '[role="checkbox"]',
    '[role="option"]',
    '[role="textbox"]',
    '[role="combobox"]',
    '[role="searchbox"]',
    'summary',
    '[onclick]',
    '[tabindex="0"]',
  ].join(',');

  doc.querySelectorAll('[data-kwami-id]').forEach((el) => el.removeAttribute('data-kwami-id'));

  const main = doc.querySelector('main, article, [role="main"], #content') || body;
  // What a person reads first: headings and paragraphs, not the tables, infoboxes, menus,
  // headers and footers many pages put before them (on Wikipedia the infobox filled it all).
  const skip =
    'table, nav, aside, header, footer, [role="navigation"], [role="banner"], .infobox, .navbox, .sidebar';
  const blocks = [...main.querySelectorAll('h1, h2, h3, p, li, blockquote, pre')].filter(
    (el) =>
      !el.closest(skip) &&
      !(el.parentElement && el.parentElement.closest('p, li, blockquote, pre')) &&
      el.getClientRects().length > 0,
  );
  let raw = blocks
    .map((el) => (el.innerText || '').trim().replace(/\s+/g, ' '))
    .filter((t) => t.length > 1)
    .join('\n');
  if (raw.length < 200) raw = main.innerText || body.innerText || '';
  const text = raw.replace(/\n{3,}/g, '\n\n').slice(0, maxText || 8000);

  const height = window.innerHeight;
  const items = [];
  for (const el of doc.querySelectorAll(selector)) {
    if (items.length >= 150) break;
    const tag = el.tagName.toLowerCase();
    const type = tag === 'input' ? `input:${el.type || 'text'}` : tag;
    let label = (el.getAttribute('aria-label') || el.innerText || el.textContent || '')
      .trim()
      .replace(/\s+/g, ' ');
    if (!label) {
      label =
        el.getAttribute('title') ||
        el.getAttribute('placeholder') ||
        el.getAttribute('name') ||
        el.getAttribute('alt') ||
        '';
    }
    if (!label && (tag === 'input' || tag === 'textarea' || el.isContentEditable)) label = 'field';
    if (!label) continue;
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden';
    const id = `el-${items.length}`;
    el.setAttribute('data-kwami-id', id);
    items.push({
      id,
      type,
      label: label.slice(0, 80),
      x: Math.round(rect.left + rect.width / 2),
      y: Math.round(rect.top + rect.height / 2),
      visible,
      inView: visible && rect.bottom > 0 && rect.top < height,
    });
  }
  return { title: doc.title, url: location.href, text, elements: items };
}

/** Click an element stamped by extractPage, as a pointer would. */
export function clickElement(id) {
  const el = document.querySelector(`[data-kwami-id="${CSS.escape(String(id))}"]`);
  if (!el) return { error: 'not-found' };
  el.scrollIntoView({ block: 'center', inline: 'center' });
  const label = (el.getAttribute('aria-label') || el.innerText || el.value || '')
    .trim()
    .replace(/\s+/g, ' ')
    .slice(0, 60);
  const rect = el.getBoundingClientRect();
  const init = {
    bubbles: true,
    cancelable: true,
    composed: true,
    clientX: rect.left + rect.width / 2,
    clientY: rect.top + rect.height / 2,
    button: 0,
  };
  el.dispatchEvent(new PointerEvent('pointerdown', init));
  el.dispatchEvent(new MouseEvent('mousedown', init));
  if (typeof el.focus === 'function') el.focus({ preventScroll: true });
  el.dispatchEvent(new PointerEvent('pointerup', init));
  el.dispatchEvent(new MouseEvent('mouseup', init));
  el.click();
  return { ok: true, label };
}

/** Type into a field: the stamped one, or the one with the focus. Never a password. */
export function typeInto(id, text, clear) {
  const el = id
    ? document.querySelector(`[data-kwami-id="${CSS.escape(String(id))}"]`)
    : document.activeElement;
  if (!el || el === document.body || el === document.documentElement) return { error: 'no-field' };
  const kind = (el.getAttribute('type') || '').toLowerCase();
  const autocomplete = (el.getAttribute('autocomplete') || '').toLowerCase();
  if (
    kind === 'password' ||
    autocomplete.includes('password') ||
    autocomplete.includes('one-time-code') ||
    autocomplete.startsWith('cc-')
  ) {
    return { error: 'secret-field' };
  }
  el.scrollIntoView({ block: 'center' });
  if (typeof el.focus === 'function') el.focus();
  const value = String(text);
  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    const proto = el.tagName === 'INPUT' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, clear ? value : el.value + value);
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { ok: true };
  }
  if (el.isContentEditable) {
    if (clear) {
      const range = document.createRange();
      range.selectNodeContents(el);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand('delete');
    }
    document.execCommand('insertText', false, value);
    return { ok: true };
  }
  return { error: 'not-a-field' };
}

/** Press a key where the focus is. Enter in a form field sends the form, as a person's would. */
export function pressKey(key) {
  const el = document.activeElement || document.body;
  const name = key === 'Space' ? ' ' : String(key);
  const codes = {
    Enter: 13,
    Tab: 9,
    Escape: 27,
    Backspace: 8,
    Delete: 46,
    ArrowDown: 40,
    ArrowUp: 38,
    ArrowLeft: 37,
    ArrowRight: 39,
    PageDown: 34,
    PageUp: 33,
    Home: 36,
    End: 35,
    ' ': 32,
  };
  const keyCode = codes[name] || (name.length === 1 ? name.toUpperCase().charCodeAt(0) : 0);
  const init = {
    key: name,
    code: name.length === 1 ? (name === ' ' ? 'Space' : `Key${name.toUpperCase()}`) : name,
    keyCode,
    which: keyCode,
    bubbles: true,
    cancelable: true,
    composed: true,
  };
  const go = el.dispatchEvent(new KeyboardEvent('keydown', init));
  if (name.length === 1 || name === 'Enter') el.dispatchEvent(new KeyboardEvent('keypress', init));
  el.dispatchEvent(new KeyboardEvent('keyup', init));
  if (!go) return { ok: true };
  if (name === 'Enter' && el.tagName === 'INPUT') {
    const form = el.form || el.closest('form');
    if (form) {
      if (typeof form.requestSubmit === 'function') form.requestSubmit();
      else form.submit();
      return { ok: true, submitted: true };
    }
  }
  if (name === 'Tab') {
    const focusable = [
      ...document.querySelectorAll(
        'a[href], button, input:not([type=hidden]), select, textarea, [tabindex]:not([tabindex="-1"])',
      ),
    ].filter((e) => e.getClientRects().length > 0 && !e.disabled);
    const next = focusable[(focusable.indexOf(el) + 1) % Math.max(focusable.length, 1)];
    if (next) next.focus();
  }
  return { ok: true };
}

/** Scroll the page, or the biggest scrolling box when the page itself does not move. */
export function scrollPage(direction) {
  const delta = (direction === 'up' ? -1 : 1) * Math.round(window.innerHeight * 0.8);
  const root = document.scrollingElement || document.documentElement;
  const before = root.scrollTop;
  window.scrollBy({ top: delta, behavior: 'instant' });
  if (root.scrollTop !== before) return { ok: true };
  let best = null;
  let area = 0;
  for (const el of document.querySelectorAll('body *')) {
    if (el.scrollHeight <= el.clientHeight + 20) continue;
    const overflow = getComputedStyle(el).overflowY;
    if (overflow !== 'auto' && overflow !== 'scroll') continue;
    const size = el.clientWidth * el.clientHeight;
    if (size > area) {
      area = size;
      best = el;
    }
  }
  if (best) best.scrollBy({ top: delta, behavior: 'instant' });
  return { ok: true };
}

/** What a sign-in check needs: where the page is, a password field showing, the first words. */
export function probePage() {
  const password = [...document.querySelectorAll('input[type=password]')].some(
    (e) => e.getClientRects().length > 0 && getComputedStyle(e).visibility !== 'hidden',
  );
  return {
    href: location.href,
    address: location.host + location.pathname,
    password,
    head: (document.body ? document.body.innerText : '').slice(0, 2000),
  };
}

/** The player on the page, by a named action. The first-result click is for search pages. */
export function mediaAction(action, level) {
  if (action === 'first_result') {
    const selectors = [
      'ytd-video-renderer a#video-title',
      'ytmusic-responsive-list-item-renderer a',
      'a[data-testid="track-row"]',
      'ul.sc-list-nav li a',
      'a[href*="/watch"]',
      'a[href*="/track/"]',
      'a[href*="/album/"]',
    ];
    for (const s of selectors) {
      const el = document.querySelector(s);
      if (el) {
        el.click();
        return `clicked:${s}`;
      }
    }
    return 'no-result';
  }
  const m = document.querySelector('video,audio');
  if (!m) return 'no-media';
  const play = () => {
    const started = m.play();
    if (started && typeof started.catch === 'function') started.catch(() => {});
  };
  switch (action) {
    case 'pause':
      m.pause();
      return 'paused';
    case 'resume':
      play();
      return 'playing';
    case 'stop':
      m.pause();
      m.currentTime = 0;
      return 'stopped';
    case 'mute':
      m.muted = true;
      return 'muted';
    case 'unmute':
      m.muted = false;
      return 'unmuted';
    case 'restart':
      m.currentTime = 0;
      play();
      return 'restarted';
    case 'volume':
      m.volume = Math.max(0, Math.min(1, Number(level) || 0));
      m.muted = false;
      return `volume:${m.volume}`;
    case 'now_playing':
      return JSON.stringify({
        title: document.title,
        paused: m.paused,
        muted: m.muted,
        volume: m.volume,
        position: Math.round(m.currentTime || 0),
        duration: Math.round(m.duration || 0),
      });
    default:
      return 'unknown-action';
  }
}

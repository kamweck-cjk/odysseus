/**
 * Sarge �?" research-backed, resumable project campaigns.
 *
 * A tool window that turns a deep-research report into an executable plan, runs
 * it step by step with a fresh context per step, and writes an after-action
 * report. The UI is a full tool window (like Deep Research) rather than a chat
 * extension, because campaigns outlive a single conversation.
 *
 * Entry points:
 *   sargeModule.init(apiBase, markdownModule, uiModule)
 *   sargeModule.toggle()
 *   sargeModule.open()
 *   sargeModule.createFromResearch(sessionId, query)
 */

import * as api from './api.js';
import uiModule from '../ui.js';

const _esc = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

const _fmt = (value) => {
  if (!value) return '�?"';
  try {
    return new Date(value).toLocaleString();
  } catch (_) {
    return String(value);
  }
};

const _bytes = (n) => {
  const value = Number(n) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
};

const STATUS_LABELS = {
  draft: 'Draft',
  planning: 'Planning',
  awaiting_approval: 'Plan review',
  running: 'Running',
  paused: 'Paused',
  blocked: 'Blocked',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

let _initialized = false;
let _open = false;
let _apiBase = '';
let _markdown = null;
let _campaigns = [];
let _selectedId = null;
let _detail = null;
let _stream = null;
let _pollTimer = null;
let _busy = false;
let _onDocKeydown = null;
const _unread = new Set();

// ------------------------------------------------------------------
// Lifecycle
// ------------------------------------------------------------------

export function init(apiBase, markdownMod) {
  if (_initialized) return;
  _initialized = true;
  _apiBase = apiBase || '';
  _markdown = markdownMod || null;
  api.init(_apiBase);
  window.sargeModule = { toggle, open, close: closePanel, createFromResearch, refresh: _refresh };
}

export function isOpen() { return _open; }

export function toggle() {
  if (_open) closePanel();
  else open();
}

export function open(focusId) {
  document.body.classList.add('sarge-panel-view');
  const toolBtn = document.getElementById('tool-sarge-btn');
  if (toolBtn) toolBtn.classList.add('active');
  const railBtn = document.getElementById('rail-sarge');
  if (railBtn) railBtn.classList.remove('sarge-unread');

  if (_open) {
    const overlay = document.getElementById('sarge-overlay');
    if (overlay && overlay.style.display === 'none') overlay.style.display = '';
    if (focusId) _select(focusId);
    return;
  }
  _open = true;

  const overlay = document.createElement('div');
  overlay.id = 'sarge-overlay';
  overlay.className = 'modal sarge-overlay';
  overlay.innerHTML = _shellHtml();
  document.body.appendChild(overlay);

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closePanel();
  });
  _onDocKeydown = (e) => {
    if (e.key === 'Escape' && _open) {
      const createOpen = document.getElementById('sarge-create-form');
      if (createOpen && createOpen.style.display !== 'none') {
        createOpen.style.display = 'none';
        return;
      }
      closePanel();
    }
  };
  document.addEventListener('keydown', _onDocKeydown);

  _wire(overlay);
  _refresh().then(() => {
    if (focusId) _select(focusId);
  });
}

export function closePanel() {
  if (!_open) return;
  _open = false;
  _stopStream();
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  if (_onDocKeydown) {
    document.removeEventListener('keydown', _onDocKeydown);
    _onDocKeydown = null;
  }
  document.body.classList.remove('sarge-panel-view');
  document.getElementById('tool-sarge-btn')?.classList.remove('active');
  document.getElementById('sarge-overlay')?.remove();
}

/** Used by the Deep Research panel and the chat anchor flow. */
export function createFromResearch(sessionId, query) {
  open();
  const form = document.getElementById('sarge-create-form');
  if (form) form.style.display = '';
  const input = document.getElementById('sarge-source-session');
  if (input) input.value = sessionId || '';
  const objective = document.getElementById('sarge-objective');
  if (objective && !objective.value && query) {
    objective.value = `Turn the research "${query}" into a completed, verified project.`;
  }
  setTimeout(() => objective?.focus(), 50);
}

// ------------------------------------------------------------------
// Data
// ------------------------------------------------------------------

async function _refresh() {
  try {
    _campaigns = await api.listCampaigns();
  } catch (err) {
    uiModule.showError?.(`Sarge: ${err.message || err}`);
    _campaigns = [];
  }
  _renderList();
  if (_selectedId && !_campaigns.some((c) => c.id === _selectedId)) {
    _selectedId = null;
    _detail = null;
    _renderDetail();
  }
  _renderRailBadge();
}

async function _select(id) {
  if (!id) return;
  _selectedId = id;
  _renderList();
  try {
    _detail = await api.getCampaign(id);
  } catch (err) {
    uiModule.showError?.(`Sarge: ${err.message || err}`);
    _detail = null;
  }
  _renderDetail();
  _syncStream();
}

function _syncStream() {
  _stopStream();
  const status = _detail?.status;
  if (!_selectedId || !['planning', 'running'].includes(status)) return;
  _startStream(_selectedId, _detail?.events?.length ? _detail.events[_detail.events.length - 1].seq : 0);
}

function _startStream(id, after) {
  if (!('EventSource' in window)) {
    _startPolling(id);
    return;
  }
  try {
    _stream = new EventSource(api.eventStreamUrl(id, after), { withCredentials: true });
  } catch (_) {
    _startPolling(id);
    return;
  }
  _stream.onmessage = (event) => {
    let data = null;
    try { data = JSON.parse(event.data); } catch (_) { return; }
    if (!data) return;
    _appendEvent(data);
  };
  _stream.addEventListener('end', () => {
    _stopStream();
    if (_selectedId === id) {
      api.getCampaign(id).then((detail) => {
        _detail = detail;
        _renderDetail();
        _refresh();
      }).catch(() => {});
    }
  });
  _stream.onerror = () => {
    // Fall back to polling if SSE is unavailable (proxy buffering, etc.).
    _stopStream();
    _startPolling(id);
  };
}

function _stopStream() {
  if (_stream) {
    try { _stream.close(); } catch (_) {}
    _stream = null;
  }
}

function _startPolling(id) {
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(async () => {
    if (!_selectedId || _selectedId !== id) {
      clearInterval(_pollTimer);
      _pollTimer = null;
      return;
    }
    try {
      const detail = await api.getCampaign(id);
      _detail = detail;
      _renderDetail();
      if (!['planning', 'running'].includes(detail.status)) {
        clearInterval(_pollTimer);
        _pollTimer = null;
        _refresh();
      }
    } catch (_) {}
  }, 2000);
}

function _appendEvent(event) {
  if (!_detail) return;
  const events = _detail.events || (_detail.events = []);
  if (events.some((e) => e.seq === event.seq)) return;
  events.push(event);
  const log = document.getElementById('sarge-log');
  if (log) {
    log.insertAdjacentHTML('beforeend', _eventHtml(event));
    log.scrollTop = log.scrollHeight;
  }
}

function _renderRailBadge() {
  const active = _campaigns.some((c) => ['planning', 'running', 'awaiting_approval', 'blocked'].includes(c.status));
  document.getElementById('rail-sarge')?.classList.toggle('sarge-unread', active && !_open);
  document.getElementById('tool-sarge-btn')?.classList.toggle('sarge-unread', active && !_open);
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------

function _shellHtml() {
  return `
    <div class="modal-content sarge-modal-content" id="sarge-pane">
      <div class="modal-header sarge-header">
        <h4><span class="sarge-header-icon">${_shieldIcon}</span><span>Sarge</span>
          <span class="sarge-subtitle">research → plan → execute</span></h4>
        <div class="sarge-header-actions">
          <button id="sarge-refresh" class="sarge-icon-btn" type="button" title="Refresh">${_refreshIcon}</button>
          <button id="sarge-new" class="sarge-btn sarge-btn-primary" type="button">New campaign</button>
          <button id="sarge-panel-minimize" class="modal-minimize-btn" type="button" title="Minimize">${_minusIcon}</button>
          <button id="sarge-panel-close" class="close-btn" title="Close">&#x2716;</button>
        </div>
      </div>
      <div class="modal-body sarge-body">
        <form id="sarge-create-form" class="sarge-create" style="display:none">
          <div class="sarge-field">
            <label for="sarge-objective">Objective <span class="sarge-req">*</span></label>
            <textarea id="sarge-objective" rows="3" placeholder="What should the finished project achieve?"></textarea>
          </div>
          <div class="sarge-field-row">
            <div class="sarge-field">
              <label for="sarge-source-session">Research report</label>
              <select id="sarge-source-session"><option value="">(none)</option></select>
            </div>
            <div class="sarge-field">
              <label for="sarge-privacy">Privacy</label>
              <select id="sarge-privacy">
                <option value="local">Local only (no web)</option>
                <option value="web">Allow web tools</option>
              </select>
            </div>
            <div class="sarge-field">
              <label for="sarge-model">Model</label>
              <select id="sarge-model"><option value="">Loading…</option></select>
            </div>
          </div>
          <div class="sarge-field">
            <label for="sarge-brief">Operator brief (optional)</label>
            <textarea id="sarge-brief" rows="2" placeholder="Constraints, preferences, definitions of done…"></textarea>
          </div>
          <div class="sarge-create-actions">
            <label class="sarge-check"><input type="checkbox" id="sarge-unattended"> Run unattended after approval</label>
            <span class="sarge-spacer"></span>
            <button type="button" class="sarge-btn" id="sarge-create-cancel">Cancel</button>
            <button type="submit" class="sarge-btn sarge-btn-primary" id="sarge-create-submit">Create campaign</button>
          </div>
          <div class="sarge-error" id="sarge-create-error" style="display:none"></div>
        </form>

        <div class="sarge-columns">
          <aside class="sarge-col sarge-col-list">
            <div class="sarge-col-head">Campaigns</div>
            <div class="sarge-list" id="sarge-list"></div>
          </aside>
          <section class="sarge-col sarge-col-detail" id="sarge-detail"></section>
          <aside class="sarge-col sarge-col-log">
            <div class="sarge-col-head">Field log</div>
            <div class="sarge-log" id="sarge-log"><div class="sarge-empty-small">No activity yet.</div></div>
          </aside>
        </div>
      </div>
    </div>
  `;
}

function _renderList() {
  const list = document.getElementById('sarge-list');
  if (!list) return;
  if (!_campaigns.length) {
    list.innerHTML = '<div class="sarge-empty-small">No campaigns yet. Create one from a research report.</div>';
    return;
  }
  list.innerHTML = _campaigns.map((c) => {
    const status = c.status || 'draft';
    const active = c.id === _selectedId;
    const progress = c.steps_total ? `${c.steps_completed}/${c.steps_total}` : '';
    return `
      <button type="button" class="sarge-list-item${active ? ' active' : ''}" data-id="${_esc(c.id)}">
        <span class="sarge-list-title">${_esc(c.title || 'Untitled')}</span>
        <span class="sarge-list-meta">
          <span class="sarge-badge sarge-badge-${_esc(status)}">${_esc(STATUS_LABELS[status] || status)}</span>
          ${progress ? `<span class="sarge-progress">${_esc(progress)}</span>` : ''}
        </span>
      </button>
    `;
  }).join('');
}

function _renderDetail() {
  const pane = document.getElementById('sarge-detail');
  if (!pane) return;
  if (!_detail) {
    pane.innerHTML = '<div class="sarge-placeholder"><p>Select a campaign or create a new one.</p></div>';
    _renderLog();
    return;
  }
  const c = _detail;
  const status = c.status || 'draft';
  const plan = c.plan;
  const actions = _actionsHtml(c);

  pane.innerHTML = `
    <div class="sarge-detail-head">
      <div>
        <h5>${_esc(c.title || 'Untitled')}</h5>
        <div class="sarge-detail-meta">
          <span class="sarge-badge sarge-badge-${_esc(status)}">${_esc(STATUS_LABELS[status] || status)}</span>
          <span>${_esc(c.model || 'no model')}</span>
          ${c.endpoint_name ? `<span class="sarge-dot">�?�</span><span>${_esc(c.endpoint_name)}</span>` : ''}
          <span class="sarge-dot">�?�</span><span>${_esc(c.privacy === 'web' ? 'web allowed' : 'local only')}</span>
        </div>
      </div>
      <div class="sarge-detail-actions">${actions}</div>
    </div>
    ${c.status_detail ? `<div class="sarge-status-detail">${_esc(c.status_detail)}</div>` : ''}
    ${c.error ? `<div class="sarge-error">${_esc(c.error)}</div>` : ''}
    <div class="sarge-detail-body">
      <div class="sarge-section">
        <div class="sarge-section-title">Objective</div>
        <p class="sarge-wrap">${_esc(c.objective || '')}</p>
      </div>
      ${plan ? _planHtml(plan, c) : '<div class="sarge-placeholder"><p>No plan yet. Generate one with the model.</p></div>'}
      ${c.final_report ? `
        <div class="sarge-section">
          <div class="sarge-section-title">After-action report</div>
          <div class="sarge-report" id="sarge-final-report"></div>
        </div>` : ''}
    </div>
  `;

  if (c.final_report) {
    const node = document.getElementById('sarge-final-report');
    if (node) {
      if (_markdown?.renderContent) {
        try { node.innerHTML = _markdown.renderContent(c.final_report); } catch (_) {
          node.textContent = c.final_report;
        }
      } else {
        node.textContent = c.final_report;
      }
    }
  }
  _renderLog();
}

function _actionsHtml(c) {
  const status = c.status || 'draft';
  const btn = (label, action, cls = '') =>
    `<button type="button" class="sarge-btn ${cls}" data-action="${action}">${_esc(label)}</button>`;
  const parts = [];
  if (status === 'draft' || status === 'failed' || status === 'cancelled') {
    parts.push(btn(c.plan ? 'Re-plan' : 'Plan with model', 'plan', 'sarge-btn-primary'));
  }
  if (status === 'awaiting_approval') {
    parts.push(btn('Approve & run', 'approve', 'sarge-btn-primary'));
    parts.push(btn('Re-plan', 'plan'));
  }
  if (['blocked', 'paused'].includes(status) && c.plan) {
    parts.push(btn('Resume', 'resume', 'sarge-btn-primary'));
    parts.push(btn('Retry failed step', 'retry'));
  }
  if (status === 'running') {
    parts.push(btn('Pause', 'pause'));
    parts.push(btn('Cancel', 'cancel'));
  }
  if (status === 'planning') {
    parts.push(btn('Cancel', 'cancel'));
  }
  if (['completed', 'failed', 'cancelled', 'blocked', 'paused', 'awaiting_approval'].includes(status)) {
    parts.push(btn('Delete', 'delete', 'sarge-btn-danger'));
  }
  return parts.join('');
}

function _planHtml(plan, c) {
  const steps = plan.steps || [];
  return `
    <div class="sarge-section">
      <div class="sarge-section-title">${_esc(plan.title || 'Plan')}</div>
      <p class="sarge-wrap sarge-muted">${_esc(plan.summary || '')}</p>
      ${plan.success_criteria?.length ? `
        <div class="sarge-criteria">
          <div class="sarge-section-subtitle">Success criteria</div>
          <ul>${plan.success_criteria.map((s) => `<li>${_esc(s)}</li>`).join('')}</ul>
        </div>` : ''}
      <div class="sarge-steps">
        ${steps.map((s, i) => _stepHtml(s, i)).join('')}
      </div>
      <div class="sarge-artifacts">
        <div class="sarge-section-subtitle">Deliverables in workspace</div>
        <div id="sarge-artifacts">Loading…</div>
      </div>
    </div>
  `;
}

function _stepHtml(step, index) {
  const status = step.status || 'pending';
  const attempts = step.attempts ? ` �?� ${step.attempts} attempt${step.attempts > 1 ? 's' : ''}` : '';
  return `
    <details class="sarge-step sarge-step-${_esc(status)}" ${status === 'running' ? 'open' : ''}>
      <summary>
        <span class="sarge-step-num">${index + 1}</span>
        <span class="sarge-step-title">${_esc(step.title || step.id)}</span>
        <span class="sarge-badge sarge-badge-step-${_esc(status)}">${_esc(status)}${attempts}</span>
      </summary>
      <div class="sarge-step-body">
        <p class="sarge-wrap">${_esc(step.objective || '')}</p>
        ${step.deliverables?.length ? `<div class="sarge-step-sub">Deliverables</div><ul>${step.deliverables.map((d) => `<li>${_esc(d)}</li>`).join('')}</ul>` : ''}
        ${step.acceptance_criteria?.length ? `<div class="sarge-step-sub">Acceptance</div><ul>${step.acceptance_criteria.map((d) => `<li>${_esc(d)}</li>`).join('')}</ul>` : ''}
        ${step.summary ? `<div class="sarge-step-sub">Report</div><p class="sarge-wrap sarge-muted">${_esc(step.summary)}</p>` : ''}
        ${step.verification ? `<div class="sarge-step-sub">Verification</div><p class="sarge-wrap sarge-muted">${_esc(step.verification)}</p>` : ''}
      </div>
    </details>
  `;
}

function _renderLog() {
  const log = document.getElementById('sarge-log');
  if (!log) return;
  const events = _detail?.events || [];
  if (!events.length) {
    log.innerHTML = '<div class="sarge-empty-small">No activity yet.</div>';
    return;
  }
  log.innerHTML = events.map(_eventHtml).join('');
  log.scrollTop = log.scrollHeight;
}

const _EVENT_ICONS = {
  status: '�"|',
  planning: '�Y-',
  plan_ready: '�S�',
  approved: '�o.',
  step_started: '�-�',
  step_completed: '�o.',
  step_failed: '�o-',
  step_retry: '�Y"',
  step_incomplete: '�a�',
  tool_start: '�_�',
  tool_output: '�?',
  agent_round: '�Y�',
  context_reset: '�Y�',
  verify: '�o"',
  blocked: '�Y�',
  completed: '�Y�',
  cancelled: '�-',
  retry: '�Y"',
  paused: '�?�',
  resumed: '�-¶',
  recovered: '�Y�',
  after_action: '�?"',
  failed: '�o-',
};

function _eventHtml(event) {
  const type = event.type || 'log';
  const icon = _EVENT_ICONS[type] || '�?�';
  const isolation = event.data?.isolation && type === 'context_reset'
    ? `<div class="sarge-log-note">${_esc(event.data.isolation)}</div>` : '';
  return `
    <div class="sarge-log-row sarge-log-${_esc(type)}">
      <span class="sarge-log-icon">${icon}</span>
      <span class="sarge-log-time">${_esc((event.time || '').replace('T', ' ').slice(0, 19))}</span>
      <span class="sarge-log-msg">${_esc(event.message || type)}</span>
      ${isolation}
    </div>
  `;
}

async function _renderArtifacts(id) {
  const node = document.getElementById('sarge-artifacts');
  if (!node) return;
  try {
    const items = await api.listArtifacts(id);
    if (!items.length) {
      node.innerHTML = '<div class="sarge-empty-small">No files yet.</div>';
      return;
    }
    node.innerHTML = items.map((item) => `
      <a class="sarge-artifact" href="${_esc(api.artifactUrl(id, item.path))}" target="_blank" rel="noopener">
        <span class="sarge-artifact-path">${_esc(item.path)}</span>
        <span class="sarge-artifact-size">${_esc(_bytes(item.size))}</span>
      </a>
    `).join('');
  } catch (_) {
    node.innerHTML = '<div class="sarge-empty-small">Could not load files.</div>';
  }
}

// ------------------------------------------------------------------
// Wiring
// ------------------------------------------------------------------

function _wire(root) {
  root.querySelector('#sarge-panel-close')?.addEventListener('click', closePanel);
  root.querySelector('#sarge-panel-minimize')?.addEventListener('click', () => {
    const overlay = document.getElementById('sarge-overlay');
    if (overlay) overlay.style.display = 'none';
  });
  root.querySelector('#sarge-refresh')?.addEventListener('click', () => {
    _refresh();
    if (_selectedId) _select(_selectedId);
  });
  root.querySelector('#sarge-new')?.addEventListener('click', () => {
    const form = document.getElementById('sarge-create-form');
    if (form) form.style.display = '';
    _loadCreateOptions();
  });
  root.querySelector('#sarge-create-cancel')?.addEventListener('click', () => {
    const form = document.getElementById('sarge-create-form');
    if (form) form.style.display = 'none';
  });

  const list = root.querySelector('#sarge-list');
  list?.addEventListener('click', (e) => {
    const item = e.target.closest('.sarge-list-item');
    if (item) _select(item.dataset.id);
  });

  const detail = root.querySelector('#sarge-detail');
  detail?.addEventListener('click', async (e) => {
    const actionBtn = e.target.closest('[data-action]');
    if (!actionBtn || !_selectedId || _busy) return;
    await _runAction(actionBtn.dataset.action);
  });

  root.querySelector('#sarge-create-form')?.addEventListener('submit', _submitCreate);
}

async function _loadCreateOptions() {
  const select = document.getElementById('sarge-source-session');
  if (select && select.dataset.loaded !== '1') {
    select.dataset.loaded = '1';
    try {
      const sources = await api.listResearchSources();
      select.innerHTML = '<option value="">(none)</option>' + sources.map((s) =>
        `<option value="${_esc(s.session_id)}">${_esc(s.query)} (${_esc(_fmt(s.created_at))})</option>`
      ).join('');
    } catch (_) {
      select.innerHTML = '<option value="">(none)</option>';
    }
  }
  await _loadModels();
}

async function _loadModels() {
  const select = document.getElementById('sarge-model');
  if (!select || select.dataset.loaded === '1') return;
  try {
    const res = await fetch(`${_apiBase}/api/models`, { credentials: 'same-origin' });
    const data = await res.json();
    const options = [];
    for (const group of data?.items || []) {
      for (const model of group.models || []) {
        const id = `${group.endpoint_id || ''}::${model}`;
        options.push(`<option value="${_esc(id)}" data-endpoint="${_esc(group.endpoint_id || '')}" data-name="${_esc(group.endpoint_name || '')}">${_esc(group.endpoint_name ? `${group.endpoint_name} �?" ${model}` : model)}</option>`);
      }
    }
    select.innerHTML = options.length ? options.join('') : '<option value="">No models configured</option>';
    select.dataset.loaded = '1';
  } catch (_) {
    select.innerHTML = '<option value="">Could not load models</option>';
  }
}

async function _submitCreate(e) {
  e.preventDefault();
  const objective = document.getElementById('sarge-objective')?.value?.trim();
  const error = document.getElementById('sarge-create-error');
  if (!objective) {
    if (error) { error.textContent = 'An objective is required.'; error.style.display = ''; }
    return;
  }
  if (error) error.style.display = 'none';
  const modelSelect = document.getElementById('sarge-model');
  const selected = modelSelect?.selectedOptions?.[0];
  const body = {
    objective,
    brief: document.getElementById('sarge-brief')?.value?.trim() || null,
    research_session_id: document.getElementById('sarge-source-session')?.value || null,
    privacy: document.getElementById('sarge-privacy')?.value || 'local',
    autonomy: document.getElementById('sarge-unattended')?.checked ? 'unattended' : 'review',
    endpoint_id: selected?.dataset?.endpoint || null,
    endpoint_name: selected?.dataset?.name || null,
    model: modelSelect?.value?.split('::').slice(1).join('::') || null,
  };
  _busy = true;
  try {
    const campaign = await api.createCampaign(body);
    document.getElementById('sarge-create-form').style.display = 'none';
    document.getElementById('sarge-objective').value = '';
    await _refresh();
    await _select(campaign.id);
  } catch (err) {
    if (error) { error.textContent = err.message || String(err); error.style.display = ''; }
  } finally {
    _busy = false;
  }
}

async function _runAction(action) {
  const id = _selectedId;
  if (!id) return;
  _busy = true;
  try {
    if (action === 'delete') {
      if (!window.confirm('Delete this campaign and its workspace? This cannot be undone.')) return;
      await api.deleteCampaign(id);
      _selectedId = null;
      _detail = null;
      await _refresh();
      _renderDetail();
      return;
    }
    if (action === 'plan') {
      _detail = await api.planCampaign(id);
      _renderDetail();
      _syncStream();
      _refresh();
      return;
    }
    if (action === 'approve') {
      _detail = await api.approveCampaign(id);
      _renderDetail();
      _syncStream();
      _refresh();
      return;
    }
    if (action === 'pause') _detail = await api.pauseCampaign(id);
    else if (action === 'resume') _detail = await api.resumeCampaign(id);
    else if (action === 'cancel') _detail = await api.cancelCampaign(id);
    else if (action === 'retry') _detail = await api.retryCampaign(id);
    _renderDetail();
    _syncStream();
    _refresh();
  } catch (err) {
    uiModule.showError?.(`Sarge: ${err.message || err}`);
  } finally {
    _busy = false;
    if (_selectedId === id) _renderArtifacts(id);
  }
}

// Re-render artifacts after detail render (it is loaded async).
const _origRenderDetail = _renderDetail;
_renderDetail = function patchedRenderDetail() {
  _origRenderDetail();
  if (_selectedId && _detail?.plan) _renderArtifacts(_selectedId);
};

// ------------------------------------------------------------------
// Icons
// ------------------------------------------------------------------

const _shieldIcon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
const _refreshIcon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>';
const _minusIcon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="5" y1="18" x2="19" y2="18"/></svg>';

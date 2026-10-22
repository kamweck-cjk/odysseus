/**
 * Sarge API client.
 *
 * Thin wrappers over /api/sarge/*. Every call uses the same-origin session
 * cookie, matching the rest of the app's fetch helpers.
 */

let _base = '';

export function init(apiBase) {
  _base = apiBase || '';
}

function _url(path) {
  return `${_base}${path}`;
}

async function _json(res) {
  let payload = null;
  try {
    payload = await res.json();
  } catch (_) {
    payload = null;
  }
  if (!res.ok) {
    const detail = payload && (payload.detail || payload.error || payload.message);
    const error = new Error(detail || `HTTP ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return payload;
}

export async function listCampaigns() {
  const res = await fetch(_url('/api/sarge/campaigns'), { credentials: 'same-origin' });
  const data = await _json(res);
  return data?.items || [];
}

export async function getCampaign(id) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}`), {
    credentials: 'same-origin',
  });
  return _json(res);
}

export async function createCampaign(body) {
  const res = await fetch(_url('/api/sarge/campaigns'), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return _json(res);
}

export async function updateCampaign(id, body) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}`), {
    method: 'PATCH',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return _json(res);
}

export async function deleteCampaign(id) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}`), {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  return _json(res);
}

export async function planCampaign(id) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}/plan`), {
    method: 'POST',
    credentials: 'same-origin',
  });
  return _json(res);
}

export async function approveCampaign(id, plan) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}/approve`), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(plan ? { plan } : {}),
  });
  return _json(res);
}

export async function startCampaign(id) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}/start`), {
    method: 'POST',
    credentials: 'same-origin',
  });
  return _json(res);
}

async function _action(id, action) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}/${action}`), {
    method: 'POST',
    credentials: 'same-origin',
  });
  return _json(res);
}

export async function pauseCampaign(id) { return _action(id, 'pause'); }
export async function resumeCampaign(id) { return _action(id, 'resume'); }
export async function cancelCampaign(id) { return _action(id, 'cancel'); }
export async function retryCampaign(id) { return _action(id, 'retry'); }

export async function listArtifacts(id) {
  const res = await fetch(_url(`/api/sarge/campaigns/${encodeURIComponent(id)}/artifacts`), {
    credentials: 'same-origin',
  });
  const data = await _json(res);
  return data?.items || [];
}

export function artifactUrl(id, path, download = false) {
  const encoded = String(path)
    .split('/')
    .map(encodeURIComponent)
    .join('/');
  const suffix = download ? '?download=true' : '';
  return _url(`/api/sarge/campaigns/${encodeURIComponent(id)}/artifacts/${encoded}${suffix}`);
}

export async function listResearchSources() {
  const res = await fetch(_url('/api/sarge/research-sources'), { credentials: 'same-origin' });
  const data = await _json(res);
  return data?.items || [];
}

export function eventStreamUrl(id, after = 0) {
  return _url(
    `/api/sarge/campaigns/${encodeURIComponent(id)}/events?follow=true&after=${Number(after) || 0}`,
  );
}

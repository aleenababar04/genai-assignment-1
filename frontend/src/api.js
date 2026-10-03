// Small helpers for talking to the FastAPI backend.
//
// API_BASE is empty by default, so requests go to the same origin ("/api/...").
// The Vite dev server (vite.config.js) and nginx (nginx.conf) forward /api to the backend.
// Set VITE_API_BASE (e.g. "http://localhost:8000") to call another server directly.
export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

// Turn a backend path such as "/api/samples/pet-x.png" into a full URL.
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

// Send a request and return the JSON body. Any failure throws an Error whose
// message can be shown to the user directly.
async function request(path, options) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch {
    throw new Error("Cannot reach the backend. Check that the server is running and try again.");
  }

  // The backend sends errors as {"detail": ...}. Read the body as text first,
  // because a proxy error page (e.g. nginx 502) is not JSON.
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }

  if (!response.ok) {
    throw new Error(errorMessage(response.status, body));
  }
  return body;
}

// Build a readable message from an error response.
function errorMessage(status, body) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  // FastAPI validation errors (422) are a list of {loc, msg, ...}.
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join("; ");
  if (status === 502 || status === 504) {
    return "Cannot reach the backend. Check that the server is running and try again.";
  }
  return `Request failed with status ${status}.`;
}

// Build the multipart form for an image request.
// `source` is either {file: File} (an upload / webcam capture) or {sampleId: "pet-..."}.
// `fields` are extra form fields; empty values (e.g. a blank seed) are left out.
function buildForm(source, fields) {
  const form = new FormData();
  if (source.file) form.append("file", source.file);
  else form.append("sample_id", source.sampleId);
  for (const [key, value] of Object.entries(fields)) {
    if (value !== "" && value !== null && value !== undefined) form.append(key, String(value));
  }
  return form;
}

// GET /api/health -> {status, models, loaded_count, expected_count, last_inference_ms}
export function getHealth() {
  return request("/api/health");
}

// GET /api/samples -> [{id, name, kind: "pet" | "face", url}]
export function getSamples() {
  return request("/api/samples");
}

// The three restoration endpoints share the same form fields.
export const RESTORE_ENDPOINTS = {
  universal: "/api/restore/universal",
  hardRouting: "/api/restore/hard-routing",
  softMoe: "/api/restore/soft-moe",
};

// POST one of the restoration endpoints.
// Response: {input_image, output_image, inference_ms, corruption, ...extra fields}.
export function restoreImage(endpoint, source, { corruption, severity, seed }) {
  return request(endpoint, {
    method: "POST",
    body: buildForm(source, { corruption, severity, seed }),
  });
}

// POST /api/sketch -> {input_image, output_image, style, inference_ms}
export function generateSketch(source, style) {
  return request("/api/sketch", {
    method: "POST",
    body: buildForm(source, { style }),
  });
}

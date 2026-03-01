const BASE = "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const api = {
  getState: () => request("/api/state/"),

  upload: (file) => {
    const form = new FormData();
    form.append("myfile", file);
    return request("/api/upload/", { method: "POST", body: form });
  },

  hybridUploadLow: (file) => {
    const form = new FormData();
    form.append("img_low", file);
    return request("/api/hybrid-low/", { method: "POST", body: form });
  },

  hybridUploadHigh: (file) => {
    const form = new FormData();
    form.append("img_high", file);
    return request("/api/hybrid-high/", { method: "POST", body: form });
  },

  hybridMix: () =>
    request("/api/hybrid-mix/", { method: "POST" }),

  applyFilter: (payload) =>
    request("/api/apply-filter/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  undo: () => request("/api/undo/", { method: "POST" }),

  reset: () => request("/api/reset/", { method: "POST" }),

  drawHistogram: () => request("/api/histogram/", { method: "POST" }),

  equalize: () => request("/api/equalize/", { method: "POST" }),

  switchMode: (mode) =>
    request("/api/switch-mode/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
};

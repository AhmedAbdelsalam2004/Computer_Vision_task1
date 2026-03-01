import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "./api";
import FilterPanel from "./FilterPanel";
import HistogramPanel from "./HistogramPanel";
import "./index.css";

const EMPTY_STATE = {
  mode: "filter",
  current_url: null,
  original_url: null,
  can_undo: false,
  has_image: false,
  filter_name: "",
  kernel_size: 3,
  canny_low: 50,
  canny_high: 150,
  noise_amount: 5,
  hist_orig_data: null,
  hist_eq_data: null,
  hist_eq_url: null,
  history_length: 0,
  hybrid_low_url: null,
  hybrid_high_url: null,
  hybrid_out_url: null,
};

export default function App() {
  const [state, setState] = useState(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");

  // Filter UI state (local, mirrors backend)
  const [filterName, setFilterName] = useState("");
  const [kernelSize, setKernelSize] = useState(3);
  const [cannyLow, setCannyLow] = useState(50);
  const [cannyHigh, setCannyHigh] = useState(150);
  const [noiseAmount, setNoiseAmount] = useState(5);
  const [category, setCategory] = useState("filters");
  const [tab, setTab] = useState("lpf");

  // Hybrid image local state
  const hybridLowInputRef = useRef(null);
  const hybridHighInputRef = useRef(null);
  const [hybridLowFile, setHybridLowFile] = useState(null);
  const [hybridHighFile, setHybridHighFile] = useState(null);
  const [hybridLowName, setHybridLowName] = useState("");
  const [hybridHighName, setHybridHighName] = useState("");

  const fetchState = useCallback(async () => {
    try {
      const data = await api.getState();
      setState(data);
      if (data.kernel_size) setKernelSize(data.kernel_size);
      if (data.canny_low) setCannyLow(data.canny_low);
      if (data.canny_high) setCannyHigh(data.canny_high);
      if (data.noise_amount) setNoiseAmount(data.noise_amount);
      if (data.filter_name) setFilterName(data.filter_name);
    } catch {
      // fresh session, use defaults
    }
  }, []);

  useEffect(() => { fetchState(); }, [fetchState]);

  const withLoading = async (fn) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      setState(prev => ({ ...prev, ...data }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setFileName(file.name);
  };

  const handleUpload = async () => {
    const file = selectedFile || fileInputRef.current?.files?.[0];
    if (!file) {
      setError("Please select an image file first.");
      return;
    }
    await withLoading(() => api.upload(file));
    setFilterName("");
    setSelectedFile(null);
  };

  const handleApplyFilter = (payload) =>
    withLoading(() => api.applyFilter({
      ...payload,
      kernel_size: kernelSize,
      canny_low: cannyLow,
      canny_high: cannyHigh,
      noise_amount: noiseAmount,
    }));

  const handleUndo = () => withLoading(api.undo);
  const handleReset = () => withLoading(() => api.reset().then(d => { setFilterName(""); return d; }));

  const handleHistogram = () =>
    withLoading(() => api.drawHistogram().then(d => ({ ...state, ...d })));

  const handleEqualize = () =>
    withLoading(() => api.equalize().then(d => ({ ...state, ...d })));

  const handleSwitchMode = (mode) =>
    withLoading(() => api.switchMode(mode).then(() => ({ mode })));

  // Hybrid handlers
  const handleHybridLowSelect = (file) => {
    if (!file) return;
    setHybridLowFile(file);
    setHybridLowName(file.name);
  };

  const handleHybridHighSelect = (file) => {
    if (!file) return;
    setHybridHighFile(file);
    setHybridHighName(file.name);
  };

  const handleHybridLowUpload = async () => {
    const file = hybridLowFile || hybridLowInputRef.current?.files?.[0];
    if (!file) {
      setError("Please select the first image for hybrid (low-pass).");
      return;
    }
    await withLoading(() => api.hybridUploadLow(file));
  };

  const handleHybridHighUpload = async () => {
    const file = hybridHighFile || hybridHighInputRef.current?.files?.[0];
    if (!file) {
      setError("Please select the second image for hybrid (high-pass).");
      return;
    }
    await withLoading(() => api.hybridUploadHigh(file));
  };

  const handleHybridMix = () =>
    withLoading(api.hybridMix);

  const isHistMode = state.mode === "histogram";
  const isHybridMode = state.mode === "hybrid";

  return (
    <div className={`app-root ${isHistMode ? "mode-hist" : ""}`}>
      {/* ── Topbar ────────────────────────────────────────────── */}
      <nav className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark" />
          <span className="brand-name">Vision<span>Lab</span></span>
        </div>

        <div className="mode-switch">
          <button
            className={`mode-btn ${state.mode === "filter" ? "active-filter" : ""}`}
            onClick={() => handleSwitchMode("filter")}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
            Filter
          </button>
          <button
            className={`mode-btn ${isHistMode ? "active-hist" : ""}`}
            onClick={() => handleSwitchMode("histogram")}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
            Histogram
          </button>
          <button
            className={`mode-btn ${isHybridMode ? "active-hybrid" : ""}`}
            onClick={() => handleSwitchMode("hybrid")}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 20L10 4l4 16 6-10" /></svg>
            Hybrid Image
          </button>
        </div>

        <span className="topbar-meta">COMPUTER VISION · CV LAB</span>
      </nav>

      {/* ── Error Banner ───────────────────────────────────────── */}
      {error && (
        <div className="err-wrap">
          <div className="err-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
            {error}
            <button className="err-close" onClick={() => setError(null)}>×</button>
          </div>
        </div>
      )}

      <div className="page">
        {/* ── Sidebar ─────────────────────────────────────────── */}
        <aside className="sidebar">

          {/* Upload Block (hidden in Hybrid mode) */}
          {!isHybridMode && (
            <div className="block">
              <div className="block-header">
                <svg className="block-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span className="block-title">Upload Image</span>
              </div>
              <div className="block-body">
                <div
                  className={`drop-zone ${selectedFile ? "has-file" : ""}`}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add("drag"); }}
                  onDragLeave={e => e.currentTarget.classList.remove("drag")}
                  onDrop={e => {
                    e.preventDefault();
                    e.currentTarget.classList.remove("drag");
                    const file = e.dataTransfer.files[0];
                    if (file) handleFileSelect(file);
                  }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={e => handleFileSelect(e.target.files[0])}
                  />
                  <svg className="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
                  </svg>
                  {selectedFile ? (
                    <>
                      <p className="dz-label" style={{ color: "var(--cyan)" }}><strong>✓ Ready to upload</strong></p>
                      <div id="fname">{fileName}</div>
                      <p className="dz-formats" style={{ marginTop: 4 }}>Click to change file</p>
                    </>
                  ) : (
                    <>
                      <p className="dz-label"><strong>Click to browse</strong> or drag &amp; drop</p>
                      <p className="dz-formats">PNG &nbsp;·&nbsp; JPG &nbsp;·&nbsp; WEBP</p>
                    </>
                  )}
                </div>
                <button
                  className={`btn btn-primary ${loading ? "loading" : ""}`}
                  onClick={handleUpload}
                  disabled={loading}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="16 16 12 12 8 16" /><line x1="12" y1="12" x2="12" y2="21" /><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" /></svg>
                  Upload
                </button>
              </div>
            </div>
          )}

          {/* Filter/Histogram/Hybrid Sidebar Controls */}
          {!isHistMode && !isHybridMode && (
            <div className="panel-filter block">
              <div className="block-header">
                <svg className="block-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
                <span className="block-title">Apply Filter / Noise</span>
              </div>
              <div className="block-body">
                <div className="cat-switch">
                  <button className={`cat-btn ${category === "filters" ? "active" : ""}`} onClick={() => setCategory("filters")}>Filters</button>
                  <button className={`cat-btn ${category === "noise" ? "active" : ""}`} onClick={() => setCategory("noise")}>Noise</button>
                </div>

                {category === "filters" && (
                  <>
                    <div className="tabs">
                      <button className={`tab ${tab === "lpf" ? "on" : ""}`} onClick={() => setTab("lpf")}>Low-Pass</button>
                      <button className={`tab ${tab === "hpf" ? "on" : ""}`} onClick={() => setTab("hpf")}>High-Pass</button>
                    </div>
                    <div className="fgroup show">
                      {tab === "lpf" && [
                        { id: "box", name: "Box Filter", desc: "K = 1/9 × ones(3,3)" },
                        { id: "gaussian", name: "Gaussian Filter", desc: "Binomial outer product" },
                      ].map(f => (
                        <div key={f.id} className={`fcard lpf ${filterName === f.id ? "sel" : ""}`} onClick={() => setFilterName(f.id)}>
                          <div className="fcard-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4" /></svg>
                          </div>
                          <div className="fcard-info">
                            <span className="fcard-name">{f.name}</span>
                            <span className="fcard-kernel">{f.desc}</span>
                          </div>
                          <div className="sel-indicator" />
                        </div>
                      ))}
                      {tab === "hpf" && [
                        { id: "sobel", name: "Sobel Filter", desc: "Generalised Sobel, size k×k" },
                        { id: "prewitt", name: "Prewitt Filter", desc: "Generalised Prewitt, size k×k" },
                        { id: "roberts", name: "Roberts Cross", desc: "Fixed 2×2 (OpenCV)" },
                        { id: "canny", name: "Canny Detector", desc: "Gaussian + NMS (OpenCV)" },
                      ].map(f => (
                        <div key={f.id} className={`fcard hpf ${filterName === f.id ? "sel" : ""}`} onClick={() => setFilterName(f.id)}>
                          <div className="fcard-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4" /></svg>
                          </div>
                          <div className="fcard-info">
                            <span className="fcard-name">{f.name}</span>
                            <span className="fcard-kernel">{f.desc}</span>
                          </div>
                          <div className="sel-indicator" />
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {category === "noise" && (
                  <div className="fgroup show">
                    {[
                      { id: "uniform_noise", name: "Uniform Noise", desc: "cv2.randu" },
                      { id: "gaussian_noise", name: "Gaussian Noise", desc: "cv2.randn" },
                      { id: "salt_pepper_noise", name: "Salt & Pepper", desc: "Probability Matrix" },
                    ].map(f => (
                      <div key={f.id} className={`fcard lpf ${filterName === f.id ? "sel" : ""}`} onClick={() => setFilterName(f.id)}>
                        <div className="fcard-icon">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4" /></svg>
                        </div>
                        <div className="fcard-info">
                          <span className="fcard-name">{f.name}</span>
                          <span className="fcard-kernel">{f.desc}</span>
                        </div>
                        <div className="sel-indicator" />
                      </div>
                    ))}
                  </div>
                )}

                <div className="divider" />

                {filterName === "roberts" && (
                  <div className="roberts-note show">⚡ Roberts uses fixed 2×2 kernels — kernel size is ignored</div>
                )}

                {category === "filters" && (
                  <div className="kernel-row">
                    <label>Kernel Size</label>
                    <div className="kernel-stepper">
                      <button className="kstep-btn" onClick={() => setKernelSize(k => Math.max(3, k - 2))}>−</button>
                      <div id="kernelSizeDisplay">{kernelSize}</div>
                      <button className="kstep-btn" onClick={() => setKernelSize(k => Math.min(21, k + 2))}>+</button>
                    </div>
                    <span className="kernel-tag">{kernelSize}×{kernelSize}</span>
                  </div>
                )}

                {filterName === "canny" && (
                  <div className="canny-controls show">
                    <div className="canny-title">Hysteresis Thresholds</div>
                    <div className="canny-row">
                      <label>Low</label>
                      <input type="range" min="0" max="254" value={cannyLow} onChange={e => setCannyLow(+e.target.value)} />
                      <span className="canny-val">{cannyLow}</span>
                    </div>
                    <div className="canny-row">
                      <label>High</label>
                      <input type="range" min="1" max="255" value={cannyHigh} onChange={e => setCannyHigh(+e.target.value)} />
                      <span className="canny-val">{cannyHigh}</span>
                    </div>
                  </div>
                )}

                {category === "noise" && (
                  <div className="noise-controls show">
                    <div className="canny-title">Noise Intensity</div>
                    <div className="canny-row">
                      <label>Level</label>
                      <input type="range" min="1" max="100" value={noiseAmount} onChange={e => setNoiseAmount(+e.target.value)} />
                      <span className="canny-val">{noiseAmount}%</span>
                    </div>
                  </div>
                )}

                <button
                  className={`btn btn-primary ${loading ? "loading" : ""}`}
                  disabled={!filterName || !state.has_image || loading}
                  onClick={() => handleApplyFilter({ filter_name: filterName })}
                >
                  {loading ? <span className="spinner" /> : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
                  )}
                  {loading ? "Processing…" : category === "noise" ? "Apply Noise" : "Apply Filter"}
                </button>

                <div className="act-row">
                  <button className="btn btn-secondary" disabled={!state.can_undo || loading} onClick={handleUndo}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 14 4 9 9 4" /><path d="M20 20v-7a4 4 0 0 0-4-4H4" /></svg>
                    Undo
                  </button>
                  <button className="btn btn-outline-red" disabled={!state.has_image || loading} onClick={handleReset}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.36" /></svg>
                    Reset
                  </button>
                </div>
              </div>
            </div>
          )}

          {isHistMode && (
            <div className="panel-hist block">
              <div className="block-header">
                <svg className="block-header-icon green" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                <span className="block-title">Histogram Tools</span>
              </div>
              <div className="block-body">
                <div style={{ background: "var(--surf2)", border: "1px solid var(--border)", borderLeft: "3px solid var(--green)", borderRadius: 6, padding: "12px 14px", marginBottom: 14 }}>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: ".62rem", color: "var(--green)", letterSpacing: 1, marginBottom: 6 }}>FORMULA</div>
                  <div style={{ fontSize: ".8rem", color: "var(--text2)", lineHeight: 1.6 }}>
                    f(i) = ⌊ CDF(i) / n × (L−1) ⌋
                  </div>
                  <div style={{ fontSize: ".73rem", color: "var(--text3)", marginTop: 6, lineHeight: 1.5 }}>
                    Redistributes pixel intensities across the full [0,255] range.
                  </div>
                </div>
                <button className={`btn btn-purple ${loading ? "loading" : ""}`} disabled={!state.has_image || loading} onClick={handleHistogram} style={{ marginBottom: 10 }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="10" width="4" height="11" /><rect x="10" y="6" width="4" height="15" /><rect x="17" y="3" width="4" height="18" /></svg>
                  Draw Histogram
                </button>
                <button className={`btn btn-green ${loading ? "loading" : ""}`} disabled={!state.has_image || loading} onClick={handleEqualize} style={{ marginBottom: 10 }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10" /><path d="M12 20V4" /><path d="M6 20v-6" /></svg>
                  Equalize Contrast
                </button>
                <button className="btn btn-outline-red" disabled={!state.has_image || loading} onClick={handleReset}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.36" /></svg>
                  Reset
                </button>
              </div>
            </div>
          )}

          {isHybridMode && (
            <div className="panel-filter block">
              <div className="block-header">
                <svg className="block-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M4 20L10 4l4 16 6-10" />
                </svg>
                <span className="block-title">Hybrid Image Builder</span>
              </div>
              <div className="block-body">
                <div className="hybrid-upload-group">
                  <div className="hybrid-upload">
                    <div className="hybrid-label">Low-Pass Image (Box 5×5)</div>
                    <div
                      className={`drop-zone ${hybridLowFile ? "has-file" : ""}`}
                      onClick={() => hybridLowInputRef.current?.click()}
                    >
                      <input
                        ref={hybridLowInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: "none" }}
                        onChange={e => handleHybridLowSelect(e.target.files[0])}
                      />
                      <svg className="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
                      </svg>
                      {hybridLowFile ? (
                        <>
                          <p className="dz-label" style={{ color: "var(--cyan)" }}><strong>✓ Ready</strong></p>
                          <div>{hybridLowName}</div>
                        </>
                      ) : (
                        <>
                          <p className="dz-label"><strong>Click to choose</strong> first image</p>
                        </>
                      )}
                    </div>
                    <button
                      className={`btn btn-primary ${loading ? "loading" : ""}`}
                      onClick={handleHybridLowUpload}
                      disabled={loading}
                      style={{ marginTop: 8 }}
                    >
                      Apply Low-Pass
                    </button>
                  </div>

                  <div className="hybrid-upload" style={{ marginTop: 16 }}>
                    <div className="hybrid-label">High-Pass Image (Sobel 5×5)</div>
                    <div
                      className={`drop-zone ${hybridHighFile ? "has-file" : ""}`}
                      onClick={() => hybridHighInputRef.current?.click()}
                    >
                      <input
                        ref={hybridHighInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: "none" }}
                        onChange={e => handleHybridHighSelect(e.target.files[0])}
                      />
                      <svg className="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
                      </svg>
                      {hybridHighFile ? (
                        <>
                          <p className="dz-label" style={{ color: "var(--cyan)" }}><strong>✓ Ready</strong></p>
                          <div>{hybridHighName}</div>
                        </>
                      ) : (
                        <>
                          <p className="dz-label"><strong>Click to choose</strong> second image</p>
                        </>
                      )}
                    </div>
                    <button
                      className={`btn btn-primary ${loading ? "loading" : ""}`}
                      onClick={handleHybridHighUpload}
                      disabled={loading}
                      style={{ marginTop: 8 }}
                    >
                      Apply High-Pass
                    </button>
                  </div>
                </div>

                <div className="divider" />

                <button
                  className={`btn btn-purple ${loading ? "loading" : ""}`}
                  disabled={loading || !state.hybrid_low_url || !state.hybrid_high_url}
                  onClick={handleHybridMix}
                >
                  Mix Hybrid Image
                </button>
              </div>
            </div>
          )}
        </aside>

        {/* ── Main Content ─────────────────────────────────────── */}
        <section className="right">
          {isHistMode && (
            <HistogramPanel state={state} onHistogram={handleHistogram} onEqualize={handleEqualize} loading={loading} />
          )}

          {!isHistMode && !isHybridMode && (
            <div className="right-filter">
              {state.current_url ? (
                <>
                  <div className="status-bar">
                    <div className="status-item">
                      <div className="status-dot active" />
                      <span className="status-label">STATUS</span>
                      <span className="status-val">Image loaded</span>
                    </div>
                    <span className="status-sep">/</span>
                    <div className="status-item">
                      <span className="status-label">FILTER</span>
                      <span className="status-val">{state.filter_name ? state.filter_name.toUpperCase() : "NONE"}</span>
                    </div>
                    <span className="status-sep">/</span>
                    <div className="status-item">
                      <span className="status-label">HISTORY</span>
                      <span className="status-val">{state.history_length} step{state.history_length !== 1 ? "s" : ""}</span>
                    </div>
                  </div>

                  <div className="img-grid">
                    <ImageBox label="Source" badge="b-src" badgeText="ORIGINAL" url={state.original_url || state.current_url} />
                    <ImageBox
                      label="Output"
                      badge={getBadge(state.filter_name)}
                      badgeText={state.filter_name ? state.filter_name.toUpperCase() : "PREVIEW"}
                      url={state.current_url}
                      downloadable
                    />
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
                  </svg>
                  <div className="empty-title">No Image Loaded</div>
                  <div className="empty-sub">Upload an image from the sidebar to get started</div>
                </div>
              )}
            </div>
          )}

          {isHybridMode && (
            <div className="right-filter">
              {state.hybrid_low_url || state.hybrid_high_url || state.hybrid_out_url ? (
                <div className="img-grid" style={{ gridTemplateColumns: "1fr 1.4fr" }}>
                  <div>
                    <ImageBox
                      label="Low-Pass"
                      badge="b-lpf"
                      badgeText="BOX 5×5"
                      url={state.hybrid_low_url}
                    />
                    <ImageBox
                      label="High-Pass"
                      badge="b-hpf"
                      badgeText="SOBEL 5×5"
                      url={state.hybrid_high_url}
                    />
                  </div>
                  <div className="ibox">
                    <div className="ibox-head">
                      <span className="ibox-label">Hybrid Output</span>
                      <span className="badge b-prev">HYBRID</span>
                    </div>
                    <div className="ibox-frame">
                      {state.hybrid_out_url ? (
                        <img src={resolveUrl(state.hybrid_out_url)} alt="Hybrid" />
                      ) : (
                        <div className="empty-sub">Upload both images and click Mix Hybrid Image</div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
                  </svg>
                  <div className="empty-title">No Hybrid Image</div>
                  <div className="empty-sub">Upload two images in the sidebar to build a hybrid image</div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-ring" />
        </div>
      )}
    </div>
  );
}

function getBadge(filter) {
  if (!filter) return "b-prev";
  if (["box", "gaussian"].includes(filter)) return "b-lpf";
  if (["uniform_noise", "gaussian_noise", "salt_pepper_noise"].includes(filter)) return "b-hist";
  return "b-hpf";
}

const MEDIA_BASE = "http://localhost:8000";
function resolveUrl(url) {
  if (!url) return null;
  return url.startsWith("http") ? url : `${MEDIA_BASE}${url}`;
}

function ImageBox({ label, badge, badgeText, url, downloadable }) {
  return (
    <div className="ibox">
      <div className="ibox-head">
        <span className="ibox-label">{label}</span>
        <span className={`badge ${badge}`}>{badgeText}</span>
      </div>
      <div className="ibox-frame">
        {url && <img src={resolveUrl(url)} alt={label} />}
      </div>
      {downloadable && url && (
        <div style={{ padding: 10, textAlign: "right" }}>
          <a href={resolveUrl(url)} download className="btn-save">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
            Download
          </a>
        </div>
      )}
    </div>
  );
}

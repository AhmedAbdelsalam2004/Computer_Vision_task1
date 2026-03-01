const MEDIA_BASE = "http://127.0.0.1:8000";

function resolveUrl(url) {
    if (!url) return null;
    return url.startsWith("http") ? url : `${MEDIA_BASE}${url}`;
}

export default function FilterPanel({ state, onApplyFilter, onUndo, onReset, loading,
    filterName, setFilterName, kernelSize, setKernelSize,
    cannyLow, setCannyLow, cannyHigh, setCannyHigh,
    noiseAmount, setNoiseAmount, category, setCategory, tab, setTab }) {

    const { current_url, original_url, can_undo, has_image, filter_name } = state;

    const filterCards = {
        lpf: [
            { id: "box", name: "Box Filter", desc: "K = 1/9 × ones(3,3)" },
            { id: "gaussian", name: "Gaussian Filter", desc: "Binomial outer product" },
        ],
        hpf: [
            { id: "sobel", name: "Sobel Filter", desc: "Generalised Sobel, size k×k" },
            { id: "prewitt", name: "Prewitt Filter", desc: "Generalised Prewitt, size k×k" },
            { id: "roberts", name: "Roberts Cross", desc: "Fixed 2×2 (OpenCV)" },
            { id: "canny", name: "Canny Detector", desc: "Gaussian + NMS (OpenCV)" },
        ],
        noise: [
            { id: "uniform_noise", name: "Uniform Noise", desc: "cv2.randu" },
            { id: "gaussian_noise", name: "Gaussian Noise", desc: "cv2.randn" },
            { id: "salt_pepper_noise", name: "Salt & Pepper", desc: "Probability Matrix" },
        ],
    };

    const isNoise = ["uniform_noise", "gaussian_noise", "salt_pepper_noise"].includes(filterName);
    const isRoberts = filterName === "roberts";
    const isCanny = filterName === "canny";

    function handleApply() {
        if (!filterName) return;
        onApplyFilter({
            filter_name: filterName,
            kernel_size: kernelSize,
            canny_low: cannyLow,
            canny_high: cannyHigh,
            noise_amount: noiseAmount,
        });
    }

    const getBadgeClass = (f) => {
        if (!f) return "b-prev";
        if (["box", "gaussian"].includes(f)) return "b-lpf";
        if (["uniform_noise", "gaussian_noise", "salt_pepper_noise"].includes(f)) return "b-hist";
        return "b-hpf";
    };

    return (
        <div className="right-filter">
            {current_url ? (
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
                            <span className="status-val">{filter_name ? filter_name.toUpperCase() : "NONE"}</span>
                        </div>
                        <span className="status-sep">/</span>
                        <div className="status-item">
                            <span className="status-label">HISTORY</span>
                            <span className="status-val">{state.history_length} step{state.history_length !== 1 ? "s" : ""}</span>
                        </div>
                    </div>

                    <div className="img-grid">
                        <div className="ibox">
                            <div className="ibox-head">
                                <span className="ibox-label">Source</span>
                                <span className="badge b-src">ORIGINAL</span>
                            </div>
                            <div className="ibox-frame">
                                <img src={resolveUrl(original_url || current_url)} alt="Original" />
                            </div>
                            {original_url && current_url !== original_url && (
                                <div style={{ padding: 10, textAlign: "right" }}>
                                    <a href={resolveUrl(original_url)} download className="btn-save">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                                        Download
                                    </a>
                                </div>
                            )}
                        </div>

                        <div className="ibox">
                            <div className="ibox-head">
                                <span className="ibox-label">Output</span>
                                <span className={`badge ${getBadgeClass(filter_name)}`}>
                                    {filter_name ? filter_name.toUpperCase() : "PREVIEW"}
                                </span>
                            </div>
                            <div className="ibox-frame">
                                <img src={resolveUrl(current_url)} alt="Filtered output" />
                            </div>
                            <div style={{ padding: 10, textAlign: "right" }}>
                                <a href={resolveUrl(current_url)} download className="btn-save">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                                    Download
                                </a>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="empty-state">
                    <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                    </svg>
                    <div className="empty-title">No Image Loaded</div>
                    <div className="empty-sub">Upload an image from the sidebar to get started</div>
                </div>
            )}

            {/* Filter selector (bottom) -- shown when image is loaded */}
            {has_image && (
                <div className="filter-config-bar">
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
                                {filterCards[tab].map(f => (
                                    <div
                                        key={f.id}
                                        className={`fcard ${tab === "lpf" ? "lpf" : "hpf"} ${filterName === f.id ? "sel" : ""}`}
                                        onClick={() => setFilterName(f.id)}
                                    >
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
                            {filterCards.noise.map(f => (
                                <div
                                    key={f.id}
                                    className={`fcard lpf ${filterName === f.id ? "sel" : ""}`}
                                    onClick={() => setFilterName(f.id)}
                                >
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

                    {isRoberts && (
                        <div className="roberts-note show">
                            ⚡ Roberts uses fixed 2×2 kernels — kernel size is ignored
                        </div>
                    )}

                    {!isNoise && !isRoberts && (
                        <div className="kernel-row">
                            <label>Kernel Size</label>
                            <div className="kernel-stepper">
                                <button className="kstep-btn" onClick={() => setKernelSize(Math.max(3, kernelSize - 2))}>−</button>
                                <div id="kernelSizeDisplay">{kernelSize}</div>
                                <button className="kstep-btn" onClick={() => setKernelSize(Math.min(21, kernelSize + 2))}>+</button>
                            </div>
                            <span className="kernel-tag">{kernelSize}×{kernelSize}</span>
                        </div>
                    )}

                    {isCanny && (
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

                    {isNoise && (
                        <div className="canny-controls show">
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
                        disabled={!filterName || loading}
                        onClick={handleApply}
                    >
                        {loading ? (
                            <span className="spinner" />
                        ) : (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
                        )}
                        {loading ? "Processing…" : "Apply Filter"}
                    </button>

                    <div className="act-row">
                        <button className="btn btn-secondary" disabled={!can_undo || loading} onClick={onUndo}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 14 4 9 9 4" /><path d="M20 20v-7a4 4 0 0 0-4-4H4" /></svg>
                            Undo
                        </button>
                        <button className="btn btn-outline-red" disabled={!has_image || loading} onClick={onReset}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.36" /></svg>
                            Reset
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

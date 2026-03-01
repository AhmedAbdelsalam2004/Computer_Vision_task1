import { useRef, useEffect } from "react";

const MEDIA_BASE = "http://localhost:8000";

function resolveUrl(url) {
    if (!url) return null;
    if (url.startsWith("http") || url.startsWith("data:")) return url;
    return url.startsWith("http") ? url : `${MEDIA_BASE}${url}`;
}

function HistogramCanvas({ data, color, label }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!data || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const max = Math.max(...data);
        const barW = W / data.length;

        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, color);
        grad.addColorStop(1, `${color}33`);

        ctx.fillStyle = grad;
        data.forEach((v, i) => {
            const bh = (v / max) * (H - 4);
            ctx.fillRect(i * barW, H - bh, barW, bh);
        });
    }, [data, color]);

    return (
        <div className="hist-canvas-wrap">
            <div className="hist-label">{label}</div>
            <canvas ref={canvasRef} width={512} height={180} className="hc" />
        </div>
    );
}

export default function HistogramPanel({ state, onHistogram, onEqualize, loading }) {
    const { hist_orig_data, hist_eq_data, hist_eq_url, has_image, original_url, current_url } = state;

    return (
        <div className="right-hist">
            <div className="status-bar">
                <div className="status-item">
                    <div className={`status-dot ${has_image ? "green" : ""}`} />
                    <span className="status-label">MODE</span>
                    <span className="status-val">Histogram Analysis</span>
                </div>
            </div>

            {has_image && (
                <div className="img-grid">
                    <div className="ibox">
                        <div className="ibox-head">
                            <span className="ibox-label">Original</span>
                            <span className="badge b-src">SOURCE</span>
                        </div>
                        <div className="ibox-frame">
                            <img src={resolveUrl(original_url || current_url)} alt="Original" />
                        </div>
                    </div>
                    <div className="ibox">
                        <div className="ibox-head">
                            <span className="ibox-label">Equalized</span>
                            <span className="badge b-eq">EQ</span>
                        </div>
                        <div className="ibox-frame">
                            {hist_eq_url ? (
                                <img src={resolveUrl(hist_eq_url)} alt="Equalized" />
                            ) : (
                                <div style={{ textAlign: "center", color: "var(--text3)", fontFamily: "'JetBrains Mono',monospace", fontSize: ".72rem" }}>
                                    [ run equalization ]
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {hist_orig_data && (
                <div className="hist-panel">
                    <div className="hist-panel-header">
                        <span className="hist-panel-title orig">◈ Original Histogram</span>
                    </div>
                    <HistogramCanvas data={hist_orig_data} color="#4fffb0" label="Pixel Intensity Distribution" />
                </div>
            )}

            {hist_eq_data && (
                <div className="hist-panel">
                    <div className="hist-panel-header">
                        <span className="hist-panel-title eq">◈ Equalized Histogram</span>
                    </div>
                    <HistogramCanvas data={hist_eq_data} color="#a78bfa" label="After Equalization" />
                </div>
            )}

            {!has_image && (
                <div className="empty-state">
                    <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                    <div className="empty-title">No Histogram Yet</div>
                    <div className="empty-sub">Upload an image and click Draw Histogram or Equalize Contrast</div>
                </div>
            )}
        </div>
    );
}

import { useRef, useEffect, useState } from "react";

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

        // Simple plot area with margins for axes/labels
        const marginLeft = 32;
        const marginRight = 8;
        const marginTop = 8;
        const marginBottom = 24;
        const plotW = W - marginLeft - marginRight;
        const plotH = H - marginTop - marginBottom;

        const max = Math.max(...data) || 1;
        const barW = plotW / data.length;

        // Axes
        ctx.strokeStyle = "#4b5563";
        ctx.lineWidth = 1;
        ctx.beginPath();
        // y-axis
        ctx.moveTo(marginLeft, marginTop);
        ctx.lineTo(marginLeft, marginTop + plotH);
        // x-axis
        ctx.lineTo(marginLeft + plotW, marginTop + plotH);
        ctx.stroke();

        // Axis labels and ticks (x: 0, 128, 255; y: 0, max)
        ctx.fillStyle = "var(--text3)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        [0, 128, 255].forEach((v, idx) => {
            const x = marginLeft + (v / 255) * plotW;
            ctx.fillText(String(v), x, marginTop + plotH + 4);
        });
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("0", marginLeft - 4, marginTop + plotH);
        ctx.fillText(String(max), marginLeft - 4, marginTop);

        // Bars
        const grad = ctx.createLinearGradient(0, marginTop, 0, marginTop + plotH);
        grad.addColorStop(0, color);
        grad.addColorStop(1, `${color}33`);

        ctx.fillStyle = grad;
        data.forEach((v, i) => {
            const norm = v / max;
            const bh = norm * plotH;
            const x = marginLeft + i * barW;
            const y = marginTop + plotH - bh;
            ctx.fillRect(x, y, barW, bh);
        });
    }, [data, color]);

    return (
        <div className="hist-canvas-wrap">
            <div className="hist-label">{label}</div>
            <canvas ref={canvasRef} width={512} height={180} className="hc" />
        </div>
    );
}

function CdfCanvas({ data, color, label }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!data || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const marginLeft = 32;
        const marginRight = 8;
        const marginTop = 8;
        const marginBottom = 24;
        const plotW = W - marginLeft - marginRight;
        const plotH = H - marginTop - marginBottom;

        const max = data[data.length - 1] || 1;

        // Axes
        ctx.strokeStyle = "#4b5563";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(marginLeft, marginTop);
        ctx.lineTo(marginLeft, marginTop + plotH);
        ctx.lineTo(marginLeft + plotW, marginTop + plotH);
        ctx.stroke();

        // Axis labels (x: 0,128,255; y: 0,1.0)
        ctx.fillStyle = "var(--text3)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        [0, 128, 255].forEach(v => {
            const x = marginLeft + (v / 255) * plotW;
            ctx.fillText(String(v), x, marginTop + plotH + 4);
        });
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("0.0", marginLeft - 4, marginTop + plotH);
        ctx.fillText("1.0", marginLeft - 4, marginTop);

        // CDF curve (normalized 0–1)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const lastVal = max || 1;
        data.forEach((v, i) => {
            const norm = v / lastVal;
            const x = marginLeft + (i / (data.length - 1)) * plotW;
            const y = marginTop + plotH - norm * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }, [data, color]);

    return (
        <div className="hist-canvas-wrap">
            <div className="hist-label">{label}</div>
            <canvas ref={canvasRef} width={512} height={180} className="hc" />
        </div>
    );
}

export default function HistogramPanel({ state }) {
    const {
        hist_orig_data, hist_orig_cdf,
        hist_eq_data, hist_eq_cdf, hist_eq_url,
        hist_rgb_data, hist_rgb_cdf,
        has_image, original_url, current_url,
    } = state;

    const [showRGB, setShowRGB] = useState(true);

    return (
        <div className="right-hist">
            <div className="status-bar">
                <div className="status-item">
                    <div className={`status-dot ${has_image ? "green" : ""}`} />
                    <span className="status-label">MODE</span>
                    <span className="status-val">Histogram Analysis</span>
                </div>
                <div style={{ marginLeft: "auto" }}>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: "4px 10px", fontSize: "0.7rem" }}
                        onClick={() => setShowRGB(v => !v)}
                        disabled={!hist_rgb_data && !hist_rgb_cdf}
                    >
                        {showRGB ? "Hide RGB Histograms" : "Show RGB Histograms"}
                    </button>
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

            {(hist_orig_data || hist_orig_cdf || hist_eq_data || hist_eq_cdf || hist_rgb_data || hist_rgb_cdf) && (
                <div
                    style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                        gap: 16,
                        marginTop: 16,
                    }}
                >
                    {hist_orig_data && (
                        <div className="hist-panel">
                            <div className="hist-panel-header">
                                <span className="hist-panel-title orig">◈ Original Histogram</span>
                            </div>
                            <HistogramCanvas data={hist_orig_data} color="#4fffb0" label="Pixel Intensity Distribution" />
                        </div>
                    )}

                    {hist_orig_cdf && (
                        <div className="hist-panel">
                            <div className="hist-panel-header">
                                <span className="hist-panel-title orig">◈ Original CDF</span>
                            </div>
                            <CdfCanvas data={hist_orig_cdf} color="#4fffb0" label="Cumulative Distribution Function" />
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

                    {hist_eq_cdf && (
                        <div className="hist-panel">
                            <div className="hist-panel-header">
                                <span className="hist-panel-title eq">◈ Equalized CDF</span>
                            </div>
                            <CdfCanvas data={hist_eq_cdf} color="#a78bfa" label="CDF After Equalization" />
                        </div>
                    )}

                    {showRGB && hist_rgb_data && (
                        <>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ R Histogram</span>
                                </div>
                                <HistogramCanvas data={hist_rgb_data.r} color="#ff5a5f" label="R channel" />
                            </div>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ G Histogram</span>
                                </div>
                                <HistogramCanvas data={hist_rgb_data.g} color="#4fffb0" label="G channel" />
                            </div>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ B Histogram</span>
                                </div>
                                <HistogramCanvas data={hist_rgb_data.b} color="#4f8bff" label="B channel" />
                            </div>
                        </>
                    )}

                    {showRGB && hist_rgb_cdf && (
                        <>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ R CDF</span>
                                </div>
                                <CdfCanvas data={hist_rgb_cdf.r} color="#ff5a5f" label="R CDF" />
                            </div>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ G CDF</span>
                                </div>
                                <CdfCanvas data={hist_rgb_cdf.g} color="#4fffb0" label="G CDF" />
                            </div>
                            <div className="hist-panel">
                                <div className="hist-panel-header">
                                    <span className="hist-panel-title orig">◈ B CDF</span>
                                </div>
                                <CdfCanvas data={hist_rgb_cdf.b} color="#4f8bff" label="B CDF" />
                            </div>
                        </>
                    )}
                </div>
            )}

            {has_image && !hist_orig_data && !hist_eq_data && !hist_rgb_data && (
                <div className="empty-state">
                    <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                    <div className="empty-title">No Histogram Yet</div>
                    <div className="empty-sub">Click Draw Histogram (RGB + CDF) or Equalize Contrast</div>
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

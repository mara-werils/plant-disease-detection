"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import {
  Leaf, Search, Beaker, Network, Info, ArrowRight,
  UploadCloud, BrainCircuit, Activity, CheckCircle2, AlertTriangle, Layers,
  Thermometer, Droplets, Mountain
} from "lucide-react";

// ==================== NAVBAR ====================
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <div className="container">
        <a href="#" className="nav-logo">
          <div className="nav-logo-icon">
            <Leaf size={20} strokeWidth={2.5} color="white" />
          </div>
          PlantGuard AI
        </a>
        <ul className={`nav-links ${menuOpen ? "open" : ""}`}>
          <li><a href="#how-it-works" onClick={() => setMenuOpen(false)}>Methodology</a></li>
          <li><a href="#detection" onClick={() => setMenuOpen(false)}>Diagnostics</a></li>
          <li><a href="#research" onClick={() => setMenuOpen(false)}>Research</a></li>
          <li><a href="#methodology" onClick={() => setMenuOpen(false)}>Architecture</a></li>
          <li><a href="#about" onClick={() => setMenuOpen(false)}>About</a></li>
          <li><a href="#detection" className="nav-cta" onClick={() => setMenuOpen(false)}>Start Analysis</a></li>
        </ul>
        <button className="mobile-toggle" onClick={() => setMenuOpen(!menuOpen)}>
          <span /><span /><span />
        </button>
      </div>
    </nav>
  );
}

// ==================== HERO ====================
function HeroSection() {
  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content" style={{ maxWidth: "100%", textAlign: "center", margin: "0 auto" }}>
          <div className="hero-badge" style={{ margin: "0 auto 28px" }}>
            <span className="dot" />
            Scientific Research Project
          </div>
          <h1>
            Precision Agronomy with{" "}
            <span className="gradient-text">Explainable AI</span>
          </h1>
          <p className="hero-description" style={{ maxWidth: 640, margin: "0 auto 40px" }}>
            Advanced diagnostic system powered by MobileNetV2 Transfer Learning with
            Test-Time Augmentation, achieving high accuracy across 38 crop disease categories.
            Integrated with Saliency Map explainability and LLM-driven pathology recommendations.
          </p>
          <div className="hero-buttons" style={{ justifyContent: "center" }}>
            <a href="#detection" className="btn-primary">
              <Search size={18} /> Run Diagnostics
            </a>
            <a href="#research" className="btn-outline">
              <Beaker size={18} /> View Research Metrics
            </a>
          </div>
          <div className="hero-stats" style={{ justifyContent: "center" }}>
            <div className="hero-stat">
              <h3>38</h3>
              <p>Disease Classes</p>
            </div>
            <div className="hero-stat">
              <h3>54K+</h3>
              <p>Training Images</p>
            </div>
            <div className="hero-stat">
              <h3>14</h3>
              <p>Crop Species</p>
            </div>
            <div className="hero-stat">
              <h3>&lt;2s</h3>
              <p>Inference Time</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== HOW IT WORKS ====================
function HowItWorks() {
  return (
    <section className="section how-it-works" id="how-it-works">
      <div className="container">
        <div className="animate-in">
          <div className="section-badge"><Layers size={14} style={{ marginRight: 6 }} /> Workflow</div>
          <h2 className="section-title">Diagnostic Protocol</h2>
          <p className="section-subtitle">
            A streamlined three-stage pipeline for automated agronomic assessment
          </p>
        </div>
        <div className="steps-container animate-in">
          <div className="glass-card step-card">
            <div className="step-number">1</div>
            <div className="step-icon">
              <UploadCloud size={32} color="var(--primary)" />
            </div>
            <h3>Data Ingestion</h3>
            <p>
              Upload cellular or macro leaf imagery. The system automatically
              normalizes and preprocesses the input tensor for the CNN.
            </p>
          </div>
          <div className="step-connector" />
          <div className="glass-card step-card">
            <div className="step-number">2</div>
            <div className="step-icon">
              <BrainCircuit size={32} color="var(--primary)" />
            </div>
            <h3>CNN Inference</h3>
            <p>
              The ResNet50 framework identifies pathological patterns. Grad-CAM
              generates spatial heatmaps to validate the model's focus.
            </p>
          </div>
          <div className="step-connector" />
          <div className="glass-card step-card">
            <div className="step-number">3</div>
            <div className="step-icon">
              <Activity size={32} color="var(--primary)" />
            </div>
            <h3>LLM Synthesis</h3>
            <p>
              Mistral-7B synthesizes the diagnostic output to formulate
              actionable, context-aware agricultural interventions.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== FEATURES ====================
import { Target, Eye, Cpu, Zap, Globe2, Cloud } from "lucide-react";

function FeaturesSection() {
  const features = [
    { icon: <Target size={24} color="var(--primary)" />, title: "High Accuracy Validation", desc: "94% test accuracy achieved on the PlantVillage dataset using a fine-tuned Transfer Learning approach." },
    { icon: <Eye size={24} color="var(--primary)" />, title: "XAI Integration (Grad-CAM)", desc: "Visual activation maps provide interpretability, ensuring diagnoses are based on actual pathological symptoms." },
    { icon: <BrainCircuit size={24} color="var(--primary)" />, title: "Generative AI Agent", desc: "Mistral-7B LLM orchestration generates sophisticated, tailored agricultural treatment protocols natively." },
    { icon: <Zap size={24} color="var(--primary)" />, title: "Real-time Edge Inference", desc: "Optimized model weights enable sub-2 second predictions, feasible for deployment on constrained edge devices." },
    { icon: <Globe2 size={24} color="var(--primary)" />, title: "Comprehensive Coverage", desc: "Capable of classifying 38 distinct pathological states across 14 major agricultural crop species." },
    { icon: <Cloud size={24} color="var(--primary)" />, title: "API-driven Architecture", desc: "Decoupled Next.js client and FastAPI server ensuring high scalability and modular integration." },
  ];

  return (
    <section className="section">
      <div className="container">
        <div className="animate-in" style={{ textAlign: "center" }}>
          <div className="section-badge"><Cpu size={14} style={{ marginRight: 6 }} /> System Capabilities</div>
          <h2 className="section-title">Technological Innovations</h2>
          <p className="section-subtitle" style={{ margin: "0 auto" }}>
            Bridging the gap between empirical deep learning and practical agronomy
          </p>
        </div>
        <div className="grid-3 animate-in" style={{ marginTop: 48 }}>
          {features.map((f, i) => (
            <div className="glass-card feature-card" key={i}>
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}


// ==================== DETECTION DEMO (REAL API) ====================
const API_URL = "http://localhost:8000";

function DetectionDemo() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.4);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Environmental Context Variables for Multi-Modal Input
  const [envTemperature, setEnvTemperature] = useState(25);
  const [envHumidity, setEnvHumidity] = useState(50);
  const [envSoil, setEnvSoil] = useState("Loamy");

  const sampleImages = [
    { id: "healthy", src: "/images/healthy-leaf.png", label: "Healthy" },
    { id: "early-blight", src: "/images/diseased-leaf.png", label: "Early Blight" },
  ];

  const handleAnalyze = useCallback(async (file) => {
    setAnalyzing(true);
    setResults(null);
    setRecommendations(null);
    setError(null);
    setPipelineStep(0);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("temperature", envTemperature.toString());
    formData.append("humidity", envHumidity.toString());
    formData.append("soil_type", envSoil);

    try {
      // Animate pipeline steps
      setPipelineStep(1); // Preprocessing
      await new Promise(r => setTimeout(r, 400));
      setPipelineStep(2); // CNN Inference

      const resp = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) throw new Error("Prediction failed");

      setPipelineStep(3); // Grad-CAM
      await new Promise(r => setTimeout(r, 300));

      const data = await resp.json();
      setPipelineStep(4); // Complete
      setResults(data);
      setAnalyzing(false);

      // Fetch LLM recommendations
      setLoadingRecs(true);
      try {
        const recResp = await fetch(`${API_URL}/recommend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prediction: data.prediction.description,
            isHealthy: data.prediction.isHealthy,
            context: {
              temperature: envTemperature,
              humidity: envHumidity,
              soilType: envSoil
            }
          }),
        });
        const recData = await recResp.json();
        setRecommendations(recData);
      } catch (e) {
        console.error("Recommendations error:", e);
      }
      setLoadingRecs(false);
    } catch (e) {
      setError(e.message);
      setAnalyzing(false);
    }
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedImage(url);
      setSelectedFile(file);
      await handleAnalyze(file);
    }
  };

  const handleSampleClick = async (img) => {
    setSelectedImage(img.src);
    // Fetch sample image as File object
    const resp = await fetch(img.src);
    const blob = await resp.blob();
    const file = new File([blob], `${img.id}.png`, { type: "image/png" });
    setSelectedFile(file);
    await handleAnalyze(file);
  };

  const pipelineSteps = [
    "Upload", "Preprocess", "ResNet50", "Grad-CAM", "Complete"
  ];

  return (
    <section className="section detection-section" id="detection">
      <div className="container">
        <div className="animate-in" style={{ textAlign: "center" }}>
          <div className="section-badge"><Search size={14} style={{ marginRight: 6 }} /> Live AI Inference</div>
          <h2 className="section-title">Disease Diagnostics</h2>
          <p className="section-subtitle" style={{ margin: "0 auto" }}>
            Upload a leaf specimen — real-time ResNet50 inference integrated with Grad-CAM explainability
          </p>
        </div>

        <div className="detection-grid animate-in">
          {/* Upload Zone */}
          <div>
            <div
              className={`glass-card upload-zone ${selectedImage ? "has-image" : ""}`}
              onClick={() => !selectedImage && fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <>
                  <img src={selectedImage} alt="Selected leaf" className="upload-preview" />
                  <button
                    className="btn-outline"
                    style={{ marginTop: 16, padding: "8px 20px", fontSize: "0.85rem" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedImage(null);
                      setSelectedFile(null);
                      setResults(null);
                      setRecommendations(null);
                      setError(null);
                    }}
                  >
                    Clear & Upload New
                  </button>
                </>
              ) : (
                <>
                  <div className="upload-icon"><UploadCloud size={48} color="var(--primary-light)" /></div>
                  <h3>Drop specimen imagery here</h3>
                  <p>or click to browse • JPG, PNG formats supported</p>
                </>
              )}
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                accept="image/*"
                onChange={handleFileUpload}
              />
            </div>
            <div className="sample-images">
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", alignSelf: "center" }}>
                Try samples:
              </span>
              {sampleImages.map((img) => (
                <img
                  key={img.id}
                  src={img.src}
                  alt={img.label}
                  className={`sample-img ${selectedImage === img.src ? "active" : ""}`}
                  onClick={() => handleSampleClick(img)}
                />
              ))}
            </div>

            {/* ENVIRONMENTAL CONTEXT CONTROLS */}
            <div className="glass-card" style={{ marginTop: 24, padding: 24, animation: "fadeIn 0.5s ease" }}>
              <h4 style={{ marginBottom: 16, fontSize: "0.95rem", display: "flex", alignItems: "center", gap: 8 }}>
                <Layers size={18} color="var(--primary)" /> Environmental Context
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 400, marginLeft: "auto" }}>
                  (Multi-Modal AI Input)
                </span>
              </h4>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 8 }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}><Thermometer size={14} /> Temperature: {envTemperature}°C</span>
                  </label>
                  <input type="range" min="0" max="50" value={envTemperature} onChange={(e) => setEnvTemperature(e.target.value)} style={{ width: "100%", accentColor: "var(--primary)" }} />
                </div>

                <div>
                  <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 8 }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}><Droplets size={14} /> Regional Humidity: {envHumidity}%</span>
                  </label>
                  <input type="range" min="0" max="100" value={envHumidity} onChange={(e) => setEnvHumidity(e.target.value)} style={{ width: "100%", accentColor: "var(--primary)" }} />
                </div>

                <div>
                  <label style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 8 }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}><Mountain size={14} /> Soil Classification</span>
                  </label>
                  <select
                    value={envSoil}
                    onChange={(e) => setEnvSoil(e.target.value)}
                    style={{
                      width: "100%", padding: "10px 12px", borderRadius: 8,
                      border: "1px solid var(--border-glass)", background: "rgba(0,0,0,0.3)",
                      color: "white", fontSize: "0.85rem", outline: "none",
                      cursor: "pointer"
                    }}
                  >
                    <option value="Loamy" style={{ color: "black" }}>Loamy Soil (Balanced)</option>
                    <option value="Clay" style={{ color: "black" }}>Clay Soil (High Moisture)</option>
                    <option value="Sandy" style={{ color: "black" }}>Sandy Soil (Fast Draining)</option>
                    <option value="Peaty" style={{ color: "black" }}>Peaty Soil (Acidic/Organic)</option>
                    <option value="Chalky" style={{ color: "black" }}>Chalky Soil (Alkaline)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="results-panel">
            {analyzing && (
              <div className="glass-card result-card" style={{ textAlign: "center", padding: 48 }}>
                <div className="loading-spinner" />
                <p className="analyzing-text">Running AI inference<span className="dots" /></p>
                <div className="progress-steps" style={{ marginTop: 24 }}>
                  {pipelineSteps.map((step, i) => (
                    <div key={i} className={`progress-step ${i <= pipelineStep ? "active" : ""}`}>
                      <div className="progress-fill" />
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 8 }}>
                  {pipelineSteps[pipelineStep]} → {pipelineSteps[Math.min(pipelineStep + 1, 4)]}
                </p>
              </div>
            )}

            {error && (
              <div className="glass-card result-card" style={{ borderLeft: "3px solid var(--danger)" }}>
                <p style={{ color: "var(--danger)", display: "flex", alignItems: "center", gap: 8 }}>
                  <AlertTriangle size={18} /> {error}
                </p>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
                  Ensure API server is active: <code>cd api && source venv/bin/activate && python server.py</code>
                </p>
              </div>
            )}

            {results && !analyzing && (
              <>
                {/* Diagnosis */}
                <div className="glass-card result-card" style={{ animation: "fadeInUp 0.5s ease" }}>
                  <div className="result-header">
                    <div className={`result-status ${results.prediction.isHealthy ? "healthy" : "disease"}`}>
                      {results.prediction.isHealthy ? <CheckCircle2 size={24} color="white" /> : <AlertTriangle size={24} color="white" />}
                    </div>
                    <div>
                      <h3>{results.prediction.description}</h3>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                        Species: {results.prediction.crop} • Pathology: {results.prediction.disease}
                      </p>
                    </div>
                  </div>
                  <div className="confidence-label">
                    <span>Confidence</span>
                    <span style={{ fontWeight: 700, color: "var(--primary)" }}>
                      {(results.prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${!results.prediction.isHealthy ? "danger" : ""}`}
                      style={{ width: `${results.prediction.confidence * 100}%` }}
                    />
                  </div>
                  {!results.prediction.isHealthy && (
                    <div style={{
                      display: "inline-flex", alignItems: "center", gap: 6,
                      padding: "4px 12px", background: "rgba(245,158,11,0.1)",
                      borderRadius: 100, fontSize: "0.8rem", fontWeight: 600,
                      color: "var(--warning)", marginTop: 8
                    }}>
                      <Info size={14} /> Severity Assessment: {results.prediction.severity}
                    </div>
                  )}
                </div>

                {/* Real Grad-CAM — Side by Side */}
                <div className="glass-card result-card" style={{ animation: "fadeInUp 0.5s ease 0.2s both" }}>
                  <h4 style={{ marginBottom: 12, fontSize: "0.95rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 8 }}><Eye size={18} color="var(--primary)" /> XAI Activation Map</span>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 400 }}>
                      Layer: Deep Convolutional Node
                    </span>
                  </h4>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <div>
                      <img
                        src={`data:image/png;base64,${results.gradcam.original}`}
                        alt="Original"
                        style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border-glass)" }}
                      />
                      <p style={{ textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 6 }}>Original Image</p>
                    </div>
                    <div>
                      <img
                        src={`data:image/png;base64,${results.gradcam.overlay}`}
                        alt="Grad-CAM Overlay"
                        style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border-glass)" }}
                      />
                      <p style={{ textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 6 }}>Grad-CAM Overlay</p>
                    </div>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: 4 }}>
                      <span>Heatmap Only</span>
                    </div>
                    <img
                      src={`data:image/png;base64,${results.gradcam.heatmap}`}
                      alt="Heatmap"
                      style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border-glass)", maxHeight: 120, objectFit: "cover" }}
                    />
                  </div>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 8 }}>
                    Grad-CAM highlights regions the CNN focused on for classification (Deep Convolutional Node layer).
                    Red/warm areas = high activation. Blue/cool = low activation.
                  </p>
                </div>

                {/* Top-5 Predictions */}
                <div className="glass-card result-card" style={{ animation: "fadeInUp 0.5s ease 0.3s both" }}>
                  <h4 style={{ marginBottom: 12, fontSize: "0.95rem", display: "flex", alignItems: "center", gap: 8 }}>
                    <Activity size={18} color="var(--primary)" /> Top-K Classification Probabilities
                  </h4>
                  {results.top5.map((pred, i) => (
                    <div key={i} style={{ marginBottom: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: 3 }}>
                        <span style={{ color: i === 0 ? "var(--primary-light)" : "var(--text-secondary)", fontWeight: i === 0 ? 600 : 400, display: "flex", alignItems: "center", gap: 4 }}>
                          {i === 0 ? <CheckCircle2 size={12} /> : null}
                          {pred.description}
                        </span>
                        <span style={{ fontWeight: 600, color: i === 0 ? "var(--primary)" : "var(--text-muted)" }}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{
                        height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden"
                      }}>
                        <div style={{
                          height: "100%", borderRadius: 3,
                          width: `${pred.confidence * 100}%`,
                          background: i === 0
                            ? "linear-gradient(90deg, var(--primary), var(--primary-light))"
                            : "rgba(255,255,255,0.15)",
                          transition: "width 0.8s ease"
                        }} />
                      </div>
                    </div>
                  ))}
                </div>

                {/* LLM Recommendations */}
                <div style={{ animation: "fadeInUp 0.5s ease 0.4s both" }}>
                  <h4 style={{ marginBottom: 12, fontSize: "0.95rem", display: "flex", alignItems: "center", gap: 8 }}>
                    <BrainCircuit size={18} color="var(--primary)" /> Algorithmic Recommendations
                    <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 400 }}>
                      {recommendations?.source === "mistral-7b" ? "Mistral-7B Inference" : "Deterministic Fallback"}
                    </span>
                  </h4>
                  {loadingRecs ? (
                    <div className="glass-card recommendation-card" style={{ textAlign: "center", padding: 24 }}>
                      <div className="loading-spinner" style={{ width: 24, height: 24, borderWidth: 2 }} />
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 8 }}>
                        Generating recommendations via LLM...
                      </p>
                    </div>
                  ) : recommendations ? (
                    <div className="glass-card recommendation-card" style={{ padding: 20 }}>
                      <div style={{ fontSize: "0.85rem", lineHeight: 1.7, color: "var(--text-secondary)", whiteSpace: "pre-line" }}>
                        {recommendations.recommendations}
                      </div>
                    </div>
                  ) : null}
                </div>
              </>
            )}

            {!analyzing && !results && !error && (
              <div className="glass-card result-card" style={{ textAlign: "center", padding: 60, opacity: 0.6 }}>
                <div style={{ marginBottom: 16, display: "flex", justifyContent: "center" }}><Leaf size={48} color="var(--text-muted)" /></div>
                <h3 style={{ fontSize: "1.1rem", marginBottom: 8 }}>System Standby</h3>
                <p style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
                  Provide optical input to initialize the diagnostic pipeline
                </p>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 12, opacity: 0.7 }}>
                  Computational Graph: ResNet50 → Grad-CAM → Mistral-7B
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}


// ==================== RESEARCH DASHBOARD ====================
function ResearchDashboard() {
  const models = [
    { name: "Baseline CNN", train: 83, val: 77, test: 79 },
    { name: "ResNet50 Frozen", train: 70, val: 89, test: 85 },
    { name: "ResNet50 Fine-Tuned", train: 83, val: 94, test: 94 },
  ];

  const confusionData = [
    [95, 2, 1, 0, 1, 1],
    [1, 93, 2, 2, 1, 1],
    [0, 3, 94, 1, 1, 1],
    [1, 1, 2, 96, 0, 0],
    [2, 1, 0, 1, 92, 4],
    [1, 0, 1, 0, 3, 95],
  ];
  const confLabels = ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy", "Early Blight", "Late Blight"];

  return (
    <section className="section dashboard-section" id="research">
      <div className="container">
        <div className="animate-in" style={{ textAlign: "center" }}>
          <div className="section-badge"><Activity size={14} style={{ marginRight: 6 }} /> Research Data</div>
          <h2 className="section-title">Model Performance Analytics</h2>
          <p className="section-subtitle" style={{ margin: "0 auto" }}>
            Detailed quantitative metrics and comparative architectural analysis
          </p>
        </div>

        {/* Key Metrics */}
        <div className="metrics-grid animate-in" style={{ marginTop: 48 }}>
          {[
            { value: "94%", label: "Test Accuracy" },
            { value: "38", label: "Disease Classes" },
            { value: "54,305", label: "Training Images" },
            { value: "0.94", label: "F1 Score" },
          ].map((m, i) => (
            <div className="glass-card metric-card" key={i}>
              <div className="metric-value">{m.value}</div>
              <div className="metric-label">{m.label}</div>
            </div>
          ))}
        </div>

        {/* Model Comparison */}
        <div className="glass-card chart-container animate-in">
          <div className="chart-header">
            <div className="chart-title">Model Accuracy Comparison</div>
            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "#3B82F6" }} />
                Training
              </div>
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "#10B981" }} />
                Validation
              </div>
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "#F59E0B" }} />
                Test
              </div>
            </div>
          </div>
          <div className="bar-chart">
            {models.map((m, i) => (
              <div className="bar-group" key={i}>
                <div className="bars">
                  <div className="bar" style={{ height: `${m.train * 2.8}px`, background: "linear-gradient(180deg, #3B82F6, #2563EB)" }}>
                    <span className="bar-value">{m.train}%</span>
                  </div>
                  <div className="bar" style={{ height: `${m.val * 2.8}px`, background: "linear-gradient(180deg, #10B981, #059669)" }}>
                    <span className="bar-value">{m.val}%</span>
                  </div>
                  <div className="bar" style={{ height: `${m.test * 2.8}px`, background: "linear-gradient(180deg, #F59E0B, #D97706)" }}>
                    <span className="bar-value">{m.test}%</span>
                  </div>
                </div>
                <div className="bar-label">{m.name}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="grid-2 animate-in">
          <div className="glass-card chart-container">
            <div className="chart-title" style={{ marginBottom: 20 }}>Confusion Matrix (Sample)</div>
            <div className="confusion-grid" style={{ gridTemplateColumns: `repeat(${confusionData.length}, 1fr)` }}>
              {confusionData.flat().map((val, idx) => {
                const intensity = val / 100;
                const bg = val > 90
                  ? `rgba(16, 185, 129, ${intensity * 0.8})`
                  : val > 5
                    ? `rgba(245, 158, 11, ${intensity * 0.6})`
                    : `rgba(255, 255, 255, 0.03)`;
                return (
                  <div key={idx} className="confusion-cell" style={{ background: bg }}>
                    {val > 0 ? val : ""}
                  </div>
                );
              })}
            </div>
            <div className="confusion-labels">
              {confLabels.map((l, i) => (
                <span key={i} style={{ fontSize: "0.6rem" }}>{l.split(" ")[0]}</span>
              ))}
            </div>
          </div>
          <div className="glass-card chart-container">
            <div className="chart-title" style={{ marginBottom: 20 }}>Training Strategy</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-glass)" }}>
                  <th style={{ padding: "10px 8px", textAlign: "left", color: "var(--text-muted)" }}>Phase</th>
                  <th style={{ padding: "10px 8px", textAlign: "left", color: "var(--text-muted)" }}>Layers</th>
                  <th style={{ padding: "10px 8px", textAlign: "left", color: "var(--text-muted)" }}>LR</th>
                  <th style={{ padding: "10px 8px", textAlign: "left", color: "var(--text-muted)" }}>Result</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border-glass)" }}>
                  <td style={{ padding: "12px 8px" }}>
                    <span style={{ color: "var(--primary-light)", fontWeight: 600 }}>Phase 1</span>
                  </td>
                  <td style={{ padding: "12px 8px", color: "var(--text-secondary)" }}>Frozen base</td>
                  <td style={{ padding: "12px 8px", color: "var(--text-secondary)" }}>1e-4</td>
                  <td style={{ padding: "12px 8px" }}>
                    <span style={{ color: "var(--success)", fontWeight: 600 }}>No Overfit</span>
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "12px 8px" }}>
                    <span style={{ color: "var(--primary-light)", fontWeight: 600 }}>Phase 2</span>
                  </td>
                  <td style={{ padding: "12px 8px", color: "var(--text-secondary)" }}>Last 30 unfrozen</td>
                  <td style={{ padding: "12px 8px", color: "var(--text-secondary)" }}>1e-5</td>
                  <td style={{ padding: "12px 8px" }}>
                    <span style={{ color: "var(--success)", fontWeight: 600 }}>+2% gain</span>
                  </td>
                </tr>
              </tbody>
            </table>
            <div style={{ marginTop: 24 }}>
              <div className="chart-title" style={{ marginBottom: 12, fontSize: "0.95rem" }}>Key Observations</div>
              <ul style={{ listStyle: "none" }}>
                {[
                  "Excellent generalization — no significant train/test gap",
                  "Strong class separation confirmed by Grad-CAM analysis",
                  "Real-time performance achieved without GPU",
                  "Transfer Learning from ImageNet boosted accuracy by ~15%",
                ].map((obs, i) => (
                  <li key={i} style={{
                    fontSize: "0.85rem", color: "var(--text-secondary)", padding: "6px 0",
                    display: "flex", alignItems: "flex-start", gap: 8
                  }}>
                    <span style={{ color: "var(--primary)", fontWeight: 600 }}>→</span> {obs}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== METHODOLOGY ====================
function MethodologySection() {
  const archSteps = [
    { title: "Input", desc: "224×224×3 RGB" },
    { title: "ResNet50", desc: "Pretrained ImageNet" },
    { title: "GAP", desc: "Global Avg Pooling" },
    { title: "Dense 256", desc: "ReLU + Dropout(0.4)" },
    { title: "Softmax", desc: "38 Classes Output" },
    { title: "Grad-CAM", desc: "Explainability" },
    { title: "LLM", desc: "Mistral-7B Recs" },
  ];

  return (
    <section className="section methodology-section" id="methodology">
      <div className="container">
        <div className="animate-in" style={{ textAlign: "center" }}>
          <div className="section-badge"><Network size={14} style={{ marginRight: 6 }} /> Architecture</div>
          <h2 className="section-title">System Methodology</h2>
          <p className="section-subtitle" style={{ margin: "0 auto" }}>
            End-to-end deep learning pipeline from data ingestion to edge deployment
          </p>
        </div>

        {/* Architecture Flow */}
        <div className="architecture-flow animate-in">
          {archSteps.map((step, i) => (
            <span key={i} style={{ display: "contents" }}>
              <div className="glass-card arch-block">
                <h4>{step.title}</h4>
                <p>{step.desc}</p>
              </div>
              {i < archSteps.length - 1 && <span className="arch-arrow">→</span>}
            </span>
          ))}
        </div>

        {/* Inference Flow */}
        <div className="glass-card inference-flow animate-in" style={{ marginTop: 32 }}>
          <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginRight: 12, fontWeight: 600 }}>
            Inference Pipeline:
          </span>
          {["Leaf Image", "CNN Prediction", "Disease Class", "LLM Prompt", "API Call", "Recommendations", "User Display"].map((s, i) => (
            <span key={i} style={{ display: "contents" }}>
              <div className="flow-step">{s}</div>
              {i < 6 && <span className="flow-arrow">→</span>}
            </span>
          ))}
        </div>

        {/* Pipeline Phases */}
        <div className="pipeline-section animate-in">
          <h3 style={{ fontSize: "1.4rem", textAlign: "center", marginBottom: 8 }}>Development Pipeline</h3>
          <div className="pipeline-grid">
            <div className="glass-card pipeline-card">
              <div className="phase-tag">Phase 1 — Preprocessing</div>
              <h4>Data Preparation</h4>
              <p>Raw leaf images transformed into model-ready tensors</p>
              <ul>
                <li>Resize to 224×224 pixels</li>
                <li>Pixel normalization [0,1]</li>
                <li>70/15/15 stratified split</li>
                <li>Augmentation (flip, rotate, zoom)</li>
              </ul>
            </div>
            <div className="glass-card pipeline-card">
              <div className="phase-tag">Phase 2 — Training</div>
              <h4>Model Development</h4>
              <p>Progressive training with transfer learning</p>
              <ul>
                <li>ResNet50 pretrained backbone</li>
                <li>Frozen base → fine-tuning</li>
                <li>Adam optimizer + scheduling</li>
                <li>EarlyStopping + ReduceLR</li>
              </ul>
            </div>
            <div className="glass-card pipeline-card">
              <div className="phase-tag">Phase 3 — Deployment</div>
              <h4>Production Pipeline</h4>
              <p>Real-time inference with explainability</p>
              <ul>
                <li>Streamlit web application</li>
                <li>Grad-CAM visualization</li>
                <li>HuggingFace LLM API</li>
                <li>Cloud deployment ready</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Dataset Stats */}
        <div className="dataset-stats animate-in" style={{ marginTop: 60 }}>
          {[
            { val: "54,305", label: "Total Images" },
            { val: "38", label: "Disease Classes" },
            { val: "14", label: "Crop Species" },
            { val: "224×224", label: "Image Resolution" },
          ].map((s, i) => (
            <div className="glass-card ds-stat" key={i}>
              <h3>{s.val}</h3>
              <p>{s.label}</p>
            </div>
          ))}
        </div>

        {/* Crops Supported */}
        <div className="animate-in" style={{ marginTop: 60 }}>
          <h3 style={{ fontSize: "1.4rem", textAlign: "center", marginBottom: 8 }}>Supported Crops</h3>
          <p style={{ textAlign: "center", color: "var(--text-muted)", marginBottom: 32 }}>
            Multi-crop disease classification across major agricultural species
          </p>
          <div className="crops-grid">
            {[
              { emoji: "🍎", name: "Apple", diseases: 4 },
              { emoji: "🫐", name: "Blueberry", diseases: 1 },
              { emoji: "🍒", name: "Cherry", diseases: 2 },
              { emoji: "🌽", name: "Corn", diseases: 4 },
              { emoji: "🍇", name: "Grape", diseases: 4 },
              { emoji: "🍊", name: "Orange", diseases: 1 },
              { emoji: "🍑", name: "Peach", diseases: 2 },
              { emoji: "🫑", name: "Pepper", diseases: 2 },
              { emoji: "🥔", name: "Potato", diseases: 3 },
              { emoji: "🫘", name: "Soybean", diseases: 1 },
              { emoji: "🍓", name: "Strawberry", diseases: 2 },
              { emoji: "🍅", name: "Tomato", diseases: 10 },
              { emoji: "🌱", name: "Raspberry", diseases: 1 },
              { emoji: "🎃", name: "Squash", diseases: 1 },
            ].map((crop, i) => (
              <div className="glass-card crop-card" key={i}>
                <div className="crop-emoji">{crop.emoji}</div>
                <h4>{crop.name}</h4>
                <p>{crop.diseases} class{crop.diseases > 1 ? "es" : ""}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== ABOUT & DATASET ====================
function AboutSection() {
  return (
    <section className="section" id="about" style={{ background: "var(--bg-dark)" }}>
      <div className="container">
        <div className="animate-in" style={{ textAlign: "center" }}>
          <div className="section-badge"><Info size={14} style={{ marginRight: 6 }} /> Dataset</div>
          <h2 className="section-title">PlantVillage Corpus</h2>
          <p className="section-subtitle" style={{ margin: "0 auto" }}>
            Open access repository of over 50,000 expertly curated plant images
          </p>
        </div>

        <div className="about-grid animate-in">
          <div className="about-content">
            <h3>Why Plant Disease Detection Matters</h3>
            <p>
              Plant diseases cause serious yield losses worldwide. Traditional disease
              identification is time-consuming, requires expert presence, is error-prone,
              and critically — not scalable for large farms.
            </p>
            <p>
              This project builds an AI system that detects plant diseases using leaf images —
              fast, accurate, and scalable. By leveraging deep learning and transfer learning,
              we achieve expert-level accuracy accessible to any farmer with a smartphone.
            </p>
            <ul className="about-highlights">
              <li><span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 24, paddingRight: 8 }}><Leaf size={16} color="var(--primary)" /></span> 38 unique crop-disease pairs classification</li>
              <li><span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 24, paddingRight: 8 }}><BrainCircuit size={16} color="var(--primary)" /></span> ResNet50 Transfer Learning backbone</li>
              <li><span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 24, paddingRight: 8 }}><Eye size={16} color="var(--primary)" /></span> Grad-CAM visual interpretability layer</li>
              <li><span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 24, paddingRight: 8 }}><Activity size={16} color="var(--primary)" /></span> LLM-powered recommendations via Mistral-7B</li>
            </ul>
          </div>
          <div className="about-image-grid">
            <div className="glass-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
              <Leaf size={32} color="var(--primary)" style={{ marginBottom: 8 }} />
              <h4 style={{ fontSize: "0.95rem", textAlign: "center" }}>38 Disease Classes</h4>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", textAlign: "center", marginTop: 4 }}>
                Comprehensive coverage
              </p>
            </div>
            <div className="glass-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
              <BrainCircuit size={32} color="var(--primary)" style={{ marginBottom: 8 }} />
              <h4 style={{ fontSize: "0.95rem", textAlign: "center" }}>Multi-Modal AI</h4>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", textAlign: "center", marginTop: 4 }}>
                Vision + Environment + LLM
              </p>
            </div>
            <div className="glass-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
              <div style={{ fontSize: "2rem", marginBottom: 8 }}>🏆</div>
              <h4 style={{ fontSize: "0.95rem", textAlign: "center" }}>Research Competition Entry</h4>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", textAlign: "center", marginTop: 4 }}>
                Scientific Research Project
              </p>
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="animate-in" style={{ marginTop: 60, textAlign: "center" }}>
          <h3 style={{ fontSize: "1.3rem", marginBottom: 24 }}>Technology Stack</h3>
          <div className="tech-badges" style={{ justifyContent: "center" }}>
            {[
              "🐍 Python 3.8+",
              "🧠 TensorFlow 2.x",
              "⚡ Keras",
              "🔬 ResNet50",
              "🤖 Mistral-7B",
              "🤗 HuggingFace API",
              "📊 Matplotlib",
              "📐 NumPy / Pandas",
              "👁️ OpenCV",
              "🌐 Streamlit",
              "⚛️ Next.js",
              "📱 React Native",
            ].map((tech, i) => (
              <div className="tech-badge" key={i}>{tech}</div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== CTA ====================
function CTASection() {
  return (
    <section className="cta-section">
      <div className="container">
        <div className="cta-content animate-in">
          <h2>Deploy Diagnostics the Field</h2>
          <p>
            Experience the combination of high-accuracy convolutional networks and
            generative AI protocols via our real-time API.
          </p>
          <div className="cta-buttons" style={{ marginTop: 32, display: "flex", gap: 16, justifyContent: "center" }}>
            <a href="#detection" className="btn-primary" style={{ background: "white", color: "var(--bg-deep)" }}>
              <Search size={18} /> Initialize Analysis
            </a>
          </div>
          <div className="cta-tech-stack" style={{ marginTop: 40, display: "flex", gap: 24, justifyContent: "center", flexWrap: "wrap", opacity: 0.8, fontSize: "0.9rem" }}>
            {[
              "ResNet50", "Grad-CAM",
              "Mistral-7B", "FastAPI", "React"
            ].map(t => (
              <span key={t} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <CheckCircle2 size={14} /> {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// ==================== FOOTER ====================
function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div className="footer-brand">
            <h3>🌿 PlantGuard AI</h3>
            <p>
              AI-powered plant disease detection using deep learning and transfer learning.
              A scientific research project for automated agricultural diagnostics.
            </p>
          </div>
          <div className="footer-col">
            <h4>Navigation</h4>
            <ul>
              <li><a href="#how-it-works">How It Works</a></li>
              <li><a href="#detection">Detection Demo</a></li>
              <li><a href="#research">Research Dashboard</a></li>
              <li><a href="#methodology">Methodology</a></li>
            </ul>
          </div>
          <div className="footer-col">
            <h4>Technology</h4>
            <ul>
              <li><a href="#">ResNet50</a></li>
              <li><a href="#">Grad-CAM</a></li>
              <li><a href="#">Mistral-7B LLM</a></li>
              <li><a href="#">PlantVillage Dataset</a></li>
            </ul>
          </div>
          <div className="footer-col">
            <h4>Links</h4>
            <ul>
              <li><a href="https://github.com/Ishaaq09/Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li><a href="https://automated-plant-disease-detection.streamlit.app/" target="_blank" rel="noopener noreferrer">Live App</a></li>
              <li><a href="#about">About</a></li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© 2026 PlantGuard AI — Scientific Research Project</span>
          <div className="footer-socials">
            <a href="https://github.com/Ishaaq09" target="_blank" rel="noopener noreferrer" title="GitHub">
              🔗
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

// ==================== SCROLL ANIMATION OBSERVER ====================
function useScrollAnimations() {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll(".animate-in").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
}

// ==================== MAIN PAGE ====================
export default function Home() {
  useScrollAnimations();

  return (
    <>
      <Navbar />
      <HeroSection />
      <HowItWorks />
      <FeaturesSection />
      <DetectionDemo />
      <ResearchDashboard />
      <MethodologySection />
      <AboutSection />
      <CTASection />
      <Footer />
    </>
  );
}

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  FileScan,
  HeartPulse,
  Menu,
  Microscope,
  ScanLine,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";

const acceptedExtensions = [".npy"];
const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const apiUrl = (path) => `${apiBaseUrl}${path}`;

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [animatedPercent, setAnimatedPercent] = useState(0);
  const [modelReady, setModelReady] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch(apiUrl("/api/health"))
      .then((response) => response.json())
      .then((data) => setModelReady(Boolean(data.model_ready)))
      .catch(() => setModelReady(false));
  }, []);

  useEffect(() => {
    if (!result) {
      setAnimatedPercent(0);
      return undefined;
    }

    const target = result.displayProbability * 100;
    const duration = 1100;
    let animationFrame;
    let startTime;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      setAnimatedPercent(target * easedProgress);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [result]);

  const chooseFile = (file) => {
    if (!file) return;
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!acceptedExtensions.includes(extension)) {
      setSelectedFile(null);
      setStatus("error");
      setMessage("Please select a NumPy MRI volume (.npy).");
      return;
    }
    setSelectedFile(file);
    setStatus("idle");
    setMessage("");
    setResult(null);
  };

  const analyzeScan = async () => {
    if (!selectedFile) {
      fileInputRef.current?.click();
      return;
    }

    setStatus("loading");
    setMessage("");
    setResult(null);
    const formData = new FormData();
    formData.append("scan", selectedFile);

    try {
      const response = await fetch(apiUrl("/api/predict"), {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "The scan could not be analyzed.");
      }

      const abnormalProbability = data.probability;
      const isAbnormal = data.prediction === "Abnormal";
      setResult({
        prediction: data.prediction,
        abnormalProbability,
        normalProbability: 1 - abnormalProbability,
        displayProbability: isAbnormal
          ? abnormalProbability
          : 1 - abnormalProbability,
        threshold: data.threshold,
      });
      setStatus("success");
      setMessage(
        `${data.prediction} - ${(data.probability * 100).toFixed(1)}% abnormal-class probability`,
      );
    } catch (error) {
      setStatus("error");
      setMessage(error.message);
    }
  };

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    setMenuOpen(false);
  };

  return (
    <div className="app-shell">
      <div className="research-banner">
        <Microscope size={15} />
        <span>Research prototype. Not intended for clinical diagnosis.</span>
      </div>

      <header className="site-header">
        <a className="brand" href="#top" aria-label="KneeSight AI home">
          <span className="brand-mark"><ScanLine size={23} /></span>
          <span>KneeSight<span>AI</span></span>
        </a>

        <nav className={menuOpen ? "nav-links open" : "nav-links"}>
          <button onClick={() => scrollTo("how-it-works")}>How it works</button>
          <button onClick={() => scrollTo("technology")}>Technology</button>
          <button onClick={() => scrollTo("research")}>Research</button>
          <button className="nav-cta" onClick={() => scrollTo("analyze")}>
            Analyze a scan <ArrowRight size={15} />
          </button>
        </nav>

        <button
          className="menu-button"
          onClick={() => setMenuOpen((value) => !value)}
          aria-label="Toggle navigation"
        >
          {menuOpen ? <X /> : <Menu />}
        </button>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-image" aria-hidden="true" />
          <div className="hero-shade" aria-hidden="true" />
          <div className="hero-content">
            <div className="eyebrow">
              <Sparkles size={15} />
              AI-assisted knee MRI research
            </div>
            <h1>See knee MRI screening through a clearer lens.</h1>
            <p>
              A research platform using pretrained 3D deep learning to identify
              patterns associated with normal and abnormal knee MRI volumes.
            </p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => scrollTo("analyze")}>
                Analyze MRI scan <ArrowRight size={18} />
              </button>
              <button className="text-button" onClick={() => scrollTo("technology")}>
                Explore the model
              </button>
            </div>
            <div className="hero-proof">
              <div><strong>82.51%</strong><span>Independent test AUC</span></div>
              <div><strong>3D</strong><span>Volumetric analysis</span></div>
              <div><strong>Top-50</strong><span>Selected MRI slices</span></div>
            </div>
          </div>
        </section>

        <section className="trust-strip" aria-label="Platform qualities">
          <div><ShieldCheck /><span><strong>Privacy-minded</strong>Local research workflow</span></div>
          <div><BrainCircuit /><span><strong>Transfer learning</strong>Pretrained MedicalNet</span></div>
          <div><Clock3 /><span><strong>Efficient review</strong>One volume at a time</span></div>
          <div><FileScan /><span><strong>Transparent output</strong>Probability-based result</span></div>
        </section>

        <section className="analysis-section" id="analyze">
          <div className="section-heading">
            <span className="kicker">MRI analysis workspace</span>
            <h2>Upload a knee MRI volume</h2>
            <p>
              The current research pipeline accepts a preprocessed 3D NumPy
              volume. Clinical DICOM and NIfTI ingestion will be added next.
            </p>
          </div>

          <div className="analysis-grid">
            <div className="upload-card">
              <div className="card-topline">
                <span className="step-number">01</span>
                <span className={`system-pill ${modelReady ? "ready" : ""}`}>
                  <span />
                  {modelReady === null
                    ? "Checking model"
                    : modelReady
                      ? "Model ready"
                      : "Checkpoint required"}
                </span>
              </div>

              <div
                className={isDragging ? "drop-zone dragging" : "drop-zone"}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  chooseFile(event.dataTransfer.files[0]);
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".npy"
                  onChange={(event) => chooseFile(event.target.files[0])}
                  hidden
                />
                <span className="upload-icon"><UploadCloud size={27} /></span>
                {selectedFile ? (
                  <>
                    <strong>{selectedFile.name}</strong>
                    <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB selected</span>
                  </>
                ) : (
                  <>
                    <strong>Drop your MRI volume here</strong>
                    <span>or click to browse your device</span>
                  </>
                )}
                <small>Supported now: .NPY | Expected shape: 128 x 128 x 64</small>
              </div>

              {message && (
                <div className={`status-message ${status}`}>
                  {status === "success" ? <Check size={17} /> : <CircleAlert size={17} />}
                  <span>{message}</span>
                </div>
              )}

              <button
                className="analyze-button"
                onClick={analyzeScan}
                disabled={status === "loading"}
              >
                {status === "loading" ? (
                  <>Analyzing volume <span className="loader" /></>
                ) : (
                  <>Run AI analysis <ScanLine size={18} /></>
                )}
              </button>
              <p className="privacy-note">
                <ShieldCheck size={14} /> Files are processed for this session and are not
                intended to be retained.
              </p>
            </div>

            <div className={`result-preview ${result ? result.prediction.toLowerCase() : ""}`}>
              <div className="preview-header">
                <div>
                  <span className="step-number pale">02</span>
                  <span>{result ? "Analysis result" : "What you will receive"}</span>
                </div>
                <span className={`preview-label ${result ? "live" : ""}`}>
                  {result ? "Live result" : "Sample layout"}
                </span>
              </div>
              {result ? (
                <div
                  className="probability-gauge"
                  style={{ "--gauge-progress": `${animatedPercent * 3.6}deg` }}
                  role="img"
                  aria-label={`${result.prediction} probability ${result.displayProbability * 100}%`}
                >
                  <div className="gauge-inner">
                    <strong>{animatedPercent.toFixed(1)}%</strong>
                    <span>{result.prediction}</span>
                  </div>
                </div>
              ) : (
                <div className={`scan-visual ${status === "loading" ? "analyzing" : ""}`}>
                  <div className="scan-ring ring-one" />
                  <div className="scan-ring ring-two" />
                  {status === "loading" ? (
                    <span className="scan-percentage">0%</span>
                  ) : (
                    <ScanLine size={52} strokeWidth={1.2} />
                  )}
                </div>
              )}
              <div className="preview-result">
                <span>Screening output</span>
                <strong>
                  {result
                    ? `${result.prediction} pattern predicted`
                    : status === "loading"
                      ? "Analyzing selected MRI slices..."
                      : "Normal / Abnormal probability"}
                </strong>
              </div>
              <div className="confidence-row">
                <div className="confidence-heading">
                  <span>{result ? `${result.prediction} probability` : "Model probability visualization"}</span>
                  {result && <strong>{animatedPercent.toFixed(1)}%</strong>}
                </div>
                <div className="confidence-track">
                  <i style={{ width: result ? `${animatedPercent}%` : "0%" }} />
                </div>
              </div>
              {result && (
                <div className="probability-breakdown">
                  <div>
                    <span>Normal</span>
                    <strong>{(result.normalProbability * 100).toFixed(1)}%</strong>
                  </div>
                  <div>
                    <span>Abnormal</span>
                    <strong>{(result.abnormalProbability * 100).toFixed(1)}%</strong>
                  </div>
                </div>
              )}
              <div className="review-note">
                <HeartPulse size={20} />
                <p>
                  <strong>Clinical review remains essential.</strong>
                  The output supports research evaluation and does not replace a radiologist.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="process-section" id="how-it-works">
          <div className="section-heading light">
            <span className="kicker">Designed for clarity</span>
            <h2>From volume to research insight</h2>
          </div>
          <div className="process-grid">
            <ProcessCard
              number="01"
              icon={<UploadCloud />}
              title="Upload volume"
              text="Provide a de-identified, preprocessed 3D knee MRI array."
            />
            <ProcessCard
              number="02"
              icon={<Activity />}
              title="Select key slices"
              text="The pipeline ranks slices by intensity variation and keeps the top 50."
            />
            <ProcessCard
              number="03"
              icon={<BrainCircuit />}
              title="Analyze in 3D"
              text="MedicalNet ResNet-18 evaluates spatial information across the volume."
            />
            <ProcessCard
              number="04"
              icon={<FileScan />}
              title="Review output"
              text="Receive the predicted class, probability, and model metadata."
            />
          </div>
        </section>

        <section className="technology-section" id="technology">
          <div className="technology-copy">
            <span className="kicker">Built on validated research practice</span>
            <h2>Purpose-built for volumetric medical imaging.</h2>
            <p>
              Unlike a conventional 2D image classifier, the model learns from
              depth, structure, and continuity across selected knee MRI slices.
            </p>
            <ul>
              <li><Check /> Pretrained MedicalNet 3D ResNet-18 backbone</li>
              <li><Check /> Volume-level stratified train, validation, and test split</li>
              <li><Check /> Consistent spatial augmentation across each volume</li>
              <li><Check /> Independent test evaluation with ROC-AUC</li>
            </ul>
          </div>
          <div className="model-card">
            <div className="model-card-head">
              <div>
                <small>Current research model</small>
                <strong>MedicalNet ResNet-18</strong>
              </div>
              <span>v1.0</span>
            </div>
            <div className="metric-feature">
              <div className="metric-circle"><strong>0.825</strong><span>Test AUC</span></div>
              <p>
                Evaluated on an independent test set of 447 knee MRI volumes.
              </p>
            </div>
            <div className="model-stats">
              <div><span>Input</span><strong>50 x 128 x 128</strong></div>
              <div><span>Task</span><strong>Binary screening</strong></div>
              <div><span>Framework</span><strong>PyTorch</strong></div>
            </div>
          </div>
        </section>

        <section className="research-section" id="research">
          <div>
            <span className="kicker">Research provenance</span>
            <h2>Developed through academic research at NIT Warangal.</h2>
          </div>
          <p>
            This prototype extends a research internship project on knee MRI
            abnormality classification using the OAI 3D DESS dataset, under the
            guidance of Prof. V Rama, Department of ECE.
          </p>
          <a href="#analyze">Open analysis workspace <ArrowRight size={17} /></a>
        </section>

        <section className="faq-section">
          <div className="section-heading">
            <span className="kicker">Important questions</span>
            <h2>Know what the platform can and cannot do</h2>
          </div>
          <div className="faq-list">
            <Faq
              question="Does this platform provide a medical diagnosis?"
              answer="No. It is a research prototype that produces a model prediction for evaluation. A qualified clinician must interpret medical imaging and decide any next steps."
            />
            <Faq
              question="What scan format can I upload?"
              answer="The first version accepts de-identified NumPy arrays with the same 128 x 128 x 64 structure used during model development. DICOM and NIfTI support are planned."
            />
            <Faq
              question="How was the model evaluated?"
              answer="The selected MedicalNet model reached 0.8251 ROC-AUC on an independent test set after volume-level stratified splitting."
            />
          </div>
        </section>
      </main>

      <footer>
        <div className="footer-brand">
          <a className="brand inverse" href="#top">
            <span className="brand-mark"><ScanLine size={23} /></span>
            <span>KneeSight<span>AI</span></span>
          </a>
          <p>AI-assisted knee MRI screening for research and education.</p>
        </div>
        <div className="footer-note">
          <CircleAlert size={17} />
          <span>Research use only. Not a medical device or diagnostic service.</span>
        </div>
        <small>Copyright 2026 KneeSight AI Research Project</small>
      </footer>
    </div>
  );
}

function ProcessCard({ number, icon, title, text }) {
  return (
    <article className="process-card">
      <span>{number}</span>
      <div>{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

function Faq({ question, answer }) {
  const [open, setOpen] = useState(false);
  return (
    <article className={open ? "faq-item open" : "faq-item"}>
      <button onClick={() => setOpen((value) => !value)}>
        <span>{question}</span>
        <ChevronDown />
      </button>
      {open && <p>{answer}</p>}
    </article>
  );
}

export default App;

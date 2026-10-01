/* SymptomLens frontend. Plain JavaScript, no external libraries.
   All server data is inserted with textContent / createTextNode (never innerHTML). */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";

  function h(tag, props = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  function bar(fraction, cls = "") {
    const fill = h("span");
    fill.style.width = Math.max(0, Math.min(1, fraction)) * 100 + "%";
    return h("div", { class: "bar " + cls, "aria-hidden": "true" }, fill);
  }

  function sevBadge(level) {
    if (level == null) return null;
    const band = level >= 5 ? "high" : level >= 3 ? "mid" : "low";
    return h("span", { class: "sev", "data-level": band, title: `Severity weight ${level} of 7` }, level);
  }

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch {
      throw new Error("Could not reach the server. Check that the application is running.");
    }
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON body */ }
    if (!res.ok) {
      const err = new Error((body && body.error) || `Request failed (${res.status}).`);
      err.field = body && body.field;
      throw err;
    }
    if (body == null) throw new Error("The server returned an empty response.");
    return body;
  }

  /* ------------------------------------------------------------ routing */
  function route() {
    const view = location.hash === "#performance" ? "performance" : "predict";
    document.querySelectorAll(".view").forEach((s) => { s.hidden = s.dataset.view !== view; });
    document.querySelectorAll("[data-nav]").forEach((a) => {
      if (a.dataset.nav === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (view === "performance") loadPerformance();
    window.scrollTo(0, 0);
  }

  /* ------------------------------------------------------ symptom picker */
  const selected = new Map();      // name -> {name,label,severity}
  const boxes = new Map();         // name -> checkbox
  const rows = new Map();          // name -> li
  let catalog = [];

  async function loadSymptoms() {
    const list = $("#sym-list");
    try {
      const data = await api("/api/symptoms");
      catalog = data.symptoms;
      list.replaceChildren();
      for (const s of catalog) {
        const box = h("input", { type: "checkbox", value: s.name, id: "sym-" + s.name });
        box.addEventListener("change", () => toggle(s, box.checked));
        const li = h("li", {}, h("label", { for: "sym-" + s.name }, box, h("span", { text: s.label }), sevBadge(s.severity)));
        boxes.set(s.name, box); rows.set(s.name, li); list.append(li);
      }
      filterSymptoms();
    } catch (e) {
      list.replaceChildren(h("li", { class: "empty", role: "alert", text: e.message }));
    }
  }

  function toggle(sym, on) {
    if (on) selected.set(sym.name, sym); else selected.delete(sym.name);
    boxes.get(sym.name).checked = on;
    renderChips();
    if (selected.size) $("#symptoms-err").textContent = "";
  }

  function renderChips() {
    const box = $("#chips");
    box.replaceChildren();
    if (!selected.size) box.append(h("p", { class: "empty", text: "No symptoms selected yet. Search below and tick the ones that apply." }));
    for (const s of selected.values()) {
      box.append(h("span", { class: "chip" }, s.label, sevBadge(s.severity),
        h("button", { type: "button", "aria-label": `Remove ${s.label}`, onclick: () => toggle(s, false) }, "\u00d7")));
    }
    $("#sym-count").textContent = `${selected.size} selected`;
  }

  function filterSymptoms() {
    const q = $("#sym-search").value.trim().toLowerCase();
    let shown = 0;
    for (const s of catalog) {
      const match = !q || s.label.toLowerCase().includes(q);
      rows.get(s.name).hidden = !match;
      if (match) shown++;
    }
    $("#sym-matches").textContent = shown ? `${shown} of ${catalog.length} symptoms` : "No symptoms match your search";
  }

  /* -------------------------------------------------------------- form */
  const fieldIds = ["age", "gender"];
  function setError(id, msg) {
    const el = $("#" + id + "-err");
    if (el) el.textContent = msg || "";
    const input = $("#" + id);
    if (input) { if (msg) input.setAttribute("aria-invalid", "true"); else input.removeAttribute("aria-invalid"); }
  }
  function clearErrors() {
    fieldIds.forEach((id) => setError(id, ""));
    $("#symptoms-err").textContent = "";
    const alert = $("#form-alert"); alert.hidden = true; alert.textContent = "";
  }

  function readForm() {
    const val = (id) => $("#" + id).value.trim();
    const num = (id) => (val(id) === "" ? null : Number(val(id)));
    const extra = {
      height: num("height"), weight: num("weight"), blood_group: val("blood_group"),
      smoking_status: val("smoking_status"), alcohol_consumption: val("alcohol_consumption"),
      existing_conditions: val("existing_conditions"), allergies: val("allergies"),
      medications: val("medications"), family_history: val("family_history"),
    };
    for (const k of Object.keys(extra)) if (extra[k] === "" || extra[k] === null) delete extra[k];
    return { age: num("age"), gender: val("gender"), symptoms: [...selected.keys()], additional_info: extra };
  }

  function validateClient(p) {
    let ok = true;
    if (p.age == null || !Number.isFinite(p.age) || p.age < 1 || p.age > 120) { setError("age", "Enter an age between 1 and 120."); ok = false; }
    if (!p.gender) { setError("gender", "Select a gender."); ok = false; }
    if (!p.symptoms.length) { $("#symptoms-err").textContent = "Select at least one symptom."; ok = false; }
    if (!ok) {
      const first = $("[aria-invalid='true']") || $("#sym-search");
      first.focus();
    }
    return ok;
  }

  async function submit(ev) {
    ev.preventDefault();
    clearErrors();
    const payload = readForm();
    if (!validateClient(payload)) return;
    const btn = $("#predict-btn");
    btn.disabled = true; btn.setAttribute("aria-busy", "true");
    $("#predict-label").textContent = "Analyzing symptoms...";
    try {
      const result = await api("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      renderResult(result);
    } catch (e) {
      if (e.field && $("#" + e.field + "-err")) setError(e.field, e.message);
      else if (e.field === "symptoms") $("#symptoms-err").textContent = e.message;
      const alert = $("#form-alert"); alert.textContent = e.message; alert.hidden = false;
    } finally {
      btn.disabled = false; btn.removeAttribute("aria-busy");
      $("#predict-label").textContent = "Predict disease";
    }
  }

  /* ------------------------------------------------------------ results */
  const PATIENT_LABELS = {
    age: "Age", gender: "Gender", height_cm: "Height (cm)", weight_kg: "Weight (kg)", bmi: "BMI (calculated)",
    blood_group: "Blood group", smoking_status: "Smoking", alcohol_consumption: "Alcohol",
    existing_conditions: "Existing conditions", allergies: "Allergies", medications: "Medications",
    family_history: "Family history",
  };

  function renderResult(r) {
    $("#res-disease").textContent = r.predicted_disease;
    $("#res-conf").textContent = pct(r.ensemble_confidence);
    $("#res-conf-bar").style.width = r.ensemble_confidence * 100 + "%";
    const n = Object.keys(r.model_predictions).length;
    $("#res-agree").textContent = `${r.models_agreeing} of ${n} models chose this disease.`;
    $("#res-mode").textContent = `Ensemble confidence: ${r.ensemble_mode === "equal" ? "equal-weight" : "performance-weighted"} soft voting.`;

    $("#res-warnings").replaceChildren(...r.warnings.map((w) => h("li", { text: w })));

    $("#res-votes").replaceChildren(...Object.values(r.model_details).map((m) => {
      const agrees = m.prediction === r.predicted_disease;
      return h("div", { class: "vote" },
        h("div", { class: "vote-model" }, m.label, h("small", { text: `weight ${pct(m.weight, 1)}` })),
        h("div", { class: "vote-pred" }, h("span", { class: "dot" + (agrees ? "" : " differs"), title: agrees ? "Agrees with ensemble" : "Differs from ensemble" }), m.prediction),
        h("div", { class: "vote-bar" }, bar(m.confidence, agrees ? "" : "amber"), h("span", { text: pct(m.confidence, 0) })));
    }));

    $("#res-alts").replaceChildren(...r.top_predictions.map((t) =>
      h("div", { class: "alt" }, h("span", { text: t.disease }), h("b", { text: pct(t.probability) }), bar(t.probability))));

    $("#res-desc").textContent = r.description || "No description is available in the supplied dataset for this disease.";
    $("#res-prec").replaceChildren(...(r.precautions.length
      ? r.precautions.map((p) => h("li", { text: p.charAt(0).toUpperCase() + p.slice(1) }))
      : [h("li", { text: "No precautions are listed in the supplied dataset." })]));

    $("#res-symptoms").replaceChildren(...r.selected_symptoms.map((s) => h("span", { class: "chip" }, s.label, sevBadge(s.severity))));
    const sev = r.severity_summary;
    $("#res-sev").textContent = sev.max == null ? "" : `Combined severity weight: ${sev.total} (highest single symptom: ${sev.max}). Context only; not used by the models.`;
    const dl = $("#res-patient"); dl.replaceChildren();
    for (const [k, v] of Object.entries(r.patient_info)) dl.append(h("dt", { text: PATIENT_LABELS[k] || k }), h("dd", { text: String(v) }));

    const sec = $("#results");
    sec.hidden = false;
    sec.scrollIntoView({ behavior: "smooth", block: "start" });
    sec.focus({ preventScroll: true });
  }

  /* -------------------------------------------------------- performance */
  let perfLoaded = false;
  async function loadPerformance() {
    if (perfLoaded) return;
    const root = $("#perf-root");
    try {
      const d = await api("/api/model-performance");
      root.replaceChildren(...buildPerformance(d));
      perfLoaded = true;
    } catch (e) {
      root.replaceChildren(h("p", { class: "alert", role: "alert", text: e.message }));
    }
  }

  function buildPerformance(d) {
    const keys = Object.keys(d.models);
    const ens = d.ensemble, selMode = ens.selected_mode;
    const ensRows = [["equal", "Ensemble (equal weights)"], ["performance_weighted", "Ensemble (performance-weighted)"]];
    const all = [
      ...keys.map((k) => ({ name: d.models[k].label, m: d.models[k] })),
      ...ensRows.map(([mode, name]) => ({ name, m: ens.modes[mode], ens: true, selected: mode === selMode })),
    ];
    const f = (v) => (v == null ? "n/a" : v.toFixed(3));

    const table = h("div", { class: "table-wrap" }, h("table", {},
      h("caption", { class: "hint", text: "Macro-averaged over all diseases. Validation = stratified cross-validation on the training split." }),
      h("thead", {}, h("tr", {}, ...["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 (mean \u00b1 sd)"].map((t) => h("th", { scope: "col", text: t })))),
      h("tbody", {}, ...all.map((r) => h("tr", { class: r.ens ? "ens" : "" },
        h("td", {}, r.name, r.selected ? h("span", { class: "tag", text: "in use" }) : null),
        h("td", { text: f(r.m.accuracy) }), h("td", { text: f(r.m.precision) }), h("td", { text: f(r.m.recall) }),
        h("td", { text: f(r.m.f1) }), h("td", { text: f(r.m.roc_auc_ovr_macro) }),
        h("td", { text: `${f(r.m.cv_f1_mean)} \u00b1 ${f(r.m.cv_f1_std)}` }))))));

    const chart = (title, metric) => h("section", { class: "card" }, h("h3", { text: title }),
      ...all.map((r) => h("div", { class: "metric-row" }, h("span", { text: r.name.replace("Ensemble ", "Ens. ") }), bar(r.m[metric], r.ens ? "" : "amber"), h("b", { text: pct(r.m[metric], 1) }))));

    const w = ens.weights.performance_weighted, v = ens.validation_macro_f1;
    const weighting = h("section", { class: "card prose" }, h("h3", { text: "Ensemble weighting" }),
      h("p", {}, `Both modes use soft voting: the class probabilities of the six models are combined as a weighted average. Equal-weight gives each model 1/6. Performance-weighted gives each model a share proportional to its cross-validated macro-F1 on the training split.`),
      h("p", {}, `Validation macro-F1: equal ${f(v.equal)}, performance-weighted ${f(v.performance_weighted)} (weights re-estimated on the other folds for each held-out fold). Weighted voting is only adopted if it wins by more than 0.01, so the application uses ${selMode === "equal" ? "equal weights" : "performance-weighted voting"}.`),
      ...keys.map((k) => h("div", { class: "metric-row" }, h("span", { text: d.models[k].label }), bar(w[k] / Math.max(...Object.values(w))), h("b", { text: pct(w[k], 1) }))));

    const imgSelect = h("select", { id: "cm-select" }, ...[...keys.map((k) => [k, d.models[k].label]), ["ensemble", "Ensemble (in use)"]].map(([k, l]) => h("option", { value: k, text: l })));
    const img = h("img", { class: "cm-img", alt: "", loading: "lazy" });
    const setImg = () => {
      const key = imgSelect.value;
      img.src = `/api/report-image/confusion_matrix_${key}`;
      img.alt = `Confusion matrix on the test set for ${imgSelect.selectedOptions[0].textContent}. Rows are true diseases, columns are predicted diseases.`;
    };
    imgSelect.addEventListener("change", setImg); setImg();
    const cm = h("section", { class: "card" }, h("h3", { text: "Confusion matrix" }),
      h("div", { class: "cm-controls" }, h("label", { for: "cm-select", text: "Model" }), imgSelect),
      h("p", { class: "hint", text: `Test set of ${d.dataset.test_rows} records across ${d.dataset.classes.length} diseases. A perfect model fills only the diagonal.` }), img);

    const ds = d.dataset, lk = d.leakage_demo;
    const notes = h("section", { class: "card prose" }, h("h3", { text: "How to read these numbers" }),
      h("ul", {},
        h("li", { text: `The source file has ${ds.raw_rows} rows but only ${ds.unique_rows} unique ones; ${ds.exact_duplicates_dropped} exact duplicates were removed before splitting.` }),
        h("li", { text: `A naive split of the raw file would place an exact copy of ${pct(lk.naive_split_test_rows_with_exact_copy_in_train, 0)} of test rows in the training data and report ${pct(lk.naive_split_rf_accuracy, 0)} accuracy. That is leakage, so it is not used here.` }),
        h("li", { text: `The test set is small (${ds.test_rows} records, one to two per disease), so a single mistake moves accuracy by about ${(100 / ds.test_rows).toFixed(1)} points.` }),
        h("li", { text: "Most records are subsets of the same disease's symptom list, which suggests the data is synthetic. High scores show the models learned this dataset, not that they would be accurate for real patients." }),
        h("li", { text: "On this dataset Random Forest and KNN match or beat the ensemble. The ensemble is kept so the result does not depend on any single model, and it down-weights the weaker Decision Tree and Gradient Boosting, but these numbers do not show it is better than Random Forest." })));

    const imp = h("section", { class: "card" }, h("h3", { text: "Most influential symptoms (Random Forest)" }),
      h("p", { class: "hint", text: "Share of the forest's splitting power attributed to each symptom, with its dataset severity weight." }),
      ...d.rf_symptom_importance.slice(0, 10).map((s) => h("div", { class: "metric-row" }, h("span", { text: s.symptom.replace(/_/g, " ") }), bar(s.importance / d.rf_symptom_importance[0].importance), h("b", { text: pct(s.importance, 1) }))));

    return [
      h("section", { class: "card" }, h("h2", { text: "Results on the held-out test set" }), table),
      h("div", { class: "perf-grid two" }, chart("Accuracy by model", "accuracy"), chart("F1-score by model", "f1")),
      h("div", { class: "perf-grid two" }, weighting, imp),
      h("div", { class: "perf-grid" }, cm, notes),
    ];
  }

  /* --------------------------------------------------------------- init */
  function resetAll() {
    $("#predict-form").reset();
    selected.clear(); boxes.forEach((b) => { b.checked = false; });
    renderChips(); filterSymptoms(); clearErrors();
    $("#results").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("#predict-form").addEventListener("submit", submit);
    $("#sym-search").addEventListener("input", filterSymptoms);
    $("#sym-search").addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      const first = catalog.find((s) => !rows.get(s.name).hidden && !selected.has(s.name));
      if (first) toggle(first, true);
    });
    $("#clear-symptoms").addEventListener("click", () => { selected.clear(); boxes.forEach((b) => { b.checked = false; }); renderChips(); });
    $("#age").addEventListener("input", () => setError("age", ""));
    $("#gender").addEventListener("change", () => setError("gender", ""));
    $("#reset-btn").addEventListener("click", resetAll);
    $("#edit-btn").addEventListener("click", () => $("#predict-form").scrollIntoView({ behavior: "smooth" }));
    window.addEventListener("hashchange", route);
    route();
    loadSymptoms();
  });
})();

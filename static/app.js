const $ = (id) => document.getElementById(id);
const drop = $("drop"), input = $("file"), preview = $("preview");
let current = null;

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showError("That file is not an image.");
  current = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("frame-text").hidden = true;
  $("check").disabled = false;
  $("clear").hidden = false;
  resetResult();
}

function resetResult() {
  $("result").hidden = true; $("error").hidden = true; $("empty").hidden = false;
}

function showError(msg) {
  $("empty").hidden = true; $("result").hidden = true;
  $("error").textContent = msg; $("error").hidden = false;
}

input.addEventListener("change", () => setFile(input.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

$("clear").addEventListener("click", () => {
  current = null; input.value = ""; preview.hidden = true; $("frame-text").hidden = false;
  $("check").disabled = true; $("clear").hidden = true; resetResult();
});

$("check").addEventListener("click", async () => {
  if (!current) return;
  const btn = $("check");
  btn.disabled = true; btn.textContent = "Checking…";
  try {
    const fd = new FormData(); fd.append("image", current);
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) return showError(data.error || "Something went wrong. Try again.");
    render(data);
  } catch {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    btn.disabled = false; btn.textContent = "Check this note";
  }
});

function render(d) {
  $("empty").hidden = true; $("error").hidden = true; $("result").hidden = false;
  const stamp = $("stamp");
  const cls = d.status === "ok" ? d.label : d.status;
  const text = { real: "Likely real", fake: "Likely fake", uncertain: "Not sure", retake: "Retake photo" }[cls];
  stamp.textContent = text;
  stamp.className = "stamp pop " + cls;
  $("msg").textContent = d.message;
  $("conf").textContent = d.confidence + "%";
  $("meter").style.width = d.confidence + "%";
  $("sharp").textContent = d.details.sharpness;
  $("edge").textContent = d.details.edge_density;
}

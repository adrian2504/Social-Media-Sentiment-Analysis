document.getElementById("toggle").onclick =
  () => document.getElementById("sidebar").classList.toggle("collapsed");

// initial bar chart
fetch("/static/data/distribution.json")
  .then(r => r.json())
  .then(d => {
    new Chart(
      document.getElementById("sentChart"),
      { type: "bar",
        data: { labels: d.labels,
                datasets: [{ data: d.counts }] } });
  });

// live sentiment
async function checkSent() {
  const txt = document.getElementById("postTxt").value;
  const res = await fetch("/api/sentiment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: txt })
  });
  const j = await res.json();
  document.getElementById("mood").innerText =
  `${j.emoji}  ${j.label} (${(j.score*100).toFixed(1)}%)`;
}

// next-word generator
async function genText() {
  const seed = document.getElementById("seedTxt").value;
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seed })
  });
  const j = await res.json();
  document.getElementById("genOut").innerText = j.generated;
}

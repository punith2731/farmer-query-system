// Crop Price Forecasting Dashboard interactions
// - Sidebar toggle (mobile)
// - Dark mode toggle with persistence
// - Mock async prediction fetch with spinner
// - Trend chart + confidence interval updates

const menuToggle = document.getElementById("menuToggle");
const sidebar = document.getElementById("sidebar");
const themeToggle = document.getElementById("themeToggle");
const loadingOverlay = document.getElementById("loadingOverlay");
const refreshBtn = document.getElementById("refreshBtn");
const cropSelect = document.getElementById("cropSelect");
const mandiSelect = document.getElementById("mandiSelect");

const price7 = document.getElementById("price7");
const price14 = document.getElementById("price14");
const price30 = document.getElementById("price30");
const confidenceText = document.getElementById("confidenceText");
const confidenceMarker = document.getElementById("confidenceMarker");

let trendChart;

const cropBasePrice = {
  wheat: 2200,
  rice: 2500,
  maize: 1950,
  cotton: 6200,
  soybean: 4300
};

function formatPrice(value) {
  return `₹${Math.round(value).toLocaleString("en-IN")} / qtl`;
}

function createDaysLabels(days) {
  return Array.from({ length: days }, (_, i) => `D${i + 1}`);
}

function generateTrendSeries(base, len = 30) {
  let current = base * 0.94;

  return Array.from({ length: len }, () => {
    const drift = (Math.random() - 0.45) * (base * 0.01);
    current = Math.max(base * 0.7, current + drift);
    return Math.round(current);
  });
}

function forecastFromBase(base) {
  const f7 = base * 1.01;
  const f14 = base * 1.03;
  const f30 = base * 1.06;

  const ciLow = f14 * 0.94;
  const ciHigh = f14 * 1.06;

  return { f7, f14, f30, ciLow, ciHigh };
}

function updateForecastCards(data) {
  price7.textContent = formatPrice(data.f7);
  price14.textContent = formatPrice(data.f14);
  price30.textContent = formatPrice(data.f30);

  confidenceText.textContent = `Expected: ${formatPrice(data.f14)} (95% CI: ₹${Math.round(data.ciLow).toLocaleString("en-IN")} – ₹${Math.round(data.ciHigh).toLocaleString("en-IN")})`;

  // Position marker between low-high; use expected as % center for visual cue.
  confidenceMarker.style.left = "58%";
}

function buildTrendChart(basePrice) {
  const canvas = document.getElementById("priceTrendChart");
  const labels = createDaysLabels(30);
  const series = generateTrendSeries(basePrice, 30);

  if (trendChart) {
    trendChart.destroy();
  }

  trendChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Modal Price (₹/qtl)",
          data: series,
          borderColor: "#2f80ed",
          backgroundColor: "rgba(47, 128, 237, 0.14)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label(context) {
              return ` ₹${context.parsed.y.toLocaleString("en-IN")}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false }
        },
        y: {
          beginAtZero: false,
          ticks: {
            callback(value) {
              return `₹${Number(value).toLocaleString("en-IN")}`;
            }
          }
        }
      }
    }
  });
}

async function fetchPredictions() {
  // Simulate network call; replace with real API fetch as needed.
  loadingOverlay.classList.remove("hidden");

  const crop = cropSelect.value;
  const mandi = mandiSelect.value;
  const mandiFactor = (mandi.charCodeAt(0) % 5) * 0.007;
  const base = cropBasePrice[crop] * (1 + mandiFactor);

  await new Promise((resolve) => setTimeout(resolve, 900));

  const forecast = forecastFromBase(base);
  updateForecastCards(forecast);
  buildTrendChart(base);

  loadingOverlay.classList.add("hidden");
}

function setupSidebarToggle() {
  menuToggle.addEventListener("click", () => {
    const isOpen = sidebar.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
  });

  // Close sidebar on outside click for better mobile UX
  document.addEventListener("click", (event) => {
    const clickedInsideSidebar = sidebar.contains(event.target);
    const clickedToggle = menuToggle.contains(event.target);

    if (!clickedInsideSidebar && !clickedToggle && sidebar.classList.contains("open")) {
      sidebar.classList.remove("open");
      menuToggle.setAttribute("aria-expanded", "false");
    }
  });
}

function setupThemeToggle() {
  const storedTheme = localStorage.getItem("dashboard-theme");
  if (storedTheme === "dark") {
    document.body.classList.add("dark");
  }

  themeToggle.addEventListener("click", () => {
    const dark = document.body.classList.toggle("dark");
    localStorage.setItem("dashboard-theme", dark ? "dark" : "light");
    themeToggle.innerHTML = dark
      ? '<i class="fa-solid fa-sun"></i>'
      : '<i class="fa-solid fa-moon"></i>';
  });

  if (document.body.classList.contains("dark")) {
    themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
  }
}

function bindEvents() {
  refreshBtn.addEventListener("click", fetchPredictions);
  cropSelect.addEventListener("change", fetchPredictions);
  mandiSelect.addEventListener("change", fetchPredictions);
  window.addEventListener("resize", () => {
    if (trendChart) trendChart.resize();
  });
}

function init() {
  setupSidebarToggle();
  setupThemeToggle();
  bindEvents();
  fetchPredictions();
}

document.addEventListener("DOMContentLoaded", init);

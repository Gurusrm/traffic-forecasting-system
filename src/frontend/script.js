// Global state
let currentData = null;
let currentPrediction = null;
let sensorLocations = [];
let map = null;
let markersLayer = null;
let heatLayer = null;
let markers = [];

const STATUS = document.getElementById('status');
const SAMPLE_BTN = document.getElementById('getSampleBtn');
const SENSOR_INPUT = document.getElementById('sensorSelect');

async function logError(msg) {
    try {
        await fetch('/log_error', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, level: 'error' })
        });
    } catch (e) {
        console.error("Failed to log error:", e);
    }
}

window.addEventListener('error', (event) => {
    const msg = `Uncaught Error: ${event.message}`;
    STATUS.textContent = msg;
    STATUS.style.color = "#f87171";
    logError(`${msg} at ${event.filename}:${event.lineno}`);
});

window.addEventListener('unhandledrejection', (event) => {
    const msg = `Unhandled Promise Rejection: ${event.reason}`;
    STATUS.textContent = msg;
    STATUS.style.color = "#f87171";
    logError(msg);
});

// Event Listeners
SAMPLE_BTN.addEventListener('click', fetchSampleAndPredict);
SENSOR_INPUT.addEventListener('change', updateLineChart);

// Init Map on Load
document.addEventListener('DOMContentLoaded', async () => {
    STATUS.textContent = "Connecting to server...";
    STATUS.style.color = "#facc15"; // Yellow
    initMap();
    await fetchLocations();
});

function initMap() {
    // Center on Los Angeles
    map = L.map('trafficMap').setView([34.0522, -118.2437], 11);

    // Dark mode map tiles (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // Layer Groups
    markersLayer = L.layerGroup().addTo(map);
    if (typeof L.heatLayer === 'function') {
        heatLayer = L.heatLayer([], { radius: 25, blur: 15, maxZoom: 13 }).addTo(map);
    } else {
        console.error("L.heatLayer is not defined. Check internet connection or CDN.");
        logError("L.heatLayer missing (CDN issue?)");
    }

    // Controls
    const overlayMaps = {
        "Sensor Markers": markersLayer,
        "Traffic Heatmap": heatLayer
    };

    L.control.layers(null, overlayMaps, { collapsed: false }).addTo(map);
}

async function fetchLocations() {
    try {
        const res = await fetch('/locations');
        if (!res.ok) throw new Error("Failed to fetch locations");
        sensorLocations = await res.json();

        // Create initial markers (invisible or base color)
        sensorLocations.forEach((loc, index) => {
            const marker = L.circleMarker([loc.lat, loc.lng], {
                radius: 5,
                fillColor: '#334155', // Slate-700
                color: '#000',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            });

            marker.bindPopup(`<b>Sensor ID:</b> ${loc.id}<br><b>Index:</b> ${index}`);
            marker.on('click', () => {
                SENSOR_INPUT.value = index;
                updateLineChart();
            });

            markersLayer.addLayer(marker);
        });

        STATUS.textContent = "System Ready. Connected to Dataset.";
        STATUS.style.color = "#4ade80"; // Green

    } catch (err) {
        console.error("Error fetching locations:", err);
        STATUS.textContent = `Connection Failed: ${err.message}`;
        STATUS.style.color = "#f87171"; // Red
        logError(`fetchLocations setup error: ${err.message}`);
    }
}

function updateMapColors() {
    if (!currentPrediction || markers.length === 0) return;

    // We visualize the prediction 'speed' at step 0 (next 5 min)
    // Feature 0 is speed.
    const speedValues = currentPrediction[0].map(node => node[0]);

    // Simple color scale: Green (fast) -> Red (slow)
    // Assuming normalized or standard speed. METR-LA is usually MPH.
    // ~60+ Green, ~40 Yellow, <20 Red.

    speedValues.forEach((speed, index) => {
        let color = '#ef4444'; // Red
        if (speed > 55) color = '#22c55e'; // Green
        else if (speed > 35) color = '#eab308'; // Yellow
        else if (speed > 20) color = '#f97316'; // Orange

        if (markers[index]) {
            markers[index].setStyle({ fillColor: color });
            // Update popup with speed
            const loc = sensorLocations[index];
            markers[index].setPopupContent(`
                <b>Sensor ID:</b> ${loc ? loc.id : 'N/A'}<br>
                <b>Index:</b> ${index}<br>
                <b>Predicted Speed:</b> ${speed.toFixed(1)} mph
            `);
        }
    });

    // Update Heatmap (Congestion Intensity)
    const heatPoints = [];
    speedValues.forEach((speed, index) => {
        const loc = sensorLocations[index];
        if (loc) {
            // Invert speed for congestion intensity (0-70mph mapped to 1.0-0.0)
            let intensity = Math.max(0, (70 - speed) / 70);
            intensity = Math.min(1.0, intensity);
            heatPoints.push([loc.lat, loc.lng, intensity]);
        }
    });

    if (heatLayer) {
        heatLayer.setLatLngs(heatPoints);
    }
}

async function fetchSampleAndPredict() {
    try {
        STATUS.textContent = "Fetching Sample...";
        SAMPLE_BTN.disabled = true;

        // 1. Get Sample
        const sampleRes = await fetch('/sample');
        if (!sampleRes.ok) throw new Error("Failed to fetch sample");
        const sampleData = await sampleRes.json();

        currentData = sampleData; // {input: [12, 207, 2], target: [12, 207, 2], start_index: ...}

        // 2. Predict
        STATUS.textContent = "Running Inference on GPU...";
        const predRes = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_data: currentData.input })
        });

        if (!predRes.ok) throw new Error("Prediction failed");
        const predResult = await predRes.json();

        // Output from backend is [B, T, N, F]. B=1. 
        // We want [T, N]. Feature 0 is speed.
        currentPrediction = predResult.prediction[0]; // [12, 207, 1] usually

        STATUS.textContent = `Analysis Complete (Sample Index: ${currentData.start_index})`;
        SAMPLE_BTN.disabled = false;

        updateCharts();
        updateMapColors();

    } catch (err) {
        console.error(err);
        STATUS.textContent = `Error: ${err.message}`;
        SAMPLE_BTN.disabled = false;
        logError(`Prediction flow error: ${err.stack || err.message}`);
    }
}

function updateCharts() {
    console.log("updateCharts called");
    if (!currentData || !currentPrediction) {
        console.warn("Missing data for charts", { currentData, currentPrediction });
        return;
    }

    try {
        updateLineChart();
    } catch (e) {
        console.error("Error updating line chart:", e);
        logError(`Line chart error: ${e.message}`);
    }

    try {
        updateHeatmap();
    } catch (e) {
        console.error("Error updating heatmap:", e);
        logError(`Heatmap error: ${e.message}`);
    }
}

function updateLineChart() {
    if (!currentData || !currentPrediction) return;

    let sensorIdx = parseInt(SENSOR_INPUT.value) || 0;
    if (sensorIdx < 0) sensorIdx = 0;
    if (sensorIdx >= sensorLocations.length && sensorLocations.length > 0) sensorIdx = sensorLocations.length - 1;
    SENSOR_INPUT.value = sensorIdx;

    // Feature 0 is speed
    const history = currentData.input.map((step, t) => ({ t: t - 12, y: step[sensorIdx][0] }));
    const target = currentData.target.map((step, t) => ({ t: t, y: step[sensorIdx][0] }));
    const prediction = currentPrediction.map((step, t) => ({ t: t, y: step[sensorIdx][0] }));

    const traceHistory = {
        x: history.map(d => d.t),
        y: history.map(d => d.y),
        mode: 'lines+markers',
        name: 'History (Input)',
        line: { color: '#94a3b8' }
    };

    const traceTarget = {
        x: target.map(d => d.t),
        y: target.map(d => d.y),
        mode: 'lines+markers',
        name: 'Actual Future',
        line: { color: '#10b981' }
    };

    const tracePred = {
        x: prediction.map(d => d.t),
        y: prediction.map(d => d.y),
        mode: 'lines+markers',
        name: 'Predicted Future',
        line: { color: '#3b82f6', dash: 'dot' }
    };

    const layout = {
        title: `Traffic Speed at Sensor #${sensorIdx}`,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#f8fafc' },
        xaxis: {
            title: 'Time Steps (Relative)',
            zerolinecolor: '#334155',
            gridcolor: '#334155'
        },
        yaxis: {
            title: 'Speed (mph)',
            zerolinecolor: '#334155',
            gridcolor: '#334155'
        },
        margin: { t: 40, b: 40, l: 40, r: 20 },
        legend: { orientation: 'h', y: 1.1 }
    };

    Plotly.newPlot('lineChart', [traceHistory, traceTarget, tracePred], layout);
}

function updateHeatmap() {
    if (!currentData || !currentPrediction) return;

    const numNodes = currentPrediction[0].length;
    const numSteps = currentPrediction.length;

    let zValues = [];
    for (let n = 0; n < numNodes; n++) {
        let row = [];
        for (let t = 0; t < numSteps; t++) {
            row.push(currentPrediction[t][n][0]);
        }
        zValues.push(row);
    }

    const data = [{
        z: zValues,
        x: Array.from({ length: numSteps }, (_, i) => i),
        y: Array.from({ length: numNodes }, (_, i) => i),
        type: 'heatmap',
        colorscale: 'Viridis'
    }];

    const layout = {
        title: 'Predicted Network Traffic Speed (Future 1 Hr)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#f8fafc' },
        xaxis: { title: 'Time Step' },
        yaxis: { title: 'Node ID' },
        margin: { t: 40, b: 40, l: 50, r: 20 }
    };

    Plotly.newPlot('heatmapChart', data, layout);
}

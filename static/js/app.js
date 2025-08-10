
async function fetchJSON(url, opts={}){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }

function nowMs(){ return Date.now(); }

async function refreshLive(){
  try{
    const data = await fetchJSON(`/api/synthetic/latest?nowMs=${nowMs()}`);
    const list = document.getElementById('live-list');
    list.innerHTML = '';
    // Render most recent first
    for(let i=data.timestamps.length-1;i>=0;i--){
      const ts = data.timestamps[i];
      const li = document.createElement('li');
      li.innerHTML = `<div class="row"><span class="ts">${ts}</span><span class="val">${data.consumption[i]} kWh</span></div>`;
      list.appendChild(li);
    }
  }catch(e){ console.error(e); }
}

async function refreshForecast(){
  try{
    const payload = { steps: 1, nowMs: nowMs() };
    const data = await fetchJSON('/api/forecast', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    // Use last 24h history and the single next-hour forecast
    const histX = data.history.timestamps;
    const histY = data.history.consumption;
    const futX = data.forecast.timestamps; // length 1
    const futY = data.forecast.y_pred;     // length 1

    // Build a short connecting segment from the last history point to the forecast point
    const xForecast = [histX[histX.length - 1], futX[0]];
    const yForecast = [histY[histY.length - 1], futY[0]];

    const traceHist = { x: histX, y: histY, mode: 'lines+markers', name: 'History (24h)', line:{color:'#58a6ff'} };
    const traceFut  = { x: xForecast,  y: yForecast,  mode: 'lines+markers', name: 'Forecast (+1h)', line:{color:'#2ea043', width:3}, marker:{color:'#2ea043', size:8} };

    const layout = { paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117', font:{color:'#c9d1d9'}, margin:{t:30,l:40,r:10,b:40}, legend:{orientation:'h'}, xaxis:{gridcolor:'#30363d'}, yaxis:{gridcolor:'#30363d', title:'kWh'} };

    Plotly.newPlot('forecast-plot', [traceHist, traceFut], layout, {responsive:true});
  }catch(e){ console.error(e); }
}

async function init(){
  await refreshLive();
  await refreshForecast();
  // Refresh every hour (3600s); for demo, use shorter interval like 10s
  const HOUR_MS = 3600 * 1000;
  setInterval(refreshLive, HOUR_MS);
  setInterval(refreshForecast, HOUR_MS);
}

window.addEventListener('DOMContentLoaded', init);

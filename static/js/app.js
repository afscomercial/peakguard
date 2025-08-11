
async function fetchJSON(url, opts={}){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }

function nowMs(){ return Date.now(); }

function getSelectedDeviceId(){
  const sel = document.getElementById('device-selector');
  return sel ? parseInt(sel.value, 10) : 1;
}

async function refreshLive(){
  try{
    const tz = new Date().getTimezoneOffset();
    const deviceId = getSelectedDeviceId();
    const data = await fetchJSON(`/api/synthetic/latest?nowMs=${nowMs()}&tzOffsetMin=${tz}&deviceId=${deviceId}`);
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

function baseLayout(){
  return { paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117', font:{color:'#c9d1d9'}, margin:{t:30,l:40,r:10,b:40}, legend:{orientation:'h'}, xaxis:{gridcolor:'#30363d'}, yaxis:{gridcolor:'#30363d', title:'kWh'} };
}

async function refreshDiagnostics(){
  try{
    const metrics = await fetchJSON('/api/models/latest');
    if(metrics && metrics.loss_history){
      // Summary grid
      const sum = document.getElementById('training-summary');
      const lastValLoss = (metrics.loss_history.val || []).slice(-1)[0];
      const lastTrainLoss = (metrics.loss_history.train || []).slice(-1)[0];
      const lastRmse = (metrics.rmse_history || []).filter(v=>v!=null).slice(-1)[0];
      // thresholds (tunable)
      const rmseThreshold = 0.2; // kWh
      const statusGood = (typeof lastRmse === 'number') ? (lastRmse <= rmseThreshold) : false;
      const statusColor = statusGood ? '#2ea043' : '#f85149';
      sum.innerHTML = `
        <div class="grid">
          <div class="cell"><div class="label">Model ID</div><div class="value">${metrics.model_id ?? '-'}</div></div>
          <div class="cell"><div class="label">Created</div><div class="value">${metrics.created_at ?? '-'}</div></div>
          <div class="cell"><div class="label">Val Loss</div><div class="value">${(lastValLoss ?? '-')}</div></div>
          <div class="cell"><div class="label">Train Loss</div><div class="value">${(lastTrainLoss ?? '-')}</div></div>
          <div class="cell"><div class="label">Val RMSE (kWh)</div><div class="value">${(lastRmse ?? '-')}</div></div>
          <div class="cell"><div class="label">Status</div><div class="value" style="color:${statusColor}">${statusGood ? 'Good' : 'Needs Improvement'}</div></div>
        </div>
      `;

      const loss = metrics.loss_history.train || [];
      const valLoss = metrics.loss_history.val || [];
      const xIdx = Array.from({length: Math.max(loss.length, valLoss.length)}, (_,i)=>i+1);
      const tLoss = { x: xIdx, y: loss, mode:'lines', name:'Train Loss'};
      const vLoss = { x: xIdx, y: valLoss, mode:'lines', name:'Val Loss'};
      Plotly.newPlot('loss-plot', [tLoss, vLoss], baseLayout(), {responsive:true});

      // RMSE plot removed per request

      if(metrics.test_plot){
        const yTrue = { x: Array.from({length: metrics.test_plot.y_true.length}, (_,i)=>i), y: metrics.test_plot.y_true, mode:'lines', name:'y_true'};
        const yPred = { x: Array.from({length: metrics.test_plot.y_pred.length}, (_,i)=>i), y: metrics.test_plot.y_pred, mode:'lines', name:'y_pred'};
        Plotly.newPlot('test-plot', [yTrue, yPred], baseLayout(), {responsive:true});
      }
    }
  }catch(e){ console.error(e); }
}

async function refreshForecast(){
  try{
    const tz = new Date().getTimezoneOffset();
    const deviceId = getSelectedDeviceId();
    const payload = { steps: 1, nowMs: nowMs(), tzOffsetMin: tz, deviceId };
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

    Plotly.newPlot('forecast-plot', [traceHist, traceFut], baseLayout(), {responsive:true});
  }catch(e){ console.error(e); }
}

async function init(){
  // populate devices
  try{
    const devs = await fetchJSON('/api/devices');
    const sel = document.getElementById('device-selector');
    sel.innerHTML = '';
    for(const d of devs){
      const opt = document.createElement('option');
      opt.value = d.id;
      opt.textContent = `${d.name} (${d.timezone})`;
      sel.appendChild(opt);
    }
  }catch(e){ console.error(e); }

  await refreshLive();
  await refreshForecast();
  await refreshDiagnostics();
  // Refresh every hour (3600s); for demo, use shorter interval like 10s
  const HOUR_MS = 3600 * 1000;
  setInterval(refreshLive, HOUR_MS);
  setInterval(refreshForecast, HOUR_MS);
  setInterval(refreshDiagnostics, HOUR_MS);
  document.getElementById('device-selector').addEventListener('change', async ()=>{
    await refreshLive();
    await refreshForecast();
    await refreshDiagnostics();
  });
}

window.addEventListener('DOMContentLoaded', init);

async function loadStatus() {
  try {
    const response = await fetch('/status');
    const data = await response.json();

    document.getElementById('risk').textContent = Number(data.risk || 0).toFixed(2);
    document.getElementById('label').textContent = data.label || 'alert';
    document.getElementById('speed').textContent = `${Number(data.speed_kmph || 0).toFixed(1)} km/h`;
    document.getElementById('ear').textContent = data.ear === null ? '--' : Number(data.ear).toFixed(3);
    document.getElementById('mar').textContent = data.mar === null ? '--' : Number(data.mar).toFixed(3);
    document.getElementById('bpm').textContent = data.bpm === null ? '--' : `${Number(data.bpm).toFixed(1)} BPM`;
    document.getElementById('lane').textContent = Number(data.lane_deviation || 0).toFixed(3);
    document.getElementById('yawns').textContent = data.yawns ?? 0;
    document.getElementById('reason').textContent = data.reason || 'Monitoring...';

    const pill = document.getElementById('status-pill');
    const label = (data.label || 'alert').toLowerCase();

    if (label.includes('high') || label.includes('drowsy') || label.includes('distracted')) {
      pill.className = 'status-pill danger';
    } else if (label === 'alert') {
      pill.className = 'status-pill safe';
    } else {
      pill.className = 'status-pill alert';
    }

    pill.textContent = label;
  } catch (error) {
    console.error('Status fetch error:', error);
  }
}

setInterval(loadStatus, 500);
loadStatus();

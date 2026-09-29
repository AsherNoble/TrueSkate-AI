import {mapPoint, Capture} from './pointer.js';
const token = location.hash.slice(1) || sessionStorage.getItem('xr-control-token') || '';
if (token) sessionStorage.setItem('xr-control-token', token);
history.replaceState(null, '', location.pathname);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
for (const name of ['XR1', 'XR2']) mount(name);
function mount(name) {
  const card = document.createElement('section'); card.className = 'device';
  card.innerHTML = `<div class="heading"><h2>${name}</h2><button class="enlarge">Enlarge</button></div><div class="controls"><button data-command="connect">Connect</button><button data-command="disconnect">Disconnect</button><button data-command="activate">Open True Skate</button></div><div class="screen blocked"><img alt="${name} live screen" draggable="false"><canvas aria-label="${name} touch input"></canvas></div><div class="status" role="status">Disconnected</div><div class="freshness stale">Waiting for video</div>`;
  document.querySelector('#devices').append(card);
  const screen = card.querySelector('.screen'), img = card.querySelector('img');
  const canvas = card.querySelector('canvas'), ctx = canvas.getContext('2d');
  const status = card.querySelector('.status'), freshness = card.querySelector('.freshness');
  const capture = new Capture();
  let state = {}, localBusy = false, localError = '', frame = 0, received = -Infinity, statusAt = -Infinity;
  let gestureEpoch = null, lastUrl = null;
  const fresh = () => performance.now() - received < 1500;
  const ready = () => state.connected && state.available && !state.busy && !state.uncertain && !localBusy && fresh() && performance.now() - statusAt < 2000;
  const point = e => mapPoint(e.clientX, e.clientY, canvas.getBoundingClientRect());
  function draw() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width * devicePixelRatio); canvas.height = Math.round(rect.height * devicePixelRatio);
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const scale = Math.min(rect.width / 414, rect.height / 896), w = 414 * scale, h = 896 * scale;
    const ox = (rect.width - w) / 2, oy = (rect.height - h) / 2;
    ctx.strokeStyle = '#70f5c5'; ctx.fillStyle = '#70f5c5'; ctx.lineWidth = 3;
    ctx.beginPath(); capture.points.forEach((p, i) => i ? ctx.lineTo(ox+p.x*w, oy+p.y*h) : ctx.moveTo(ox+p.x*w, oy+p.y*h)); ctx.stroke();
    if(capture.points.length) { const p=capture.points.at(-1); ctx.beginPath();ctx.arc(ox+p.x*w,oy+p.y*h,5,0,2*Math.PI);ctx.fill(); }
  }
  function cancel() { const id=capture.id; capture.cancel(); if(id!==null && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id); draw(); }
  async function api(endpoint, body) {
    const response = await fetch(`/api/${name}/${endpoint}`, {method:body ? 'POST':'GET', cache:'no-store',
      headers:{'X-Control-Token':token, ...(body ? {'Content-Type':'application/json'}:{})},
      ...(body ? {body:JSON.stringify(body)}:{}), signal:AbortSignal.timeout(25000)});
    if (!response.ok) throw new Error((await response.json()).error || 'Request failed');
    return response;
  }
  async function command(kind, points) {
    if(localBusy) return;
    localBusy=true; localError=''; cancel();
    status.textContent=kind==='gesture' ? 'Executing gesture…' : `${kind}…`;
    const body=kind==='connect' ? {} : {epoch:state.epoch, sequence:state.next_sequence, frame, ...(points?{points}:{})};
    render();
    try { state=await (await api(kind, body)).json(); statusAt=performance.now(); }
    catch(error) { localError=error.message+' — not retried'; statusAt=-Infinity; }
    finally {localBusy=false; render();}
  }
  function render() {
    const allowed=ready(); screen.classList.toggle('blocked',!allowed);
    if(!allowed && capture.id!==null) cancel();
    for(const button of card.querySelectorAll('[data-command]')) {
      const kind=button.dataset.command;
      button.disabled=localBusy || state.busy || state.uncertain || (kind==='connect' ? state.connected : !state.connected) || (kind==='activate' && !allowed);
    }
    if(!localBusy) status.textContent=localError || state.message || 'Disconnected';
    freshness.textContent=fresh() ? `Live · frame ${frame}` : 'Video stale / unavailable — input disabled';
    freshness.classList.toggle('stale',!fresh());
  }
  card.querySelector('.enlarge').onclick=() => {cancel();card.classList.toggle('enlarged');draw();};
  card.querySelectorAll('[data-command]').forEach(b=>b.onclick=()=>command(b.dataset.command));
  canvas.onpointerdown=e=>{if(e.button!==0 || !e.isPrimary || !ready())return;e.preventDefault();if(capture.start(e.pointerId,point(e),performance.now())){gestureEpoch=state.epoch;canvas.setPointerCapture(e.pointerId);draw();}};
  canvas.onpointermove=e=>{if(capture.id===null)return;capture.move(e.pointerId,point(e),performance.now());draw();};
  canvas.onpointerup=e=>{if(e.pointerId!==capture.id)return;if(!ready() || gestureEpoch!==state.epoch){cancel();return;}const points=capture.finish(e.pointerId,point(e),performance.now());cancel();if(points)command('gesture',points);};
  canvas.onpointercancel=cancel; canvas.onlostpointercapture=cancel;
  canvas.oncontextmenu=e=>e.preventDefault();
  window.addEventListener('blur',cancel); window.addEventListener('resize',cancel);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){received=-Infinity;cancel();}});
  window.addEventListener('keydown',e=>{if(e.key==='Escape')cancel();});
  new ResizeObserver(()=>{cancel();}).observe(screen);
  setInterval(()=>{if(capture.id!==null && performance.now()-capture.started>5000)cancel();render();},100);
  (async()=>{while(true){try{const response=await api('status');const next=await response.json();if(next.epoch!==state.epoch)cancel();state=next;statusAt=performance.now();}catch(error){statusAt=-Infinity;localError=error.message;}render();await sleep(500);}})();
  (async()=>{while(true){
    if(!document.hidden) try {
      const start=performance.now(), response=await api('frame');
      const seq=Number(response.headers.get('X-Frame-Sequence'));
      const age=Number(response.headers.get('X-Frame-Age-Ms'));
      const url=URL.createObjectURL(await response.blob());
      const decoded=new Image(); decoded.src=url;
      try {await decoded.decode();
        if(seq!==frame && performance.now()-start+age<1500){img.src=url;if(lastUrl)URL.revokeObjectURL(lastUrl);lastUrl=url;frame=seq;received=start-age;}
        else URL.revokeObjectURL(url);
      } catch(error){URL.revokeObjectURL(url);throw error;}
    } catch {received=-Infinity;}
    await sleep(80);
  }})();
}

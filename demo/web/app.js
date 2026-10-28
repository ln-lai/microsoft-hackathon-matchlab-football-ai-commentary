'use strict';

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const state = {health:null,matches:[],match:null,frame:125,time:5,playing:false,speed:1,selectedId:null,prediction:null,pending:false,requestGeneration:0,matchGeneration:0,lastTick:null,lastRenderedFrame:-1,error:null,audience:'fan',commentary:null,commentaryPending:false,commentaryGeneration:0,commentaryError:null,commentaryController:null,highlights:[],activeHighlight:null,showAll:false,stopAt:null,highlightPending:false};
Object.assign(state,{focusId:null,focusGeneration:0,segment:null,stepMode:false,clockOrigin:null,clockTime:0,voice:null,voicePending:false,voiceError:null,voiceGeneration:0,audio:null,audioEnabled:false,audioStarting:false,muted:false,volume:.8,captions:true,emptyReason:null});
Object.assign(state,{replayMode:'highlight',commentaryOn:true,sessionStarted:false,runGeneration:0,runSchedule:[],audioBase:0,audioHighlight:null,voiceSkipped:null,outputAudio:null,audioUnlocked:false,unlockGeneration:0});
const CONTENT_VERSION='matchlab-commentary-v3';
const predictionCache=new Map(),voiceCache=new Map(),voiceRequests=new Map(),voiceInsightRefresh=new Set(),mediaCache=new Map();
const COLORS = ['#d5ef85','#ddaff4'];
const SCENES = {
  'space-run':{title:'Find Space',subtitle:'Watch a runner open a gap.',guidance:'Follow the runner, then compare the predicted position with the recorded next second.'},
  'defensive-shift':{title:'Defensive Shift',subtitle:'See defenders move together.',guidance:'Watch the defensive line shift and see whose nearest-opponent gap changes.'},
  'transition':{title:'Transition',subtitle:'Follow a change of direction.',guidance:'Watch the transition unfold, then check the model’s next-second estimate.'}
};
const fmt = (value, digits=2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const validXY = (xy) => Array.isArray(xy) && xy.length===2 && xy.every(Number.isFinite);
const distance = (a,b) => Math.hypot(a[0]-b[0],a[1]-b[1]);
const clock = (seconds, precision=0) => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(precision).padStart(precision?3+precision:2,'0')}`;

function svgNode(tag, attrs, text) { const el=document.createElementNS(SVG_NS,tag); for(const [k,v] of Object.entries(attrs||{}))el.setAttribute(k,String(v)); if(text!==undefined)el.textContent=text; return el; }
function svgAdd(parent,tag,attrs,text){const el=svgNode(tag,attrs,text);parent.append(el);return el;}
function clear(el){el.replaceChildren();}
function emphasizeText(el,text){
  if(!el)return;const value=String(text??''),pattern=/\b(?:Home|Away)\s+\d{1,2}\b|[+-]?\d+(?:\.\d+)?\s*(?:m\b|seconds?\b|s\b)|\b(?:wider|narrower|tighter|opened|lost|gap|space|prediction|forecast|actual)\b/gi;
  const fragments=[];let start=0;for(const match of value.matchAll(pattern)){if(match.index>start)fragments.push(document.createTextNode(value.slice(start,match.index)));const span=document.createElement(/^(Home|Away)\s/i.test(match[0])?'strong':'mark');span.textContent=match[0];fragments.push(span);start=match.index+match[0].length;}if(start<value.length)fragments.push(document.createTextNode(value.slice(start)));el.replaceChildren(...fragments);
}
async function getJSON(url){const response=await fetch(url,{headers:{Accept:'application/json'},cache:'no-store'});let body;try{body=await response.json();}catch{throw new Error(`The server returned an unreadable response (${response.status}).`);}if(!response.ok)throw new Error(typeof body.detail==='string'?body.detail:`Request failed (${response.status}).`);return body;}
function setError(error){$('connection-error').hidden=!error;$('connection-error-text').textContent=error?.message||'';}
function currentXY(index, frame=state.frame){const m=state.match;if(!m||m.available?.[frame]?.[index]===false)return null;const xy=m.positions?.[frame]?.[index];return validXY(xy)?xy:null;}
function frameAt(time){const ts=state.match.timestamps;let low=0,high=ts.length-1;while(low<high){const mid=Math.floor((low+high+1)/2);if(ts[mid]<=time+1e-7)low=mid;else high=mid-1;}return low;}
function selectedIndex(){return state.match?.players.findIndex(p=>p.id===state.selectedId)??-1;}
function isRevealed(){return !!state.prediction && state.frame>=state.prediction.target_frame;}
function predictionEntry(id){if(!state.prediction)return null;const p=state.prediction,i=p.player_ids.indexOf(id);return i<0?null:{current:p.current_xy[i],predicted:p.predicted_xy[i],index:i};}
function truthFor(id){if(!isRevealed())return null;const index=state.match.players.findIndex(p=>p.id===id);return index<0?null:currentXY(index,state.prediction.target_frame);}
function invalidatePending(){state.requestGeneration++;state.pending=false;}
function setPlaying(value){
  state.playing=!!value;state.clockOrigin=state.playing?performance.now():null;state.clockTime=state.time;
  if(!state.playing){state.mediaPlayAttempt=(state.mediaPlayAttempt||0)+1;state.audioStarting=false;state.audio?.pause();}
  else if(state.audioEnabled&&state.audio){
    const audio=state.audio,audioGeneration=state.audioGeneration,attempt=state.mediaPlayAttempt=(state.mediaPlayAttempt||0)+1;
    const elapsed=state.time-state.audioBase;
    if(elapsed>=0&&(!Number.isFinite(audio.duration)||elapsed<audio.duration)){
      if(audio.readyState>=1)audio.currentTime=elapsed;
      audio.playbackRate=state.speed;
      audio.play().catch(()=>{if(state.audio!==audio||state.audioGeneration!==audioGeneration||attempt!==state.mediaPlayAttempt)return;state.audioEnabled=false;state.voiceError='Sound is paused by your browser. Restart full replay or this highlight to listen.';renderVoice();});
    }
  }
  renderControls();renderVoice();
}
function moveTo(time,{manual=false,clearMoment=manual}={}){
  if(!state.match)return;
  if(manual){
    setPlaying(false);cancelReplayRun();state.voiceGeneration++;state.voicePending=false;state.stopAt=state.match.timestamps.at(-1);state.error=null;state.manualSeek=true;state.replayMode='manual';state.sessionStarted=true;
    if(clearMoment){invalidatePending();resetCommentary();state.activeHighlight=null;state.segment=null;state.prediction=null;state.voice=null;state.voiceError=null;state.overlayKey=null;renderHighlights();}
  }
  const ts=state.match.timestamps;state.time=Math.max(ts[0],Math.min(ts.at(-1),Number(time)));state.frame=frameAt(state.time);
  if(manual)prepareReplaySchedule(state.time);
  render();
  if(manual){renderControls();renderVoice();}
}
function selectPlayer(id,{explain=true}={}){
  if(!state.match)return;const player=state.match.players.find(p=>p.id===id);if(!player)return;
  const changed=state.selectedId!==id;if(changed)resetCommentary();state.selectedId=id;
  $('player-select').value=id;$('selected-team-dot').className=`selected-team-dot ${player.team===0?'home':'away'}`;
  if($('manual-player-select'))$('manual-player-select').value=id;
  render();if(changed&&explain&&state.prediction)explainMoment();
}
function inspectPlayer(id){
  if(state.highlightPending||!state.match?.players.some(player=>player.id===id))return;
  setPlaying(false);cancelReplayRun();invalidatePending();resetCommentary();invalidateVoice();state.replayMode='manual';state.sessionStarted=true;
  state.activeHighlight=null;state.segment=null;state.prediction=null;state.stopAt=null;state.manualSeek=true;state.emptyReason=null;state.error=null;
  selectPlayer(id,{explain:false});renderHighlights();render();
}
function activeMoment(){return state.highlights.find(h=>h.id===state.activeHighlight)||null;}
function clipFor(h){const end=state.match.timestamps.at(-1),anchor=h.time_s;return {start:Math.max(0,h.clip_start_s??anchor-3),end:Math.min(end,h.clip_end_s??anchor+7)};}
function predictionPhase(){return !state.prediction?state.pending?'preparing':'none':state.replayMode==='full'&&state.segment&&state.time>=state.segment.end?'past-highlight':state.time<state.prediction.anchor_time_s?'build-up':isRevealed()?'actual':'forecast';}
function renderPredictionStatus(){
  const target=$('prediction-status');if(!target)return;
  const p=state.prediction,phase=predictionPhase();target.dataset.phase=phase;
  target.textContent=phase==='past-highlight'?`Last highlight at ${clock(p.anchor_time_s,2)} · evidence remains anchored to that capture.`:phase==='preparing'?'Preparing the forecast · replay is available':phase==='build-up'?`Build-up → forecast at ${clock(p.anchor_time_s,2)} → actual at ${clock(p.target_time_s,2)}`:
    phase==='forecast'?`Prediction shown → actual remains hidden until ${clock(p.target_time_s,2)}`:phase==='actual'?`Actual revealed at ${clock(p.target_time_s,2)} · compare the ring and diamond`:
    activeMoment()?'This highlight is ready to replay.':state.replayMode==='full'?'Full replay · evidence appears automatically at each detected highlight.':'Choose a highlight, or capture your own prediction below.';
}
function renderFocusControls(){
  for(const button of $('focus-shortcuts')?.querySelectorAll('button[data-focus]')||[]){
    const active=(button.dataset.focus||null)===state.focusId;button.setAttribute('aria-pressed',String(active));button.disabled=!state.match||state.highlightPending;
  }
  if($('manual-player-select')){$('manual-player-select').disabled=!state.match||state.highlightPending;if(state.selectedId)$('manual-player-select').value=state.selectedId;}
  if($('focus-help'))$('focus-help').textContent=state.highlightPending?'Finding qualifying highlights for your focus…':state.emptyReason||
    (state.replayMode==='full'?`Full replay · automatically following ${state.focusId?playerName(state.focusId):'all players’ highlights'}.`:!activeMoment()&&state.match?`Inspecting ${playerName(state.selectedId)} · highlight focus unchanged.`:state.focusId?`Following ${playerName(state.focusId)}. Choose a highlight, then press Play.`:'Choose any player, then press Play. Highlights and commentary follow your choice.');
}
function selectedEvidence(){
  const p=state.prediction,entry=predictionEntry(state.selectedId);if(!p||!entry)return null;
  const opponents=p.player_ids.map((id,index)=>index).filter(index=>(index<11)!==(entry.index<11));
  const nearest=opponents.reduce((best,index)=>distance(entry.current,p.current_xy[index])<distance(entry.current,p.current_xy[best])?index:best,opponents[0]);
  const now=distance(entry.current,p.current_xy[nearest]);
  const future=Math.min(...opponents.map(index=>distance(entry.predicted,p.predicted_xy[index])));
  return {now,future,delta:future-now,nearest,entry};
}
function playerName(id){const p=state.match?.players.find(item=>item.id===id);return p?`${p.team===0?'Home':'Away'} ${String(p.number).padStart(2,'0')}`:id;}

function buildPlayers(){clear($('players-layer'));state.match.players.forEach((p,i)=>{const group=svgAdd($('players-layer'),'g',{'class':'player-group','data-index':i,'data-player-id':p.id});svgAdd(group,'title',{},`${p.team===0?'Home':'Away'} ${p.number} — ${p.name||p.id}`);svgAdd(group,'circle',{'class':'player-hit',r:3.6});svgAdd(group,'circle',{'class':'selection-ring',r:3.6});svgAdd(group,'circle',{'class':'player-marker',r:2.6,fill:COLORS[p.team]||COLORS[0]});svgAdd(group,'text',{'class':'player-label',y:.03},p.number);group.addEventListener('click',()=>focusPlayer(p.id));});}

function renderPitch(){
  const m=state.match;if(!m)return;renderFocus();
  for(const group of $('players-layer').children){const i=Number(group.dataset.index),xy=currentXY(i),selected=m.players[i].id===state.selectedId;group.style.display=xy?'':'none';if(xy)group.setAttribute('transform',`translate(${xy[0]} ${xy[1]})`);group.classList.toggle('is-selected',selected);group.querySelector('.selection-ring').style.display=selected?'':'none';group.querySelector('.selection-ring').setAttribute('r',3.6);group.querySelector('.player-marker').setAttribute('r',selected?3:2.6);group.querySelector('.player-label').style.fontSize=selected?'3.7px':'3.5px';group.style.opacity=selected||state.showAll?'1':'.86';}
  let ballMarker=$('ball-layer').firstElementChild;
  if(!ballMarker){ballMarker=svgAdd($('ball-layer'),'g',{'data-marker':'recorded-ball'});svgAdd(ballMarker,'title',{},'Ball · recorded position, not a prediction');svgAdd(ballMarker,'circle',{r:1.65,fill:'none',stroke:'#ff655c','stroke-width':.3});svgAdd(ballMarker,'circle',{r:1.1,fill:'#fff',stroke:'#183b30','stroke-width':.3});svgAdd(ballMarker,'circle',{r:.28,fill:'#466359'});svgAdd(ballMarker,'path',{d:'M0 -6.2V-2.6M-1.2 -3.8L0 -2.4L1.2 -3.8',fill:'none',stroke:'#ff544d','stroke-width':.65,'stroke-linecap':'round','stroke-linejoin':'round','data-marker':'ball-pointer'});}
  const ball=currentXY(22);ballMarker.style.display=ball?'':'none';if(ball)ballMarker.setAttribute('transform',`translate(${ball[0]} ${ball[1]})`);
  const p=state.prediction,forecastVisible=!!p&&state.time>=p.anchor_time_s&&!(state.replayMode==='full'&&state.segment&&state.time>=state.segment.end);
  const overlayKey=JSON.stringify([p?.anchor_frame,state.selectedId,state.audience,state.showAll,forecastVisible,isRevealed()]);
  if(overlayKey!==state.overlayKey){
    state.overlayKey=overlayKey;clear($('prediction-layer'));clear($('truth-layer'));
    if(forecastVisible)for(let i=0;i<p.player_ids.length;i++){
      const anchor=p.current_xy[i],pred=p.predicted_xy[i],selected=p.player_ids[i]===state.selectedId;
      if(!validXY(anchor)||!validXY(pred)||(!selected&&!state.showAll))continue;
      svgAdd($('prediction-layer'),'line',{x1:anchor[0],y1:anchor[1],x2:pred[0],y2:pred[1],'class':`forecast-line${selected?' forecast-guide':''}`,'stroke-width':selected?1.05:.18,'data-marker':'prediction-path'});
      if(selected)svgAdd($('prediction-layer'),'circle',{cx:anchor[0],cy:anchor[1],r:1.1,fill:'#d5ef85',stroke:'#173f35','stroke-width':.25,'data-marker':'forecast-origin'});
      svgAdd($('prediction-layer'),'circle',{cx:pred[0],cy:pred[1],r:selected?3.8:1.35,'class':`forecast-ring${selected?' forecast-selected':''}`,'data-marker':'forecast-endpoint'});
      if(selected){
        svgAdd($('prediction-layer'),'circle',{cx:pred[0],cy:pred[1],r:1.05,fill:'#8ff4ff',stroke:'#102f28','stroke-width':.3,'data-marker':'prediction-beacon'});
        svgAdd($('prediction-layer'),'text',{x:Math.max(3,Math.min(73,pred[0]+5)),y:Math.min(61,pred[1]+6.5),fill:'#aaf7ff','font-size':3.0,'font-weight':800,stroke:'#102f28','stroke-width':1.1,'paint-order':'stroke','class':'endpoint-label'},'AI PREDICTS +1s');
      }
      const truth=truthFor(p.player_ids[i]);if(truth){
        if(selected)svgAdd($('truth-layer'),'line',{x1:pred[0],y1:pred[1],x2:truth[0],y2:truth[1],'class':'error-line'});
        const radius=selected?4.5:.7;svgAdd($('truth-layer'),'path',{d:`M${truth[0]} ${truth[1]-radius}l${radius} ${radius}-${radius} ${radius}-${radius}-${radius}Z`,'class':`truth-diamond${selected?' truth-selected':''}`,opacity:selected?1:.68,'data-marker':'recorded-outcome'});
        if(selected)svgAdd($('truth-layer'),'text',{x:Math.max(3,Math.min(85,truth[0]+4)),y:Math.min(67,truth[1]+11),fill:'#ffda91','font-size':3.0,'font-weight':700,stroke:'#173f35','stroke-width':1.1,'paint-order':'stroke','class':'endpoint-label'},'REPLAY +1s');
      }
    }
  }
  const callout=$('forecast-callout'),entry=predictionEntry(state.selectedId),truth=truthFor(state.selectedId);
  if(callout){callout.hidden=!forecastVisible;callout.dataset.phase=isRevealed()?'actual':'forecast';
    $('forecast-callout-label').textContent=isRevealed()?'PREDICTION vs REPLAY · +1s':'AI PREDICTS · +1 SECOND';
    $('forecast-callout-value').textContent=isRevealed()&&truth&&entry?(state.audience==='fan'?'Compare cyan & gold':`${fmt(distance(entry.predicted,truth),2)} m apart`):`${playerName(state.selectedId)} → cyan target`;
  }
  if(p)$('pitch-state').textContent=state.time<p.anchor_time_s?`1 · BUILD-UP — forecast appears at ${clock(p.anchor_time_s,2)}`:isRevealed()?`3 · ACTUAL REVEALED — diamond at ${clock(p.target_time_s,2)} · replay continues`:`2 · PREDICTION — ring is the estimate; actual still hidden`;
  else $('pitch-state').textContent='RETROSPECTIVE REPLAY · choose a detected highlight';
  if(state.replayMode==='highlight'&&state.segment&&state.time>=state.segment.end-.001)$('pitch-state').textContent='HIGHLIGHT COMPLETE · replay this highlight or play all 40 seconds';
  if(state.replayMode==='full'&&!p)$('pitch-state').textContent='FULL REPLAY · highlights and prepared commentary appear at their event time';
  if(state.replayMode==='full'&&p&&!forecastVisible&&state.time>=p.anchor_time_s)$('pitch-state').textContent=`FULL REPLAY · last highlight at ${clock(p.anchor_time_s,2)}; its evidence remains on the right`;
  if(state.sessionStarted&&state.time>=m.timestamps.at(-1))$('pitch-state').textContent='FULL REPLAY COMPLETE · play again or choose a highlight';
}


function renderFocus(){
  clear($('focus-layer'));clear($('annotation-layer'));
  const index=selectedIndex(),xy=currentXY(index);if(!xy)return;
  const root=$('annotation-layer'),e=selectedEvidence();
  svgAdd($('focus-layer'),'circle',{cx:xy[0],cy:xy[1],r:5,fill:COLORS[index<11?0:1],opacity:.13});
  if(state.audience==='analyst'&&e&&state.time>=state.prediction.anchor_time_s&&state.time<state.prediction.target_time_s){
    const other=state.prediction.current_xy[e.nearest],measured=state.prediction.current_xy[e.entry.index];
    svgAdd($('focus-layer'),'line',{x1:measured[0],y1:measured[1],x2:other[0],y2:other[1],stroke:'#d7eab2','stroke-width':.27,'stroke-dasharray':'.7 .65'});
    const mx=(measured[0]+other[0])/2,my=(measured[1]+other[1])/2;
    svgAdd(root,'rect',{x:mx-4,y:my-2.4,width:8,height:3.2,rx:.7,fill:'#102f28'});
    svgAdd(root,'text',{x:mx,y:my-.3,'text-anchor':'middle',fill:'#ecf6db','font-size':1.7},`${fmt(e.now,1)} m`);
  }
  const x=Math.min(93,Math.max(12,xy[0])),y=xy[1]<9?xy[1]+7:xy[1]-6;
  svgAdd(root,'line',{x1:x,y1:y+1.5,x2:xy[0],y2:xy[1]-2.3,stroke:'#edf5dc','stroke-width':.2});
  svgAdd(root,'rect',{x:x-12.5,y:y-3,width:25,height:6.2,rx:1,fill:'#e9f3d1'});
  svgAdd(root,'text',{x,y:y+1,'text-anchor':'middle',fill:'#153c31','font-size':3,'font-weight':700},`${playerName(state.selectedId)} ↗`);
}

function renderDetail(){const root=$('detail-layer');clear(root);if(!state.match)return;const entry=predictionEntry(state.selectedId),index=selectedIndex(),anchor=entry?.current||currentXY(index),pred=entry?.predicted,truth=truthFor(state.selectedId);if(!validXY(anchor)){svgAdd(root,'text',{x:80,y:54,'text-anchor':'middle',fill:'#98a28d','font-size':8},'Position unavailable');return;}
  const maxOffset=Math.max(0,...[pred,truth].filter(validXY).map(p=>Math.max(Math.abs(p[0]-anchor[0]),Math.abs(p[1]-anchor[1]))));const span=Math.max(12,Math.ceil((maxOffset*2.8)/2)*2);$('detail-scale').textContent=`${span} m`;
  const scale=160/span,point=xy=>[80+(xy[0]-anchor[0])*scale,50+(xy[1]-anchor[1])*scale];
  const home=state.match.players[index]?.team===0,color=home?'#819e39':'#9c70ad';
  svgAdd(root,'line',{x1:80,y1:8,x2:80,y2:94,stroke:'#c9d7c4','stroke-width':.7,'stroke-dasharray':'2 3'});svgAdd(root,'line',{x1:16,y1:50,x2:144,y2:50,stroke:'#c9d7c4','stroke-width':.7,'stroke-dasharray':'2 3'});
  if(validXY(pred)){const [px,py]=point(pred);svgAdd(root,'line',{x1:80,y1:50,x2:px,y2:py,stroke:'#819773','stroke-width':1.1,'stroke-dasharray':'3 3'});svgAdd(root,'circle',{cx:px,cy:py,r:5.5,fill:'none',stroke:'#406b47','stroke-width':1.6});svgAdd(root,'text',{x:px+8,y:py-7,fill:'#64845b','font-size':5.5,'font-family':'monospace'},'PREDICTED');if(truth){const [tx,ty]=point(truth);svgAdd(root,'line',{x1:px,y1:py,x2:tx,y2:ty,stroke:'#976e36','stroke-width':1.4});svgAdd(root,'path',{d:`M${tx} ${ty-4}l4 4-4 4-4-4Z`,fill:'#fff',stroke:'#9a783d','stroke-width':1.2});svgAdd(root,'text',{x:tx+7,y:ty+10,fill:'#927445','font-size':5.5,'font-family':'monospace'},'ACTUAL');}}
  svgAdd(root,'circle',{cx:80,cy:50,r:4.6,fill:color,stroke:'#fff','stroke-width':1.3});svgAdd(root,'text',{x:72,y:66,fill:'#809073','font-size':5.5,'font-family':'monospace'},entry?'AT CAPTURE':'NOW');
}

function renderMetrics(){let selectedError=null,meanError=null,count=0;if(isRevealed()){const errors=[];for(const id of state.prediction.player_ids){const pred=predictionEntry(id)?.predicted,truth=truthFor(id);if(validXY(pred)&&validXY(truth)){const error=distance(pred,truth);errors.push(error);if(id===state.selectedId)selectedError=error;}}count=errors.length;if(count)meanError=errors.reduce((a,b)=>a+b,0)/count;}
  $('selected-error').replaceChildren(document.createTextNode(fmt(selectedError)),Object.assign(document.createElement('small'),{textContent:' m'}));$('mean-error').replaceChildren(document.createTextNode(fmt(meanError)),Object.assign(document.createElement('small'),{textContent:' m'}));
  $('result-state').textContent=isRevealed()?'OUTCOME REVEALED':state.prediction?'AWAITING +1s':'NOT YET CAPTURED';$('result-state').className=`result-state${isRevealed()?' revealed':''}`;
  $('metric-footnote').textContent=isRevealed()?`Euclidean distance at ${clock(state.prediction.target_time_s,2)} · ${count} valid players · synthetic replay.`:'Errors appear when the replay reaches the forecast time.';
}

function renderInsight(){
  const e=selectedEvidence(),h=activeMoment(),focused=h&&h.player_id===state.selectedId;
  document.body.classList.toggle('fan-mode',state.audience==='fan');
  const observed=focused?h.evidence.observed_gap_change_m:null;
  const stages=state.commentary?.stages||{};
  $('moment-kicker').textContent=focused?'RETROSPECTIVE HIGHLIGHT':state.prediction?'YOUR PREDICTION':state.replayMode==='full'?'FULL REPLAY':'MANUAL INSPECTION';
  $('moment-time').textContent=h?clock(h.time_s,2):state.prediction?clock(state.prediction.anchor_time_s,2):'—';
  $('moment-title').textContent=focused?h.headline:state.highlightPending?'Finding your highlights…':state.emptyReason?'No related highlights':e?'Follow the movement':state.replayMode==='full'?'Full replay · highlights come next':'Choose a highlight to explore';
  $('moment-summary').textContent=focused?(state.audience==='fan'?`${playerName(h.player_id)} · follow the movement`:`${playerName(h.player_id)} · captured at ${clock(h.time_s,2)}`):state.emptyReason||(state.replayMode==='full'?'The entire replay runs continuously. Detected highlights appear at their event time.':'Choose a detected highlight, or inspect any point in the replay.');
  $('stage-observed').textContent=Number.isFinite(observed)?(state.audience==='fan'?(h.kind==='movement'?'A run worth watching.':observed>=0?'Space just opened up.':'The defender is getting closer.'):`${observed>=0?'Space opened':'Space tightened'} by ${fmt(Math.abs(observed),1)} m.`):'No observed highlight at this point.';
  const observedRow=$('stage-observed').closest('[data-stage="observed"]');
  if(observedRow)observedRow.hidden=!focused;
  $('stage-current').textContent=e?`${fmt(e.now,1)} m to the nearest opponent.`:'Select a moment to see the evidence.';
  $('stage-forecast').textContent=e?(state.audience==='fan'?(Math.abs(e.delta)<.1?'The model expects a similar gap.':e.delta>=0?'The model expects more room ahead.':'The model expects that gap to tighten.'):`${fmt(e.future,1)} m predicted · ${Math.abs(e.delta)<.1?'similar gap':e.delta>=0?'more room':'less room'}.`):'The next-second estimate appears here.';
  if(state.commentary?.limitation)$('evidence-limitations').textContent=state.commentary.limitation;
  for(const [id,value]of [['space-now',e?.now],['space-future',e?.future]])$(id).replaceChildren(document.createTextNode(fmt(value,1)),Object.assign(document.createElement('small'),{textContent:' m'}));
  $('space-delta').textContent=e?`${e.delta>=0?'+':''}${fmt(e.delta,1)} m`:'—';
  $('insight-headline').textContent=$('moment-title').textContent;$('mobile-story-title').textContent=focused?`${playerName(state.selectedId)} · ${h.headline}`:e?'Follow the movement':$('moment-title').textContent;
  $('mobile-story-text').textContent=focused?(state.audience==='fan'?`${playerName(h.player_id)} · ${h.headline.toLowerCase()}`:`${playerName(h.player_id)} · past-second gap ${observed>=0?'widened':'narrowed'} ${fmt(Math.abs(observed),1)} m.`):state.emptyReason||
    (e?`${playerName(state.selectedId)} · captured at ${clock(state.prediction.anchor_time_s,2)}; compare the next second.`:state.pending?'Preparing this point’s prediction…':state.replayMode==='full'?'Highlights and their evidence appear automatically as the replay continues.':'Choose a highlight, or capture your own prediction below.');
  $('insight-explanation').textContent=e?`Captured at ${clock(state.prediction.anchor_time_s,2)}. A +1 second estimate; the nearest opponent may change.`:'A trained forecast loads independently of the replay.';
  for(const id of ['moment-summary','stage-observed','stage-current','stage-forecast','mobile-story-title','mobile-story-text'])emphasizeText($(id),$(id).textContent);
}

function renderWorkflow(){
  const root=$('workflow-steps');if(!root)return;
  const h=activeMoment(),p=state.prediction,r=state.commentary,v=state.voice;
  const voice=v&&v.player_id===state.selectedId&&v.highlight_id===h?.id?v:null;
  const key=JSON.stringify([h?.id,p?.anchor_frame,state.selectedId,state.audience,r?.agent_trace,r?.mode,r?.cached,r?.reason,state.commentaryPending,voice?.cache_key,voice?.status,voice?.cached,state.playing,state.commentaryOn,state.captions,state.muted,state.audio?.paused,!!state.audio,isRevealed()]);
  if(state.workflowRenderKey===key)return;state.workflowRenderKey=key;
  const agents=r?.agent_trace,review=agents?.reviewer;
  $('agent-analyst-status').textContent=state.commentaryPending?'Working…':agents?.analyst?.status==='completed'?(agents.analyst.cached?'Complete · cached':'Complete'):r?'Local fallback':'Preparing…';
  $('agent-reviewer-status').textContent=state.commentaryPending?'Waiting…':review?.status==='passed'?(review.cached?'Passed · cached':'Passed'):review?.status==='rejected'?'Flagged':review?.status==='unavailable'?'Unavailable':'Not run';
  $('agent-team-note').textContent=review?.status==='passed'?'Checked: player, facts, timing and forecast wording.':state.commentaryPending?'Evidence selection → independent commentary review.':'Local evidence stays available when cloud review is not complete.';
  document.querySelector('.agent-team').dataset.verified=String(review?.status==='passed');
  const steps=[
    ['Highlight discovery',h?`${playerName(h.player_id)} · ${clock(h.time_s,2)} · ranked from observed movement`:'Manual capture · no discovery claim'],
    ['Movement & space check',p?`Frozen model · ${clock(p.anchor_time_s,2)} → ${clock(p.target_time_s,2)} · +1s`:'Waiting for the prediction'],
    ['Commentary preparation',r?.mode==='azure'?`Microsoft selected verified facts${r.cached?' · saved result':''}`:state.commentaryPending?'Microsoft explanation pending · local evidence ready':'Local evidence summary',voice?.status==='ready'?`Speech ready${voice.cached?' · cached audio':''}`:'Captions available · speech not ready'],
    ['Synchronized playback',state.playing?`Playing · ${state.audio&&!state.audio.paused?'audio + picture':state.captions?'picture + captions':'picture only'}`:'Ready / paused',isRevealed()?'Recorded +1s outcome revealed':'Recorded +1s outcome remains hidden']
  ];
  clear(root);for(const [title,...lines] of steps){const li=document.createElement('li');li.append(Object.assign(document.createElement('strong'),{textContent:title}),Object.assign(document.createElement('span'),{textContent:lines.join(' · ')}));root.append(li);}
  $('workflow-json').textContent=JSON.stringify({workflow:'Two LLM roles with a fixed handoff; not autonomous tool orchestration',agents:r?.agent_trace||null,match_id:state.match?.id,highlight_id:h?.id||null,player_id:state.selectedId,anchor_frame:p?.anchor_frame,target_frame:p?.target_frame,model:p?.model?.name,checkpoint:p?.model?.checkpoint_sha256,explanation:{mode:r?.mode||'pending',cached:!!r?.cached,reason:r?.reason||state.commentaryError||null,fact_ids:r?.fact_ids||[]},speech:{status:voice?.status||'not_ready',cached:!!voice?.cached},guards:'Server validates evidence; cloud timeout, request limits and successful-response cache are enforced.'},null,2);
}

function renderControls(){
  const m=state.match,end=!!m&&state.frame>=m.timestamps.length-1;
  const boundary=!!m&&(state.frame<98||state.frame+25>=m.timestamps.length),canPredict=m&&state.health?.model_loaded&&!boundary;
  const clipEnded=!state.manualSeek&&!!state.segment&&state.time>=state.segment.end-.001;
  $('play-button').disabled=!m||!state.sessionStarted||state.time>=(state.replayMode==='highlight'&&state.segment?state.segment.end:m.timestamps.at(-1));$('step-button').disabled=!m||end;$('timeline').disabled=!m||state.highlightPending;$('player-select').disabled=!m||state.highlightPending;$('focus-select').disabled=!m||state.highlightPending;
  $('predict-button').disabled=(!canPredict&&!boundary)||state.pending;
  $('predict-label').textContent=state.pending?'Running trained model…':boundary?'Replay a highlight':'Predict this point';
  $('play-icon').textContent=state.playing?'Ⅱ':'▶';$('play-label').textContent=state.playing?'Pause':'Resume';
  $('play-button').setAttribute('aria-label',state.playing?'Pause replay':'Resume replay');
  $('next-highlight').disabled=!state.highlights.length;
  $('predict-help').className=`predict-help${state.error?' error':''}`;
  $('predict-help').textContent=state.error||(!m?'Loading replay…':boundary?'No complete prediction window here. Choose a highlight to continue.':state.pending?'Playback remains available while the model runs.':state.prediction?`CPU forward: ${fmt(state.prediction.model?.inference_ms,2)} ms · frozen model`:'Captures only preceding history.');
  $('outcome-button').hidden=true;
  $('capture-marker').hidden=!state.prediction;
  $('capture-time').textContent=state.replayMode==='full'?`FULL REPLAY · 00:00–${clock(m?.timestamps.at(-1)||40)}`:state.segment?`HIGHLIGHT ${clock(state.segment.start)}–${clock(state.segment.end)}`:state.prediction?`${clock(state.prediction.anchor_time_s,2)} → ${clock(state.prediction.target_time_s,2)}`:'Choose a highlight';
  if(state.prediction)$('capture-marker').style.left=`${state.prediction.target_time_s/(m?.timestamps.at(-1)||40)*100}%`;
  renderClock();renderPredictionStatus();renderFocusControls();
}
function renderClock(){
  const m=state.match,duration=m?.timestamps.at(-1)||40;
  $('pitch-clock').textContent=clock(state.time,2);$('timeline').value=String(state.time);$('timeline').style.setProperty('--progress',`${state.time/duration*100}%`);$('timeline').setAttribute('aria-valuetext',clock(state.time,2));
  $('frame-label').textContent=m?`Frame ${state.frame} / ${m.timestamps.length-1}`:'Frame — / —';
}
function render(){
  renderPitch();renderClock();renderSubtitles();renderWorkflow();
  const voicePhase=state.playing&&state.commentaryOn?Math.max(0,Math.ceil((activeMoment()?.time_s??Infinity)-state.time)):null;
  if(voicePhase!==state.renderedVoicePhase){state.renderedVoicePhase=voicePhase;renderVoice();}
  const key=JSON.stringify([state.match?.id,state.selectedId,state.focusId,state.prediction?.anchor_frame,state.prediction?.model?.inference_ms,predictionPhase(),state.pending,state.playing,state.manualSeek,state.replayMode,state.sessionStarted,state.activeHighlight,state.highlightPending,state.error,state.emptyReason,state.commentary?.text,state.audience,state.time>=(state.segment?.end??Infinity),state.frame<98,state.frame+25>=(state.match?.timestamps.length??Infinity)]);
  if(key===state.semanticRenderKey)return;state.semanticRenderKey=key;
  if($('inspector').open)renderDetail();renderMetrics();renderInsight();renderControls();renderCommentary();renderVoice();
}

function renderHighlights(){
  clear($('highlight-list'));clear($('highlight-markers'));
  $('highlight-status').textContent=state.highlightPending?'Scanning movement…':`${state.highlights.length} highlights · ${state.focusId?playerName(state.focusId):'all players'}`;
  if(!state.highlights.length){$('highlight-list').append(Object.assign(document.createElement('p'),{className:'highlight-empty',textContent:state.highlightPending?'Finding changes in space and movement…':state.emptyReason||'No highlights available. Use the timeline and manual prediction below.'}));return;}
  for(const h of state.highlights){
    const button=document.createElement('button');button.type='button';button.className='highlight-card';button.classList.toggle('active',h.id===state.activeHighlight);button.setAttribute('aria-pressed',String(h.id===state.activeHighlight));
    const time=Object.assign(document.createElement('span'),{textContent:clock(h.time_s)}),body=document.createElement('div');
    body.append(Object.assign(document.createElement('strong'),{textContent:h.headline}),Object.assign(document.createElement('small'),{textContent:state.audience==='fan'?`${playerName(h.player_id)} · follow this moment`:`${playerName(h.player_id)} · ${fmt(Math.abs(h.evidence.observed_gap_change_m),1)} m gap change`}));
    button.append(time,body,Object.assign(document.createElement('span'),{className:'rank',textContent:String(h.rank).padStart(2,'0')}));button.addEventListener('click',()=>chooseHighlight(h.id));$('highlight-list').append(button);
    const marker=document.createElement('button');marker.type='button';marker.className=`highlight-marker${h.id===state.activeHighlight?' active':''}`;marker.textContent='◆';marker.style.left=`${h.time_s/state.match.timestamps.at(-1)*100}%`;marker.title=`${clock(h.time_s,2)} · ${h.headline}`;marker.setAttribute('aria-label',marker.title);marker.addEventListener('click',()=>chooseHighlight(h.id));$('highlight-markers').append(marker);
  }
}
async function chooseHighlight(id,{play=false}={}){
  const h=state.highlights.find(item=>item.id===id);if(!h||!state.match)return;
  cancelReplayRun();invalidatePending();resetCommentary();invalidateVoice();setPlaying(false);state.replayMode='highlight';state.sessionStarted=false;state.activeHighlight=id;state.prediction=null;state.overlayKey=null;state.segment=clipFor(h);state.stopAt=null;state.error=null;state.manualSeek=false;
  state.selectedId=state.focusId||h.player_id;selectPlayer(state.selectedId,{explain:false});moveTo(state.segment.start);renderHighlights();
  // Position inference and speech preparation never own the playback clock.
  const predictionTask=predict({anchor:h.anchor_frame,automatic:true});
  preloadNextMoment(h,{voice:false});
  const voiceGeneration=state.voiceGeneration;
  prepareActiveVoice().finally(()=>{if(voiceGeneration===state.voiceGeneration)preloadNextMoment(h);});if(play)playHighlight();
  await predictionTask;
}
function cancelReplayRun(){state.runGeneration++;state.runSchedule=[];state.voiceSkipped=null;stopAudio();}
function playbackLimit(){
  if(state.stepMode&&state.prediction&&state.time<state.prediction.target_time_s)return state.prediction.target_time_s;
  return state.replayMode==='highlight'&&state.segment?state.segment.end:state.match?.timestamps.at(-1);
}
function prepareReplaySchedule(from=-Infinity,{only=null}={}){
  const generation=state.runGeneration;
  state.runSchedule=(only?[only]:state.highlights.filter(h=>h.time_s>from+.001)).slice().sort((a,b)=>a.time_s-b.time_s).map(h=>({h,attempted:false,voice:voiceCache.get(voiceKey(h,state.focusId||h.player_id))||null}));
  for(const entry of state.runSchedule){
    predictionFor(state.match.id,entry.h.anchor_frame).catch(()=>{});
    if(entry.voice?.status==='ready')bufferVoice(entry.voice);
    // Full-replay prefetch reads existing cache only; it cannot spend generation budget.
    prepareVoice(entry.h,state.focusId||entry.h.player_id,{cacheOnly:true}).then(result=>{if(generation!==state.runGeneration)return;entry.voice=result;if(result?.status==='ready')bufferVoice(result);primeUpcomingAudio();}).catch(()=>{});
  }
}
function playFullReplay(){
  if(!state.match)return;
  setPlaying(false);cancelReplayRun();invalidatePending();resetCommentary();invalidateVoice();
  state.replayMode='full';state.sessionStarted=true;state.activeHighlight=null;state.segment=null;state.prediction=null;state.overlayKey=null;state.manualSeek=false;
  state.time=state.match.timestamps[0];state.frame=0;state.stopAt=state.match.timestamps.at(-1);prepareReplaySchedule();unlockAudioOutput();renderHighlights();render();setPlaying(true);bringReplayIntoView();
}
function playHighlight(){
  if(!state.match)return;const h=activeMoment()||state.highlights[0];if(!h)return;
  if(h.id!==state.activeHighlight)chooseHighlight(h.id);
  setPlaying(false);cancelReplayRun();state.replayMode='highlight';state.sessionStarted=true;state.manualSeek=false;state.voiceSkipped=null;
  const clip=state.segment=clipFor(h);state.time=clip.start;state.frame=frameAt(state.time);state.stopAt=playbackLimit();prepareReplaySchedule(-Infinity,{only:h});
  // Explicit highlight intent keeps the existing rate-limited preparation path.
  prepareActiveVoice();unlockAudioOutput();render();setPlaying(true);bringReplayIntoView();
}
function playMoment(){
  if(!state.match)return;
  if(state.playing){setPlaying(false);return;}
  if(!state.sessionStarted){playFullReplay();return;}
  state.stopAt=playbackLimit();if(state.time>=state.stopAt)return;
  setPlaying(true);
}
async function focusPlayer(id){
  if(!state.match)return;const playerId=id||null;if(playerId&&!state.match.players.some(p=>p.id===playerId))return;
  const generation=++state.focusGeneration,matchId=state.match.id;
  setPlaying(false);cancelReplayRun();invalidatePending();resetCommentary();invalidateVoice();state.sessionStarted=false;state.focusId=playerId;state.prediction=null;state.highlights=[];state.activeHighlight=null;state.segment=null;state.emptyReason=null;state.highlightPending=true;
  if(playerId)state.selectedId=playerId;$('focus-select').value=playerId||'';renderHighlights();render();
  try{
    const discovery=await getJSON(`/api/highlights?match_id=${encodeURIComponent(matchId)}${playerId?`&player_id=${encodeURIComponent(playerId)}`:''}`);
    if(generation!==state.focusGeneration||matchId!==state.match?.id)return;
    state.highlights=discovery.highlights||[];state.emptyReason=discovery.empty_reason||null;state.highlightPending=false;$('discovery-method').textContent=discovery.method||'';renderHighlights();
    if(state.sessionStarted&&state.replayMode==='full'){prepareReplaySchedule(state.time);render();}
    else if(state.highlights.length)await chooseHighlight(state.highlights[0].id);else{state.emptyReason=state.emptyReason||`No qualifying highlights were found for ${playerName(playerId)} in this replay.`;render();}
  }catch(error){if(generation!==state.focusGeneration)return;state.highlightPending=false;state.emptyReason='Highlights could not be loaded. The recorded replay is still available.';state.error=error.message;renderHighlights();render();}
}
async function loadMatch(id){
  const generation=++state.matchGeneration;state.focusGeneration++;cancelReplayRun();invalidatePending();resetCommentary();invalidateVoice();setPlaying(false);state.sessionStarted=false;state.match=null;state.prediction=null;state.error=null;state.highlights=[];state.activeHighlight=null;state.segment=null;state.stopAt=null;state.highlightPending=true;state.emptyReason=null;state.overlayKey=null;state.semanticRenderKey=null;
  for(const layer of ['players-layer','ball-layer','prediction-layer','truth-layer','focus-layer','annotation-layer','detail-layer'])clear($(layer));
  render();renderHighlights();$('pitch-state').textContent='Loading replay…';$('scenario-subtitle').textContent='Loading replay…';
  for(const button of $('scenario-tabs').children){button.classList.toggle('active',button.dataset.matchId===id);button.setAttribute('aria-pressed',String(button.dataset.matchId===id));}
  try{
    const [m,discovery]=await Promise.all([getJSON(`/api/matches/${encodeURIComponent(id)}`),getJSON(`/api/highlights?match_id=${encodeURIComponent(id)}${state.focusId?`&player_id=${encodeURIComponent(state.focusId)}`:''}`).catch(()=>null)]);
    if(generation!==state.matchGeneration)return;
    if(!Array.isArray(m.players)||m.players.length!==22||!Array.isArray(m.timestamps)||!m.timestamps.length||m.positions?.length!==m.timestamps.length)throw new Error('Replay data has an invalid player or timestamp mapping.');
    state.match=m;if(state.focusId&&!m.players.some(p=>p.id===state.focusId))state.focusId=null;
    state.selectedId=state.focusId||m.players[8].id;state.highlights=discovery?.highlights||[];state.highlightPending=false;state.emptyReason=discovery?.empty_reason||null;
    $('discovery-method').textContent=discovery?.method||'Highlight discovery is unavailable. Manual prediction is still available.';
    $('scenario-subtitle').textContent=SCENES[m.id]?.title||m.title;
    if($('scene-guidance'))$('scene-guidance').textContent=SCENES[m.id]?.guidance||m.title;
    clear($('player-select'));clear($('focus-select'));if($('manual-player-select'))clear($('manual-player-select'));
    const all=document.createElement('option');all.value='';all.textContent='All players';$('focus-select').append(all);
    m.players.forEach(p=>{for(const selectId of ['player-select','focus-select','manual-player-select']){if(!$(selectId))continue;const o=document.createElement('option');o.value=p.id;o.textContent=`${p.team===0?'Home':'Away'} ${String(p.number).padStart(2,'0')}`;$(selectId).append(o);}});
    $('focus-select').value=state.focusId||'';
    $('timeline').max=String(m.timestamps.at(-1));$('timeline').min=String(m.timestamps[0]);$('timeline').step=String(1/m.fps);clear($('timeline-ticks'));
    for(let i=0;i<5;i++)$('timeline-ticks').append(Object.assign(document.createElement('span'),{textContent:clock(m.timestamps.at(-1)*i/4)}));
    buildPlayers();setError(null);renderHighlights();
    if(state.highlights.length)await chooseHighlight(state.highlights[0].id);else{moveTo(5);selectPlayer(state.selectedId,{explain:false});if(!state.focusId)await predict({automatic:true});}
  }catch(error){if(generation!==state.matchGeneration)return;state.highlightPending=false;setError(error);$('scenario-subtitle').textContent='Replay unavailable';$('pitch-state').textContent='Replay unavailable · use Try again above';renderControls();renderHighlights();}
}

function validatePrediction(p,matchId,anchor){if(p.match_id!==matchId||p.anchor_frame!==anchor||p.target_frame!==anchor+25||!Array.isArray(p.player_ids)||p.player_ids.length!==22||new Set(p.player_ids).size!==22||p.player_ids.some(id=>!state.match.players.some(player=>player.id===id))||p.current_xy?.length!==22||p.predicted_xy?.length!==22||!p.current_xy.every(validXY)||!p.predicted_xy.every(validXY))throw new Error('Prediction could not be aligned with the replay. No result was displayed.');const ts=state.match.timestamps;if(Math.abs(p.anchor_time_s-ts[anchor])>1e-6||Math.abs(p.target_time_s-ts[p.target_frame])>1e-6||Math.abs(p.target_time_s-p.anchor_time_s-1)>1e-6)throw new Error('Prediction timestamps do not match the exact +1 second target.');}
function predictionKey(matchId,anchor){return `${matchId}:${anchor}`;}
async function predictionFor(matchId,anchor){
  const key=predictionKey(matchId,anchor);if(predictionCache.has(key))return predictionCache.get(key);
  const promise=getJSON(`/api/predict?match_id=${encodeURIComponent(matchId)}&anchor_frame=${anchor}`).catch(error=>{predictionCache.delete(key);throw error;});
  predictionCache.set(key,promise);return promise;
}
async function predict({anchor=state.frame,automatic=false}={}){
  if(!state.match||state.pending)return;
  if(anchor<98||anchor+25>=state.match.timestamps.length){if(!automatic){if(state.highlights[0])return chooseHighlight(state.activeHighlight||state.highlights[0].id);return focusPlayer(null);}return;}
  if(!automatic){setPlaying(false);cancelReplayRun();invalidateVoice();state.replayMode='manual';state.sessionStarted=true;state.activeHighlight=null;state.segment=null;state.manualSeek=true;state.emptyReason=null;renderHighlights();}
  resetCommentary();const generation=++state.requestGeneration,matchId=state.match.id;state.pending=true;state.error=null;state.prediction=null;render();
  try{const p=await predictionFor(matchId,anchor);if(generation!==state.requestGeneration||state.match?.id!==matchId)return;validatePrediction(p,matchId,anchor);state.prediction=p;state.pending=false;if(state.playing)state.stopAt=playbackLimit();render();explainMoment();}
  catch(error){if(generation!==state.requestGeneration)return;state.pending=false;state.error=error.message;render();}
}
function preloadNextMoment(h,{voice=true}={}){
  const index=state.highlights.findIndex(item=>item.id===h.id),next=state.highlights[(index+1)%state.highlights.length];
  if(!next||next.id===h.id)return;
  predictionFor(state.match.id,next.anchor_frame).catch(()=>{});
  // Only one voice is prepared ahead, inside the server's existing limits.
  if(voice)prepareVoice(next,state.focusId||next.player_id,{cacheOnly:true}).catch(()=>{});
}

function resetCommentary(){state.commentaryGeneration++;state.commentaryController?.abort();state.commentaryController=null;state.commentaryPending=false;state.commentary=null;state.commentaryError=null;}
function localExplanation(){
  const e=selectedEvidence();if(!e)return 'Select a highlight to see its movement evidence.';
  if(state.audience==='analyst')return `${playerName(state.selectedId)}: nearest-opponent distance ${fmt(e.now,2)} → ${fmt(e.future,2)} m (Δ ${e.delta>=0?'+':''}${fmt(e.delta,2)} m). Forecast displacement: ${fmt(distance(e.entry.current,e.entry.predicted),2)} m. Horizon: +1.00 s.`;
  const h=activeMoment(),past=h?.player_id===state.selectedId?(h.kind==='movement'?'is making a run':h.evidence.observed_gap_change_m>=0?'just found more room':'is being closed down'):'is on the move';
  return `${playerName(state.selectedId)} ${past}. The model expects ${Math.abs(e.delta)<.1?'a similar gap':e.delta>=0?'more room':'a tighter gap'} next second.`;
}
function renderCommentary(){
  const p=state.prediction,result=state.commentary,configured=!!state.health?.services?.azure_openai?.configured;
  const key=JSON.stringify([p?.anchor_frame,p?.match_id,state.selectedId,state.audience,state.pending,state.commentaryPending,configured,result?.mode,result?.provider,result?.text,result?.cached,state.commentaryError]);
  renderWorkflow();if(key===state.commentaryRenderKey)return;state.commentaryRenderKey=key;
  $('explain-button').disabled=!p||state.pending||state.commentaryPending;
  $('explain-button').textContent=state.commentaryPending?'Explaining…':result?.mode==='azure'?'Refresh ↗':'Ask Microsoft AI ↗';
  $('explain-button').setAttribute('aria-busy',String(state.commentaryPending));
  for(const audience of ['fan','analyst'])$(`audience-${audience}`).setAttribute('aria-pressed',String(state.audience===audience));
  $('commentary-context').textContent=p?`${playerName(state.selectedId)} · ${clock(p.anchor_time_s,2)} → ${clock(p.target_time_s,2)}`:'Evidence follows the selected player.';
  const azure=result?.mode==='azure';
  $('commentary-source').textContent=azure?(result?.agent_trace?.reviewer?.status==='passed'?'Microsoft AI · reviewed':'Microsoft AI · evidence selected'):state.commentaryPending&&configured?'Evidence summary · Microsoft AI loading':result&&configured?'Local evidence · cloud unavailable':'Local evidence summary';
  $('commentary-source').classList.toggle('connected',azure);
  $('commentary-text').textContent=result?.stages?[result.stages.observed,result.stages.current,result.stages.forecast].filter(value=>typeof value==='string'&&value.trim()).join(' '):result?.text||localExplanation();
  renderSpokenPreview();renderWorkflow();
  $('commentary-cache').hidden=!result?.cached||state.commentaryPending;
  $('commentary-output').classList.toggle('loading',state.commentaryPending);
  $('commentary-help').classList.toggle('error',!!state.commentaryError);
  $('commentary-help').textContent=state.commentaryError||(azure?'AI selects verified facts from observed history and the model forecast.':state.commentaryPending?'Measured evidence is ready while the explanation loads.':'Calculated from geometry; this text is not generated by Microsoft AI.');
}

async function explainMoment(){
  if(!state.prediction||state.commentaryPending)return;
  const prediction=state.prediction, matchId=state.match.id, playerId=state.selectedId, audience=state.audience;
  const generation=++state.commentaryGeneration, controller=new AbortController();
  state.commentaryController=controller;state.commentaryPending=true;state.commentaryError=null;state.commentary=null;renderCommentary();
  const stillCurrent=()=>generation===state.commentaryGeneration&&prediction===state.prediction&&matchId===state.match?.id&&playerId===state.selectedId&&audience===state.audience;
  const timeout=setTimeout(()=>controller.abort(),30000);
  try{
    const response=await fetch('/api/insights',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},body:JSON.stringify({match_id:matchId,anchor_frame:prediction.anchor_frame,player_id:playerId,audience,language:'en',...(activeMoment()?.player_id===playerId?{highlight_id:state.activeHighlight}:{})}),signal:controller.signal});
    if(!response.ok)throw new Error('Explanation unavailable');
    const result=await response.json();
    if(!stillCurrent())return;
    if(typeof result.text!=='string'||!result.text.trim()||!['azure','local'].includes(result.mode)||typeof result.provider!=='string'||!result.provider.trim())throw new Error('Invalid explanation');
    state.commentary=result;renderInsight();
  }catch(error){if(!stillCurrent())return;state.commentaryError=error.name==='AbortError'?'The explanation took too long. Try again; your forecast is still available.':'The explanation could not be loaded. Your forecast and the geometry signal are still available.';}
  finally{clearTimeout(timeout);if(stillCurrent()){state.commentaryPending=false;state.commentaryController=null;renderCommentary();}}
}

function stopAudio(){
  state.audioGeneration=(state.audioGeneration||0)+1;state.unlockGeneration++;state.audioStarting=false;state.audioEnabled=false;
  if(state.audio){state.audio.pause();state.audio.remove();state.audio=null;}
  if(state.outputAudio){state.outputAudio.pause();state.outputAudio.remove();}
  state.audioHighlight=null;
}
function invalidateVoice(){state.voiceGeneration++;stopAudio();state.voice=null;state.voicePending=false;state.voiceError=null;renderVoice();}
function voiceKey(h,playerId,audience=state.audience,matchId=state.match?.id){return JSON.stringify([CONTENT_VERSION,matchId,h.id,playerId,'en',audience]);}
function bufferVoice(voice){
  if(voice?.status!=='ready')return null;const key=voice.cache_key||voice.audio_url;
  if(mediaCache.has(key))return mediaCache.get(key);
  const audio=new Audio(voice.audio_url);audio.preload='auto';audio.hidden=true;audio.oncanplay=()=>primeUpcomingAudio();audio.load();mediaCache.set(key,audio);return audio;
}
function unlockAudioOutput(){
  if(!state.commentaryOn)return;
  if(state.audioUnlocked){primeUpcomingAudio();return;}
  const audio=state.outputAudio||new Audio();state.outputAudio=audio;audio.id='commentary-audio';audio.hidden=true;audio.preload='auto';audio.dataset.status='unlocking';
  // A tiny PCM silence unlocks this one output element inside the Play gesture.
  // It is never presented as generated commentary; cached speech replaces its source.
  const bytes=new Uint8Array(172),view=new DataView(bytes.buffer);for(const [offset,text]of [[0,'RIFF'],[8,'WAVE'],[12,'fmt '],[36,'data']])for(let i=0;i<text.length;i++)bytes[offset+i]=text.charCodeAt(i);
  view.setUint32(4,164,true);view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,8000,true);view.setUint32(28,16000,true);view.setUint16(32,2,true);view.setUint16(34,16,true);view.setUint32(40,128,true);
  delete audio.dataset.bufferKey;audio.src=`data:audio/wav;base64,${btoa(String.fromCharCode(...bytes))}`;audio.muted=false;audio.volume=state.volume;audio.onended=null;audio.onerror=null;audio.load();document.body.append(audio);
  const generation=++state.unlockGeneration;
  audio.play().then(()=>{if(generation!==state.unlockGeneration)return;audio.pause();state.audioUnlocked=true;audio.dataset.status='ready';primeUpcomingAudio();}).catch(()=>{if(generation!==state.unlockGeneration)return;state.voiceError='The browser has blocked commentary. Replay and captions remain available.';renderVoice();});
}
function primeUpcomingAudio(){
  if(!state.commentaryOn||!state.audioUnlocked||state.audioEnabled||state.audioStarting)return;
  const entry=state.runSchedule.find(item=>!item.attempted);if(!entry)return;
  const voice=voiceCache.get(voiceKey(entry.h,state.focusId||entry.h.player_id))||entry.voice;
  if(voice?.status!=='ready')return;
  const audio=state.outputAudio,key=voice.cache_key||voice.audio_url;if(!audio||audio.dataset.bufferKey===key)return;
  audio.pause();audio.onended=null;audio.onerror=null;audio.src=voice.audio_url;audio.dataset.bufferKey=key;audio.dataset.status='prepared';audio.load();
}
async function prepareVoice(h,playerId,{cacheOnly=false}={}){
  const matchId=state.match?.id,audience=state.audience,key=voiceKey(h,playerId,audience,matchId),clip=clipFor(h);
  const requestKey=`${key}:${cacheOnly?'cached':'prepare'}`;
  if(!matchId)return null;if(voiceCache.has(key))return voiceCache.get(key);if(voiceRequests.has(requestKey))return voiceRequests.get(requestKey);
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),45000);
  const promise=(async()=>{
    const response=await fetch('/api/commentary',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},body:JSON.stringify({match_id:matchId,highlight_id:h.id,player_id:playerId,audience,language:'en',cache_only:cacheOnly}),signal:controller.signal});
    let result;try{result=await response.json();}catch{throw new Error('Voice service returned an unreadable response.');}
    if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Voice service is unavailable.');
    if(!['ready','unavailable'].includes(result.status))throw new Error('Voice response could not be aligned with this highlight.');
    if(result.status==='ready'){
      if(typeof result.text!=='string'||!result.text.trim()||typeof result.audio_url!=='string')throw new Error('The commentary audio is incomplete.');
      if(result.content_version&&result.content_version!==CONTENT_VERSION)throw new Error('Commentary is being updated. Please reload this replay.');
      if(result.match_id!==matchId||result.highlight_id!==h.id||result.player_id!==playerId||result.audience!==audience||result.language!=='en')throw new Error('Commentary belongs to a different focus. No audio was started.');
      const url=new URL(result.audio_url,location.href);if(url.origin!==location.origin)throw new Error('The audio source could not be verified.');
      if(![result.clip_start_s,result.clip_end_s,result.speech_start_s,result.duration_s].every(Number.isFinite)||Math.abs(result.clip_start_s-clip.start)>.05||Math.abs(result.clip_end_s-clip.end)>.05||Math.abs(result.speech_start_s-h.time_s)>.05||result.duration_s<=0||result.duration_s>clip.end-clip.start+.05)throw new Error('Commentary timing does not match this highlight.');
      if(result.cue_timebase!=='replay'||!Array.isArray(result.cues)||!result.cues.every(cue=>Number.isFinite(cue.start_s)&&Number.isFinite(cue.end_s)&&cue.start_s>=clip.start-.05&&cue.end_s<=clip.end+.05&&cue.end_s>=cue.start_s&&typeof cue.text==='string'))throw new Error('Commentary captions could not be aligned. No audio was started.');
      voiceCache.set(key,result);bufferVoice(result);primeUpcomingAudio();
    }
    return result;
  })().finally(()=>{clearTimeout(timeout);voiceRequests.delete(requestKey);});
  voiceRequests.set(requestKey,promise);return promise;
}
async function prepareActiveVoice({cacheOnly=false}={}){
  const h=activeMoment();if(!h||!state.match)return null;
  const playerId=state.focusId||h.player_id,key=voiceKey(h,playerId),generation=state.voiceGeneration;
  state.voicePending=true;state.voiceError=null;renderVoice();
  try{
    const result=await prepareVoice(h,playerId,{cacheOnly});
    if(generation!==state.voiceGeneration||key!==voiceKey(activeMoment()||{},state.selectedId))return null;
    state.voice=result;state.voicePending=false;
    if(result?.status!=='ready')state.voiceError=result?.notice||result?.reason||'Voice is unavailable. The replay, captions and evidence remain available.';
    renderVoice();renderSubtitles();
    if(result?.status==='ready'&&state.prediction&&state.commentary?.mode==='local'&&state.commentary?.reason==='busy'&&!state.commentaryPending&&!voiceInsightRefresh.has(key)){
      voiceInsightRefresh.add(key);explainMoment();
    }
    return result;
  }catch(error){
    if(generation!==state.voiceGeneration)return null;
    state.voicePending=false;state.voiceError=error.name==='AbortError'?'Voice preparation timed out. Replay and evidence remain available.':`${error.message} Replay and evidence remain available.`;
    renderVoice();return null;
  }
}
function activateScheduledHighlight(entry){
  const h=entry.h;if(h.id===state.activeHighlight)return;
  stopAudio();invalidatePending();resetCommentary();state.voiceGeneration++;state.voicePending=false;state.voiceError=null;state.voiceSkipped=null;
  state.activeHighlight=h.id;state.segment=clipFor(h);state.prediction=null;state.overlayKey=null;state.voice=entry.voice;
  selectPlayer(state.focusId||h.player_id,{explain:false});renderHighlights();predict({anchor:h.anchor_frame,automatic:true});
}
function scheduleNarration(){
  for(const entry of state.runSchedule){
    if(entry.attempted||state.time+1e-7<entry.h.time_s)continue;
    entry.attempted=true;activateScheduledHighlight(entry);
    const voice=voiceCache.get(voiceKey(entry.h,state.focusId||entry.h.player_id))||entry.voice;
    state.voice=voice;state.voiceSkipped=null;
    if(!state.commentaryOn){renderVoice();continue;}
    const audio=state.outputAudio;
    if(state.time-entry.h.time_s>.25||voice?.status!=='ready'||!audio||audio.dataset.bufferKey!==(voice.cache_key||voice.audio_url)||audio.readyState<2){
      state.voiceSkipped='This highlight uses captions only · audio was not ready at its event time.';renderVoice();primeUpcomingAudio();continue;
    }
    startScheduledAudio(entry,voice,audio);
  }
}
function startScheduledAudio(entry,voice,audio){
  stopAudio();const generation=state.audioGeneration,runGeneration=state.runGeneration;
  state.audio=audio;state.audioBase=voice.clip_start_s;state.audioHighlight=entry.h.id;state.audioStarting=true;state.audioEnabled=true;state.voiceError=null;
  audio.id='commentary-audio';audio.dataset.highlightId=entry.h.id;audio.dataset.anchorTime=String(entry.h.time_s);audio.dataset.clipStart=String(voice.clip_start_s);audio.dataset.replayMode=state.replayMode;
  audio.dataset.status='speaking';audio.volume=state.volume;audio.muted=false;audio.playbackRate=state.speed;audio.currentTime=state.time-voice.clip_start_s;document.body.append(audio);
  const current=()=>state.audio===audio&&state.audioGeneration===generation&&state.runGeneration===runGeneration;
  const attempt=state.mediaPlayAttempt=(state.mediaPlayAttempt||0)+1;
  audio.onerror=()=>{if(!current())return;audio.pause();state.audioEnabled=false;state.audioStarting=false;state.voiceError='Audio is unavailable for this highlight. Replay and captions continue.';renderVoice();};
  audio.onended=()=>{if(!current())return;state.audioEnabled=false;audio.dataset.status='complete';renderVoice();primeUpcomingAudio();};
  audio.play().then(()=>{
    if(!current()||attempt!==state.mediaPlayAttempt)return;
    state.audioStarting=false;
    if(!state.playing){audio.pause();state.audioEnabled=true;renderVoice();return;}
    if(state.time-entry.h.time_s>.4){stopAudio();state.voiceSkipped='Audio arrived after its event. Captions continue; replay to listen.';renderVoice();primeUpcomingAudio();return;}
    state.audioEnabled=true;renderVoice();
  }).catch(()=>{if(!current()||attempt!==state.mediaPlayAttempt)return;state.audioEnabled=false;state.audioStarting=false;state.voiceError='Your browser paused sound. Restart this highlight to listen; replay and captions continue.';renderVoice();});
}
function bringReplayIntoView(){
  // Only an explicit Play action may adjust the viewport.
  const dock=document.querySelector('.replay-dock'),pitch=document.querySelector('.pitch-wrap'),story=document.querySelector('.mobile-story');
  if(!dock||!pitch)return;
  const bottom=pitch.getBoundingClientRect().bottom,viewport=window.innerHeight,margin=12;
  if(bottom<=viewport-margin)return;
  const dockTop=dock.getBoundingClientRect().top;
  if(bottom-dockTop>viewport-margin*2)return;
  const storyTop=story?.getBoundingClientRect().top;
  const top=Number.isFinite(storyTop)&&bottom-storyTop<=viewport-margin*2?storyTop:dockTop;
  window.scrollBy({top:Math.max(0,top-margin),behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
}
function renderVoice(){
  renderWorkflow();
  const h=activeMoment(),ready=state.voice?.status==='ready';
  const next=state.runSchedule.find(entry=>!entry.attempted)?.h;
  const complete=state.sessionStarted&&!state.playing&&state.time>=(state.replayMode==='highlight'&&state.segment?state.segment.end:state.match?.timestamps.at(-1)??Infinity);
  $('commentary-play').disabled=!state.match;$('commentary-play').textContent=`▶ Play full replay · ${Math.round(state.match?.timestamps.at(-1)||40)}s`;$('commentary-play').setAttribute('aria-busy','false');
  if($('highlight-play')){$('highlight-play').disabled=!state.match||state.highlightPending||(!h&&!state.highlights.length);$('highlight-play').textContent='▶ Play this highlight';}
  if($('commentary-toggle'))$('commentary-toggle').checked=state.commentaryOn;
  $('audio-mute').textContent=state.commentaryOn?'Commentary on':'Commentary off';$('audio-mute').setAttribute('aria-pressed',String(!state.commentaryOn));
  $('voice-status').classList.toggle('error',!!state.voiceError);
  const untilSpeech=Math.max(0,Math.ceil((state.voice?.speech_start_s??h?.time_s??state.time)-state.time));
  $('voice-status').textContent=complete?(state.replayMode==='highlight'?'Highlight complete · Play this highlight to watch again, or Play full replay.':'Full replay complete · Play full replay to watch again, or choose a highlight.'):
    !state.commentaryOn?'Commentary off · both playback choices use the same switch; captions remain available.':
    state.voiceError?`${state.voiceError} AI-generated commentary.`:
    state.audioEnabled&&!state.playing?'Paused · picture and AI-generated commentary will resume together':
    state.audioEnabled?'AI-generated commentary speaking · synchronized to this highlight':
    state.voiceSkipped||
    (state.sessionStarted&&!state.playing?'Paused · Resume continues here; either Play button starts its replay again.':
    state.playing&&h&&untilSpeech>0?`Build-up · highlight in ${untilSpeech} ${untilSpeech===1?'second':'seconds'}; prepared commentary starts there.`:
    state.playing&&state.replayMode==='full'?(next?`Full replay · next highlight at ${clock(next.time_s,2)}. Prepared AI commentary starts there.`:state.highlights.length?'Full replay continues · all scheduled highlights have been shown.':'Full replay continues · no qualifying highlights for this focus. Choose another player for commentary.'):
    state.voicePending?'Preparing AI commentary in the background. Choose a player, then press Play.':
    !h&&!state.highlights.length?'No qualifying highlights for this focus. Play full replay, or choose another player.':
    ready?'AI commentary ready. Choose a player, then press Play full replay or Play this highlight.':'Choose a player, then press Play. Unready commentary uses captions; the replay keeps moving.');
  $('voice-status').dataset.replayMode=state.replayMode;$('voice-status').dataset.highlightId=h?.id||'';
  renderSpokenPreview();
}
function renderSpokenPreview(){
  const preview=$('spoken-preview');if(!preview)return;
  const h=activeMoment(),stages=state.commentary?.stages;
  const fallback=state.audience==='fan'?localExplanation():stages?[stages.observed,stages.forecast].filter(value=>typeof value==='string'&&value.trim()).join(' '):localExplanation();
  emphasizeText(preview,h?(state.voice?.text||state.commentary?.reviewed_script||fallback):state.prediction?fallback:state.replayMode==='full'?'Commentary follows the detected highlights in time order. The full replay continues between them.':'Choose a highlight to prepare its short commentary.');
  if(h){preview.dataset.anchorTime=String(h.time_s);preview.title=`Prepared for the highlight at ${clock(h.time_s,2)}; next-second forecast is anchored to that moment.`;}
  else{delete preview.dataset.anchorTime;preview.removeAttribute('title');}
}
function renderSubtitles(){
  const root=$('pitch-subtitles'),h=activeMoment(),clip=state.segment;
  if(!state.captions||!h||!clip||state.time<clip.start||state.time>=clip.end){root.hidden=true;return;}
  const voice=state.voice;
  let text='';
  if(voice?.status==='ready'&&Array.isArray(voice.cues)){
    const cue=voice.cues.find(c=>state.time>=c.start_s&&state.time<c.end_s);text=cue?.text||'';
  }else{
    const e=selectedEvidence(),delta=h.evidence.observed_gap_change_m;
    text=state.time<h.time_s?`Follow ${playerName(h.player_id)} into this highlight.`:state.time<h.time_s+2?`${playerName(h.player_id)} just ${delta>=0?'opened':'lost'} space from the nearest opponent.`:e?`The next-second forecast is an estimate. The replay shows the recorded outcome.`:'The replay continues while commentary is prepared.';
  }
  root.hidden=!text;if(root.textContent!==text)root.textContent=text;
}
function setCommentaryEnabled(enabled){
  state.commentaryOn=!!enabled;
  if(!state.commentaryOn)stopAudio();
  else{unlockAudioOutput();if(state.sessionStarted&&state.playing&&activeMoment()&&state.time>=activeMoment().time_s)state.voiceSkipped='Commentary is on for the next highlight. Replay this highlight to hear it from the start.';}
  renderVoice();
}

async function init(){setError(null);$('model-status').textContent='Connecting to model…';$('model-status').className='model-status';try{const [health,list]=await Promise.all([getJSON('/api/health'),getJSON('/api/matches')]);state.health=health;state.matches=list.matches||[];if(!state.matches.length)throw new Error('No replay scenarios are available.');$('model-status').textContent=health.model_loaded?`MODEL CONNECTED · ${(health.device||'CPU').toUpperCase()}`:'MODEL UNAVAILABLE';$('model-status').className=`model-status ${health.model_loaded?'ready':'failed'}`;$('model-status').title=`${health.model_name||'ST-GNN + TCN'} | checkpoint SHA-256: ${health.checkpoint_sha256||'unavailable'} | ${health.model_selection?.criterion||''}`;clear($('scenario-tabs'));state.matches.forEach((m,i)=>{const button=document.createElement('button');button.type='button';button.className='scenario-tab';button.dataset.matchId=m.id;button.setAttribute('aria-pressed','false');const number=document.createElement('span');number.className='tab-number';number.textContent=`0${i+1}`;const scene=SCENES[m.id]||{title:m.title,subtitle:'Explore this synthetic replay.'};const label=document.createElement('span');label.className='scene-label';label.append(Object.assign(document.createElement('strong'),{textContent:scene.title}),Object.assign(document.createElement('small'),{textContent:scene.subtitle}));button.title=scene.subtitle;button.setAttribute('aria-label',`${scene.title}: ${scene.subtitle}`);button.append(number,label);button.addEventListener('click',()=>loadMatch(m.id));$('scenario-tabs').append(button);});await loadMatch(state.matches[0].id);}catch(error){state.health=null;$('model-status').textContent='ENGINE OFFLINE';$('model-status').className='model-status failed';setError(error);renderControls();}}

$('play-button').addEventListener('click',playMoment);
$('step-button').addEventListener('click',()=>moveTo(state.match.timestamps[state.frame]+1,{manual:true,clearMoment:false}));
$('timeline').addEventListener('input',event=>moveTo(Number(event.target.value),{manual:true}));
$('speed-select').addEventListener('change',event=>{state.speed=Number(event.target.value);state.clockOrigin=performance.now();state.clockTime=state.time;if(state.audio)state.audio.playbackRate=state.speed;});
$('player-select').addEventListener('change',event=>inspectPlayer(event.target.value));
$('manual-player-select')?.addEventListener('change',event=>inspectPlayer(event.target.value));
for(const button of $('focus-shortcuts')?.querySelectorAll('button[data-focus]')||[])button.addEventListener('click',()=>focusPlayer(button.dataset.focus));
$('focus-select').addEventListener('change',event=>focusPlayer(event.target.value));
$('predict-button').addEventListener('click',()=>predict());
$('outcome-button').addEventListener('click',playMoment);
$('next-highlight').addEventListener('click',()=>{const i=state.highlights.findIndex(h=>h.id===state.activeHighlight);if(state.highlights.length)chooseHighlight(state.highlights[(i+1)%state.highlights.length].id);});
$('show-all').addEventListener('change',event=>{state.showAll=event.target.checked;renderPitch();});
$('retry-button').addEventListener('click',init);
$('explain-button').addEventListener('click',explainMoment);
$('commentary-play').addEventListener('click',playFullReplay);
$('highlight-play')?.addEventListener('click',playHighlight);
$('commentary-toggle')?.addEventListener('change',event=>setCommentaryEnabled(event.target.checked));
$('audio-mute').addEventListener('click',()=>setCommentaryEnabled(!state.commentaryOn));
$('audio-volume').addEventListener('input',event=>{state.volume=Number(event.target.value);if(state.audio)state.audio.volume=state.volume;});
$('captions-toggle').addEventListener('change',event=>{state.captions=event.target.checked;renderSubtitles();});
$('step-mode').addEventListener('change',event=>{state.stepMode=event.target.checked;if(state.playing)state.stopAt=playbackLimit();});
$('inspector').addEventListener('toggle',()=>{if($('inspector').open)renderDetail();});
for(const audience of ['fan','analyst'])$(`audience-${audience}`).addEventListener('click',()=>{if(state.audience===audience)return;resetCommentary();cancelReplayRun();invalidateVoice();state.audience=audience;if(state.sessionStarted){const h=activeMoment();if(state.replayMode==='highlight'){if(h&&state.time<h.time_s)prepareReplaySchedule(-Infinity,{only:h});}else prepareReplaySchedule(state.time);}renderInsight();renderHighlights();renderPitch();renderCommentary();explainMoment();prepareActiveVoice({cacheOnly:state.replayMode==='full'});});
function tick(now){
  if(state.playing&&state.match&&state.clockOrigin!==null){
    const stop=state.stopAt??state.match.timestamps.at(-1);
    // Absolute monotonic elapsed time prevents network delays and slow frames from losing replay time.
    state.time=Math.min(state.clockTime+(now-state.clockOrigin)/1000*state.speed,stop);const nextFrame=frameAt(state.time),changed=nextFrame!==state.frame;state.frame=nextFrame;
    scheduleNarration();
    if(changed||state.time>=stop)render();
    if(state.audioEnabled&&state.audio&&!state.audio.paused&&state.audio.readyState>=1){
      const elapsed=state.time-state.audioBase;
      if(elapsed>=0&&elapsed<state.audio.duration&&Math.abs(state.audio.currentTime-elapsed)>.3)state.audio.currentTime=elapsed;
    }
    if(state.time>=stop){setPlaying(false);render();}
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);init();

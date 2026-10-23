/* Local model chat. Conversation text never leaves the loopback server. */
'use strict';
const chatUi={runId:'',source:'adapter',drafts:{},conversations:{},maxTokens:256,reasoning:false,temperature:.7,pending:null,snapshot:{status:'unloaded'},error:'',polling:false,paintKey:''};
let chatPollTimer=null,chatPollAgain=false;
try{const saved=JSON.parse(localStorage.getItem('tuna-chat')||'{}');for(const key of ['runId','source','drafts','conversations','maxTokens','reasoning','temperature','pending'])if(saved[key]!==undefined)chatUi[key]=saved[key];}catch{}
function chatKey(){return chatUi.runId+':'+chatUi.source;}
function saveChat(){try{localStorage.setItem('tuna-chat',JSON.stringify({runId:chatUi.runId,source:chatUi.source,drafts:chatUi.drafts,conversations:chatUi.conversations,maxTokens:chatUi.maxTokens,reasoning:chatUi.reasoning,temperature:chatUi.temperature,pending:chatUi.pending}));}catch{}}
function chatRuns(){return state.runs.filter(r=>r.status==='completed');}
function selectedChatRun(){return chatRuns().find(r=>r.id===chatUi.runId);}
function chatIsCurrent(){return chatUi.snapshot.run_id===chatUi.runId&&chatUi.snapshot.source===chatUi.source;}
function chatBusy(){return ['loading','generating'].includes(chatUi.snapshot.status);}
function chatPage(){
 const runs=chatRuns();if(runs.length&&!selectedChatRun())chatUi.runId=runs[0].id;
 const run=selectedChatRun();if(run&&chatUi.source==='gguf'&&run.gguf?.status!=='completed')chatUi.source='adapter';
 return `${head('Meet your fine-tuned model.','Try a conversation. See what your model has learned.','','')}
 ${!runs.length?`<div class="panel">${empty('Your model will meet you here.','Complete a training run, then come back for a conversation.','new-run','Create a training run','fish')}</div>`:`
 <div class="chat-layout"><aside class="chat-settings panel"><div class="panel-header"><h2>Your model</h2>${icon('model')}</div><div class="panel-body">
 <div class="field"><label for="chat-model">Completed fine-tune</label><select id="chat-model">${runs.map(r=>`<option value="${esc(r.id)}" ${r.id===chatUi.runId?'selected':''}>${esc(r.name)} · ${methods[r.method]?.name||'LoRA'}</option>`).join('')}</select></div>
 <div class="field"><label for="chat-source">Model format</label><select id="chat-source"><option value="adapter" ${chatUi.source==='adapter'?'selected':''}>Trained adapter</option>${run?.gguf?.status==='completed'?`<option value="gguf" ${chatUi.source==='gguf'?'selected':''}>Exported GGUF</option>`:''}</select><small>${chatUi.source==='gguf'?'GGUF imports are cached locally and may need extra disk space.':'Uses this run’s saved adapter and its local base weights.'}</small></div>
 <div id="chat-model-state" class="chat-model-state"></div><div class="chat-load-actions"><button id="chat-load" class="button primary" data-action="chat-load">${icon('play')}Load model</button><button id="chat-unload" class="button" data-action="chat-unload">${icon('stop')}Unload</button></div>
 <div class="chat-policy"><div class="chat-policy-title">${icon(run?.method==='tuna'?'shield':'leaf')}<strong>${run?.method==='tuna'?'Saved Tuna policy':'Local conversation'}</strong></div><p>${run?.method==='tuna'?'Your saved system policy is included with every turn. Responses remain probabilistic.':'Messages and replies stay on this machine.'}</p>${run?.method==='tuna'?`<details><summary>View saved policy</summary><p>${esc(run.config?.system_prompt||'Saved with this fine-tune.')}</p></details>`:''}</div>
 <div class="chat-options"><div class="field"><label for="chat-length">Response length</label><select id="chat-length">${[[128,'Short · 128 tokens'],[256,'Standard · 256 tokens'],[512,'Detailed · 512 tokens'],[1024,'Long · 1,024 tokens'],[2048,'Extended · 2,048 tokens'],[null,'Unlimited']].map(([n,label])=>`<option value="${n===null?'unlimited':n}" ${(n===null?chatUi.maxTokens===null:chatUi.maxTokens!==null&&Number(chatUi.maxTokens)===n)?'selected':''}>${label}</option>`).join('')}</select><small id="chat-length-note"></small></div><div class="field"><label for="chat-temperature">Creativity</label><select id="chat-temperature">${[[0,'Focused'],[.7,'Balanced'],[1,'Creative']].map(([n,label])=>`<option value="${n}" ${Number(chatUi.temperature)===n?'selected':''}>${label}</option>`).join('')}</select></div><div class="field chat-reasoning-option"><label for="chat-reasoning">Reasoning</label><select id="chat-reasoning"><option value="off" ${!chatUi.reasoning?'selected':''}>Off · direct answer</option><option value="on" ${chatUi.reasoning?'selected':''}>On · think before answering</option></select><small id="chat-reasoning-note">Uses the model’s native reasoning mode when supported.</small></div></div>
 <p class="chat-memory-note">${icon('leaf')}Unload frees model memory. Idle models unload automatically after five minutes.</p></div></aside>
 <section class="chat-board panel"><div class="chat-board-head"><div><span class="chat-avatar">${icon('fish')}</span><div><h2>${esc(run?.name||'Your fine-tune')}</h2><span class="small muted">A conversation with your own model</span></div></div><button class="button ghost" id="chat-new" data-action="chat-new">${icon('plus')}New chat</button></div>
 <div id="chat-transcript" class="chat-transcript" role="log" aria-label="Conversation" aria-live="polite"></div>
 <div id="chat-error" class="chat-error" role="alert"></div>
 <form id="chat-form" class="chat-composer"><label class="sr-only" for="chat-input">Message your model</label><textarea id="chat-input" rows="3" maxlength="20000" placeholder="Say hello to your model…">${esc(chatUi.drafts[chatKey()]||'')}</textarea><div class="chat-composer-bottom"><span id="chat-hint">Enter to send · Shift + Enter for a new line</span><button id="chat-stop" type="button" class="button danger hidden" data-action="chat-stop">${icon('stop')}Stop reply</button><button id="chat-send" type="submit" class="button primary">Send ${icon('arrow')}</button></div></form>
 <p class="chat-privacy">${icon('lock')}Stored in this browser. Inference runs locally. Check important answers.</p></section></div>`}`;
}
function chatRichText(text){return String(text).split(/```/).map((part,index)=>index%2?`<pre><code>${esc(part.replace(/^[a-zA-Z0-9_+#.-]*\n/,''))}</code></pre>`:`<span>${esc(part)}</span>`).join('');}
function chatTurn(role,text,pending=false,usage=null,reasoning='',thinking=false,id=''){
 const thought=reasoning||thinking?`<details class="chat-reasoning" data-reasoning-key="${esc(id)}" ${pending&&thinking?'open':''}><summary>Reasoning${thinking?' · thinking…':''}</summary><div class="chat-message">${reasoning?chatRichText(reasoning):'<span class="muted">Thinking…</span>'}</div></details>`:'';
 const placeholder=thinking?'':reasoning&&!pending?'Stopped before an answer.':pending?'Writing an answer…':'';
 return `<article class="chat-turn ${role}"><div class="chat-turn-label">${role==='user'?'You':`${icon('fish')}Your model`}${pending?'<span class="chat-typing">replying</span>':''}</div>${thought}<div class="chat-message">${text?chatRichText(text):`<span class="muted">${placeholder}</span>`}</div>${usage?`<div class="chat-usage">${fmt(usage.output_tokens)} tokens · ${Number(usage.elapsed_seconds||0).toFixed(1)} s${Number.isFinite(usage.tokens_per_second)&&usage.tokens_per_second>=0?` · ${usage.tokens_per_second.toFixed(1)} tokens/s`:''}${Number.isFinite(usage.time_to_first_token_seconds)&&usage.time_to_first_token_seconds>=0?` · first token ${usage.time_to_first_token_seconds.toFixed(2)} s`:''}${usage.stopped?' · stopped':''}${usage.contextLimited?' · chat context reached':usage.limited?' · response limit reached':''}</div>`:''}</article>`;
}
function chatRequestHistory(messages){
 const result=[];
 for(let i=0;i+1<messages.length;i+=2)if(messages[i].role==='user'&&messages[i+1].role==='assistant'&&messages[i+1].content?.trim())result.push({role:'user',content:messages[i].content},{role:'assistant',content:messages[i+1].content});
 return result;
}
function chatRuntimeLabel(s){
 const labels={transformers:'Transformers',torch:'PyTorch',pytorch:'PyTorch',llama_cpp:'llama.cpp',bf16:'BF16',bfloat16:'BF16',fp16:'FP16',float16:'FP16',nf4:'4-bit NF4',fp32:'FP32',float32:'FP32'};
 return [s.engine,s.precision,s.optimizations?.includes('Compiled static-cache decode')?'Compiled GPU decode':''].filter(value=>typeof value==='string'&&value.trim()).map(value=>labels[value.toLowerCase()]||value.replace(/_/g,' ')).join(' · ');
}
function paintChat(){
 if(state.page!=='chat'||!document.querySelector('#chat-transcript'))return;
 const s=chatUi.snapshot,busy=chatBusy(),current=chatIsCurrent(),ready=current&&s.status==='ready',generating=s.status==='generating';
 const label=current?({unloaded:'Not loaded',loading:'Loading locally…',ready:'Ready to chat',generating:'Replying…',error:'Model needs attention'}[s.status]||'Not loaded'):'Not loaded';
 const detail=current?[s.message||'',...(['ready','generating'].includes(s.status)?[chatRuntimeLabel(s)]:[])].filter(Boolean).join(' · '):s.status==='ready'?'Another model is loaded. Load this fine-tune to switch.':'Load your fine-tune to start chatting.';
 document.querySelector('#chat-model-state').innerHTML=`<span class="${ready||generating?'ready':''}"><i class="dot"></i>${esc(label)}</span><small>${esc(detail)}</small>`;
 const load=document.querySelector('#chat-load');load.disabled=busy||ready;load.innerHTML=icon('play')+(ready?'Model loaded':'Load model');
 const unload=document.querySelector('#chat-unload');unload.disabled=s.status==='unloaded';unload.innerHTML=icon('stop')+(s.status==='loading'?'Cancel loading':'Unload');
 for(const id of ['chat-model','chat-source','chat-length','chat-temperature'])document.getElementById(id).disabled=busy;
 const reasoning=document.getElementById('chat-reasoning');reasoning.disabled=busy||(current&&s.reasoning_supported===false);reasoning.value=chatUi.reasoning?'on':'off';
 document.getElementById('chat-reasoning-note').textContent=current&&s.reasoning_supported===false?'This model does not expose a native reasoning switch.':'Uses native thinking mode. Reasoning and the answer share the response budget.';
 document.getElementById('chat-length-note').textContent=chatUi.maxTokens===null?`No reply cap; uses the remaining ${current&&s.context_tokens?fmt(s.context_tokens)+'-token ':''}chat context. Stop anytime.`:'';
 document.getElementById('chat-new').disabled=generating||!!chatUi.pending;
 document.getElementById('chat-send').disabled=!ready||!!chatUi.pending||!(chatUi.drafts[chatKey()]||'').trim();
 document.getElementById('chat-input').disabled=generating||!!chatUi.pending;
 const stop=document.getElementById('chat-stop');stop.classList.toggle('hidden',!generating||!chatUi.pending||chatUi.pending.requestId!==s.request_id);stop.disabled=!!s.stop_requested;
 document.getElementById('chat-error').textContent=chatUi.error||(current&&s.status==='error'?s.error||'':'');
 document.getElementById('chat-hint').textContent=ready?'Enter to send · Shift + Enter for a new line':generating?'Your model is writing a reply…':'Load the model to send a message.';
 const messages=chatUi.conversations[chatKey()]||[],pending=chatUi.pending?.key===chatKey()?chatUi.pending:null;
 const key=JSON.stringify([messages,pending?.text,pending?.reasoning,pending?.thinking,pending?.requestId]);
 if(chatUi.paintKey!==key||!document.querySelector('#chat-transcript').children.length){
  const transcript=document.getElementById('chat-transcript'),bottom=transcript.scrollHeight-transcript.scrollTop-transcript.clientHeight<100;
  const disclosures=new Map([...transcript.querySelectorAll('[data-reasoning-key]')].map(item=>{const body=item.querySelector('.chat-message');return [item.dataset.reasoningKey,{open:item.open,scroll:body.scrollTop,follow:body.scrollHeight-body.scrollTop-body.clientHeight<30}];}));
  transcript.innerHTML=messages.length||pending?messages.map((m,index)=>chatTurn(m.role,m.content,false,m.usage,m.reasoning||'',false,String(index))).join('')+(pending?chatTurn('user',pending.user)+chatTurn('assistant',pending.text||'',true,null,pending.reasoning||'',!!pending.thinking,'pending'):''):`<div class="chat-welcome"><div class="chat-welcome-mark">${icon('fish')}</div><h3>Give your model a hello.</h3><p>This is your fine-tune, ready for its next conversation.</p><div class="chat-suggestions">${['Tell me about yourself.','What can you help me with?','Show me a short example.'].map(text=>`<button type="button" data-action="chat-suggestion" data-prompt="${esc(text)}">${esc(text)}${icon('arrow')}</button>`).join('')}</div></div>`;
  for(const item of transcript.querySelectorAll('[data-reasoning-key]')){const previous=disclosures.get(item.dataset.reasoningKey),body=item.querySelector('.chat-message');if(previous)item.open=previous.open;body.scrollTop=item.dataset.reasoningKey==='pending'&&(!previous||previous.follow)?body.scrollHeight:previous?.scroll||0;}
  if(bottom||pending)transcript.scrollTop=transcript.scrollHeight;chatUi.paintKey=key;
 }
}
function chatMount(){chatUi.paintKey='';paintChat();scheduleChatPoll(true);}
function scheduleChatPoll(immediate=false){
 clearTimeout(chatPollTimer);chatPollTimer=null;
 if(state.page!=='chat'&&!chatUi.pending)return;
 if(chatUi.polling){chatPollAgain=chatPollAgain||immediate;return;}
 chatPollTimer=setTimeout(chatPoll,immediate?0:chatBusy()||chatUi.pending?100:1500);
}
function acceptChatSnapshot(s){
 if(s.instance_id===chatUi.snapshot.instance_id&&s.revision<chatUi.snapshot.revision)return;
 chatUi.snapshot=s;
 if(chatIsCurrent()&&s.reasoning_supported===false&&chatUi.reasoning){chatUi.reasoning=false;saveChat();}
 const p=chatUi.pending;
 if(p?.requestId&&s.request_id===p.requestId){
  p.text=s.text||'';
  p.reasoning=s.reasoning||'';p.thinking=!!s.reasoning_active;
  if(s.status!=='generating'){
   if(p.text.trim()||p.reasoning.trim())chatUi.conversations[p.key]=[...(chatUi.conversations[p.key]||[]),{role:'user',content:p.user},{role:'assistant',content:p.text,reasoning:p.reasoning,usage:{...(s.usage||{}),stopped:s.finish_reason==='cancelled'||s.finish_reason==='unloaded',limited:s.finish_reason==='length',contextLimited:s.finish_reason==='context'}}];
   else{chatUi.drafts[p.key]=p.user;chatUi.error=s.error||'The model returned no text. Try another message or a longer response.';}
   if(s.error)chatUi.error=s.error;
   chatUi.pending=null;saveChat();
   const input=document.getElementById('chat-input');if(input)input.value=chatUi.drafts[chatKey()]||'';
  }
 }
 paintChat();
}
async function chatPoll(){if(chatUi.polling||(state.page!=='chat'&&!chatUi.pending))return;chatUi.polling=true;try{
 const s=(await api('/chat')).chat;acceptChatSnapshot(s);const p=chatUi.pending;
 if(p?.requestId&&s.request_id!==p.requestId){
  const reply=(await api('/chat/replies/'+encodeURIComponent(p.requestId))).reply;
  if(reply){const live=chatUi.snapshot;acceptChatSnapshot({...live,...reply,status:'ready'});chatUi.snapshot=live;paintChat();}
  else if(s.status!=='generating'&&s.status!=='loading'){chatUi.drafts[p.key]=p.user;chatUi.pending=null;chatUi.error='The model session changed. Your message is kept as a draft.';saveChat();paintChat();}
 }
 }catch(error){chatUi.error=error.message;paintChat();}finally{chatUi.polling=false;const immediate=chatPollAgain;chatPollAgain=false;scheduleChatPoll(immediate);}}
async function loadChat(){if(!selectedChatRun())return;chatUi.error='';try{acceptChatSnapshot((await api('/chat/load',{method:'POST',body:JSON.stringify({run_id:chatUi.runId,source:chatUi.source})})).chat);scheduleChatPoll(true);}catch(error){chatUi.error=error.message;paintChat();}}
async function sendChat(){
 const user=(chatUi.drafts[chatKey()]||'').trim();if(!user||chatUi.pending||!chatIsCurrent()||chatUi.snapshot.status!=='ready')return;
 const key=chatKey(),history=chatUi.conversations[key]||[];chatUi.error='';chatUi.pending={key,user,text:'',reasoning:'',thinking:chatUi.reasoning,requestId:null};chatUi.drafts[key]='';saveChat();document.getElementById('chat-input').value='';paintChat();
 try{const result=await api('/chat/message',{method:'POST',body:JSON.stringify({run_id:chatUi.runId,source:chatUi.source,messages:[...chatRequestHistory(history),{role:'user',content:user}],max_new_tokens:chatUi.maxTokens===null?null:Number(chatUi.maxTokens),reasoning:chatUi.reasoning,temperature:Number(chatUi.temperature)})});chatUi.pending.requestId=result.chat.request_id;saveChat();acceptChatSnapshot(result.chat);scheduleChatPoll(true);}catch(error){chatUi.drafts[key]=user;chatUi.pending=null;chatUi.error=error.message;saveChat();const input=document.getElementById('chat-input');if(input)input.value=user;paintChat();}
}
document.addEventListener('click',async event=>{const target=event.target.closest('[data-action]');if(!target)return;const action=target.dataset.action;if(!action.startsWith('chat-'))return;try{
 if(action==='chat-run'){if(chatBusy()||chatUi.pending){toast('Finish or stop the current chat operation first.');return;}const id=target.dataset.id;closeModal();chatUi.runId=id;chatUi.source='adapter';chatUi.error='';saveChat();navigate('chat');await loadChat();}
 if(action==='chat-load')await loadChat();
 if(action==='chat-stop')acceptChatSnapshot((await api('/chat/stop',{method:'POST',body:'{}'})).chat);
 if(action==='chat-unload')acceptChatSnapshot((await api('/chat/unload',{method:'POST',body:'{}'})).chat);
 if(action==='chat-new'&&!chatUi.pending){chatUi.conversations[chatKey()]=[];chatUi.drafts[chatKey()]='';chatUi.error='';saveChat();render();}
 if(action==='chat-suggestion'){chatUi.drafts[chatKey()]=target.dataset.prompt;document.getElementById('chat-input').value=target.dataset.prompt;saveChat();paintChat();document.getElementById('chat-input').focus();}
 }catch(error){chatUi.error=error.message;paintChat();}});
document.addEventListener('submit',event=>{if(event.target.id==='chat-form'){event.preventDefault();sendChat();}});
document.addEventListener('input',event=>{if(event.target.id==='chat-input'){chatUi.drafts[chatKey()]=event.target.value;saveChat();paintChat();}});
document.addEventListener('change',event=>{const id=event.target.id;if(id==='chat-model'||id==='chat-source'){chatUi[id==='chat-model'?'runId':'source']=event.target.value;chatUi.error='';saveChat();render();}if(id==='chat-length'||id==='chat-temperature'){chatUi[id==='chat-length'?'maxTokens':'temperature']=event.target.value==='unlimited'?null:Number(event.target.value);saveChat();paintChat();}if(id==='chat-reasoning'){chatUi.reasoning=event.target.value==='on';saveChat();paintChat();}});
document.addEventListener('keydown',event=>{if(event.target.id==='chat-input'&&event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();sendChat();}});
scheduleChatPoll(true);

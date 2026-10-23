/* Run with node tests/test_app_ui.js. Uses only Node's standard library. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const response=(value,status=200)=>({ok:status>=200&&status<300,status,text:async()=>typeof value==='string'?value:JSON.stringify(value)});
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
function setup(options={}){
 const elements=new Map(),listeners={},timers=[];
 const toast={textContent:'',classList:{add(){},remove(){}}};
 elements.set('#toast',toast);elements.set('#modal-root',{innerHTML:''});
 const context=vm.createContext({
  document:{hidden:false,activeElement:null,body:{style:{}},addEventListener:(name,fn)=>{listeners[name]=fn;},getElementById:id=>elements.get('#'+id)||null,querySelector:selector=>elements.get(selector)||null,querySelectorAll:()=>[],createElement:()=>({})},
  window:{addEventListener(){},scrollTo(){}},location:{hash:''},
  localStorage:options.storage||{getItem:()=>null,setItem(){}},
  fetch:options.fetch||(()=>{throw new Error('Unexpected request');}),
  AbortController,TextEncoder,Blob,URL,
  setTimeout:fn=>{timers.push(fn);return timers.length;},clearTimeout(){},setInterval(){},
 });
 vm.runInContext(source,context);
 return {context,elements,listeners,timers,run:code=>vm.runInContext(code,context)};
}
const goodConfig={step:3,name:'Regression run',model:'Qwen/Qwen3-0.6B',method:'tuna',dataset_id:'data',epochs:1,learning_rate:0.0002,batch_size:1,gradient_accumulation:8,max_seq_length:1024,lora_rank:16,lora_alpha:32,seed:42,system_prompt:'Answer clearly.',gradient_checkpointing:true,offline:false};
async function main(){
 let t=setup({storage:{getItem:key=>key==='tuna-settings'?'{broken':'[]',setItem(){throw Error('Quota exceeded');}}});
 assert.equal(t.run('state.settings.offline'),false);
 assert.equal(t.run('state.baseline'),null);
 assert.equal(t.run("saveStored('test',{})"),false);
 t=setup({storage:{getItem:key=>key==='tuna-settings'?'"offline"':'null',setItem(){}}});
 assert.equal(t.run('state.settings.offline'),false);

 for(const [raw,status,pattern] of [['<html>Error</html>',503,/503/],['not JSON',200,/unreadable/],[{error:'Dataset is in use.'},409,/Dataset is in use/],['[]',200,/unexpected response/]]){
  t=setup({fetch:async()=>response(raw,status)});
  await assert.rejects(t.run("api('/status')"),error=>pattern.test(error.message)&&error.kind==='response');
 }
 for(const fetch of [async()=>{throw Error('Failed to fetch');},async()=>({ok:true,status:200,text:async()=>{throw Error('Connection reset');}})]){
  t=setup({fetch});await assert.rejects(t.run("api('/status')"),error=>error.kind==='network'&&/local engine/.test(error.message));
 }
 t=setup({fetch:(_url,{signal})=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(Error('Aborted'))))});
 const timeout=t.run("api('/status')");t.timers[0]();await assert.rejects(timeout,error=>error.kind==='timeout');

 t=setup({fetch:async url=>{
  if(url==='/api/status')return response({device:{name:'GPU',vram_total_gb:'12',vram_used_gb:'2'},missing:'invalid'});
  if(url==='/api/datasets')throw Error('Transient list failure');
  if(url==='/api/runs')return response({runs:[null,{id:'r',config:{model:123},metrics:[null,{loss:1}]}]});
  return response({settings:{offline:true}});
 }});
 t.run("state.page='chat';state.datasets=[{id:'preserved'}];");
 await t.run('refresh()');
 assert.equal(t.run('state.connected'),true);
 assert.equal(t.run('state.datasets[0].id'),'preserved');
 assert.equal(t.run('state.status.device.vram_total_gb'),12);
 assert.equal(t.run('state.runs.length'),1);
 assert.equal(t.run('state.runs[0].config.model'),'');
 assert.equal(t.run('state.runs[0].metrics.length'),1);
 assert.equal(t.run('state.settings.offline'),true);

 t=setup();t.run(`state.datasets=[{id:'data'}];wizard=${JSON.stringify(goodConfig)};`);
 assert.equal(t.run('validateWizard(true)'),'');
 for(const [key,value] of [['batch_size','1.5'],['seed','NaN'],['learning_rate','Infinity'],['lora_rank','0'],['max_seq_length','131073']]){
  t.run(`wizard=${JSON.stringify(goodConfig)};wizard.${key}=${value};`);
  assert.notEqual(t.run('validateWizard(true)'),'',key);
 }
 t.run(`wizard=${JSON.stringify(goodConfig)};wizard.epochs=0.5;`);assert.equal(t.run('validateWizard(true)'),'');
 t.run('state.datasets=[]');assert.match(t.run('validateWizard(true)'),/Choose or import/);
 assert.throws(()=>t.run("datasetRequest('valid','   ')"),/training example/);
 assert.throws(()=>t.run("datasetRequest('invalid/name','{}')"),/path separators/);

 // A closed dialog must stay closed after an earlier request finishes.
 let pending=deferred();t=setup({fetch:()=>pending.promise});
 t.run('globalThis.openCount=0;openModal=()=>{openCount++;modalEpoch++;};');
 const stale=t.run("showRun('old')");t.run('closeModal()');pending.resolve(response({run:{id:'old',config:{}}}));await stale;
 assert.equal(t.run('openCount'),0);assert.equal(t.run('activeRun'),null);

 // Quiet polling must not supersede a run the user has just opened.
 pending=deferred();let calls=0;t=setup({fetch:()=>{calls++;return pending.promise;}});
 t.run("modalType='run';activeRun='old';openModal=()=>{modalEpoch++;};");
 const next=t.run("showRun('new')");await t.run("showRun('new',true)");assert.equal(calls,1);
 pending.resolve(response({run:{id:'new',config:{}}}));await next;assert.equal(t.run('modalData.id'),'new');

 // Loading a file is scoped to that import dialog and latest selected file.
 pending=deferred();t=setup();t.run("modalType='import';");
 t.context.testFile={name:'old.jsonl',size:3,text:()=>pending.promise};
 const read=t.run('loadFile(testFile)');t.run('closeModal()');pending.resolve('{}');await read;
 assert.equal(t.run('modalType'),null);

 // Double clicks create one run, and success cannot reopen a dismissed wizard.
 pending=deferred();calls=0;t=setup({fetch:()=>{calls++;return pending.promise;}});
 t.run(`state.datasets=[{id:'data'}];wizard=${JSON.stringify(goodConfig)};modalType='wizard';refresh=async()=>{};`);
 for(const [key,value] of Object.entries(goodConfig))t.elements.set('#'+key,{value:String(value),checked:true});
 t.elements.set('[data-action=start-training]',{disabled:false,isConnected:false});
 const first=t.run('startTraining()');await t.run('startTraining()');assert.equal(calls,1);
 t.run('closeModal()');pending.resolve(response({run:{id:'created'}}));await first;
 assert.equal(t.run('modalType'),null);assert.equal(t.run('trainingPending'),false);

 console.log('Dashboard UI checks passed: API failure recovery, safe storage, partial refresh, numeric guards, duplicate submits, and stale dialog/file responses.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});

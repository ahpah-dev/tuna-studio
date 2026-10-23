/* Run with node tests/test_chat_ui.js. No browser or npm dependencies. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/chat.js'),'utf8');
function setup(saved={}){
 const timers=[],input={value:''},storage=[];
 const context=vm.createContext({
  document:{addEventListener(){},getElementById:id=>id==='chat-input'?input:null},state:{page:'overview'},
  localStorage:{getItem:()=>JSON.stringify(saved),setItem:(_key,value)=>storage.push(JSON.parse(value))},
  clearTimeout(){},setTimeout:(callback,delay)=>{timers.push({callback,delay});return timers.length;},
  icon:()=>'',fmt:String,esc:value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'),
 });
 vm.runInContext(source,context);
 return {context,timers,input,storage};
}
function read(context,expression){return JSON.parse(vm.runInContext(`JSON.stringify(${expression})`,context));}
const {context,timers}=setup({maxTokens:null,reasoning:true});
assert.equal(timers.length,0,'Off-page chat must not poll without a pending reply');
assert.equal(vm.runInContext('chatUi.maxTokens',context),null);
assert.equal(vm.runInContext('chatUi.reasoning',context),true);
const history=JSON.parse(vm.runInContext(`JSON.stringify(chatRequestHistory([
 {role:'user',content:'First'}, {role:'assistant',content:'Answer',reasoning:'Private thought'},
 {role:'user',content:'Stopped'}, {role:'assistant',content:'',reasoning:'Partial thought'},
 {role:'user',content:'Last'}, {role:'assistant',content:'Final'}]))`,context));
assert.deepEqual(history,[{role:'user',content:'First'},{role:'assistant',content:'Answer'},{role:'user',content:'Last'},{role:'assistant',content:'Final'}]);
const rendered=vm.runInContext(`chatTurn('assistant','42',false,{output_tokens:5,contextLimited:true},'<script>bad()</script>',false,'1')`,context);
assert.ok(rendered.includes('<summary>Reasoning</summary>'));
assert.ok(rendered.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!rendered.includes('<script>'));
assert.ok(rendered.includes('chat context reached'));
assert.ok(!rendered.includes('data-reasoning-key="1" open'));
assert.ok(vm.runInContext(`chatTurn('assistant','',true,null,'Working',true,'pending')`,context).includes('open><summary>Reasoning · thinking…'));
for(const saved of [null,[],42,true,{runId:{},source:'remote',drafts:null,conversations:['bad'],maxTokens:'unlimited',reasoning:'yes',temperature:'0.7',pending:true}]){
 const c=setup(saved).context;
 assert.deepEqual(read(c,'[chatUi.runId,chatUi.source,chatUi.maxTokens,chatUi.reasoning,chatUi.temperature,chatUi.pending]'),['','adapter',256,false,.7,null]);
}
const restored=setup({runId:'r',drafts:{'r:adapter':'Draft','__proto__':'bad'},conversations:{
 'r:adapter':[{role:'user',content:'Hi'},{role:'assistant',content:'Answer',usage:'invalid'},null,{role:'assistant',content:'Bad'}],
 'other:adapter':null,
},pending:{key:'r:adapter',user:'Retry me',requestId:null}});
assert.deepEqual(read(restored.context,'chatUi.conversations'),{'r:adapter':[{role:'user',content:'Hi'},{role:'assistant',content:'Answer',reasoning:'',usage:null}]});
assert.equal(read(restored.context,'chatUi.pending'),null,'A refresh during submission must not leave the composer blocked');
assert.equal(read(restored.context,'chatUi.drafts["r:adapter"]'),'Draft','Recovering a submission must preserve a newer draft');
assert.equal(read(setup({pending:{key:'r:adapter',user:'Retry me',requestId:null}}).context,'chatUi.drafts["r:adapter"]'),'Retry me');
assert.throws(()=>vm.runInContext(`chatRequestMessages(Array.from({length:200},(_,i)=>({role:i%2?'assistant':'user',content:'x'})),'Next')`,context),/conversation is full/);
assert.throws(()=>vm.runInContext(`chatRequestMessages([{role:'user',content:'x'},{role:'assistant',content:'x'.repeat(100000)}],'Next')`,context),/conversation is full/);
assert.deepEqual(read(context,`chatRequestMessages([{role:'user',content:'Hi'},{role:'assistant',content:'Hello'}],'Next')`),[{role:'user',content:'Hi'},{role:'assistant',content:'Hello'},{role:'user',content:'Next'}]);
const quota=setup();
quota.context.localStorage.setItem=()=>{throw Error('QuotaExceededError');};
vm.runInContext('saveChat()',quota.context);
assert.match(read(quota.context,'chatUi.storageError'),/Keep this tab open/,'Failed storage writes must not pretend messages were saved');
quota.context.localStorage.setItem=()=>{};
vm.runInContext('saveChat()',quota.context);
assert.equal(read(quota.context,'chatUi.storageError'),'');

(async()=>{
 const session=setup({runId:'r',pending:{key:'r:adapter',user:'Keep this message',requestId:'old-request'}});
 session.context.api=async url=>url==='/chat'?{chat:{instance_id:'new-session',revision:1,status:'unloaded'}}:{reply:null};
 await vm.runInContext('chatPoll()',session.context);
 assert.equal(read(session.context,'chatUi.pending'),null);
 assert.equal(read(session.context,'chatUi.drafts["r:adapter"]'),'Keep this message');
 assert.equal(session.input.value,'Keep this message','Recovered draft must be visible in the composer');

 const finished=setup({runId:'r',pending:{key:'r:adapter',user:'Hello',requestId:'request'}});
 finished.context.api=async url=>url==='/chat'?{chat:{instance_id:'session',revision:8,status:'unloaded',request_id:null}}:{reply:{request_id:'request',run_id:'r',source:'adapter',text:'Final answer',reasoning:'Thinking',finish_reason:'stop',usage:{output_tokens:4}}};
 await vm.runInContext('chatPoll()',finished.context);
 assert.equal(read(finished.context,'chatUi.pending'),null);
 assert.equal(read(finished.context,'chatUi.snapshot.status'),'unloaded','Reply recovery must not replace the live worker status');
 assert.equal(read(finished.context,'chatUi.conversations["r:adapter"][1].content'),'Final answer');

 const network=setup();
 vm.runInContext(`state.page='chat';chatUi.error='Keep the action error'`,network.context);
 network.context.document.querySelector=()=>null;
 network.context.api=async()=>{throw Error('Local server unavailable');};
 await vm.runInContext('chatPoll()',network.context);
 assert.equal(read(network.context,'chatUi.connectionError'),'Local server unavailable');
 network.context.api=async()=>({chat:{status:'unloaded',instance_id:'session',revision:1}});
 await vm.runInContext('chatPoll()',network.context);
 assert.equal(read(network.context,'chatUi.connectionError'),'','A healthy poll must clear a temporary connection error');
 assert.equal(read(network.context,'chatUi.error'),'Keep the action error','Successful polling must preserve actionable errors');

 const stopped=setup({runId:'r',pending:{key:'r:adapter',user:'Try again',requestId:'stop-request'}});
 vm.runInContext(`acceptChatSnapshot({status:'ready',request_id:'stop-request',text:'',finish_reason:'cancelled'})`,stopped.context);
 assert.equal(stopped.input.value,'Try again');
 assert.match(read(stopped.context,'chatUi.error'),/Reply stopped/);
 console.log('Chat UI checks passed: safe persistence, submission and session recovery, request limits, reasoning, escaping and connection recovery.');
})().catch(error=>{console.error(error);process.exitCode=1;});

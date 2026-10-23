/* Run with node tests/test_chat_ui.js. No browser or npm dependencies. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const context=vm.createContext({
 document:{addEventListener(){}},state:{page:'overview'},
 localStorage:{getItem:()=>JSON.stringify({maxTokens:null,reasoning:true}),setItem(){}},
 clearTimeout(){},setTimeout(){throw Error('Off-page chat must not poll');},
 icon:()=>'',fmt:String,esc:value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'),
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/chat.js'),'utf8'),context);
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
console.log('Chat UI checks passed: saved options, reasoning disclosure, escaping and incomplete-history handling.');

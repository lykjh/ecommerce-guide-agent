import assert from 'node:assert/strict';
import test from 'node:test';
import {page,json,answerFrames} from '../../http_ui.test.mjs';
const notice={type:'mall:identity-changed',version:1};
function reset(ui) { ui.window.handlers.message({origin:'https://shop.test',source:ui.window.parent,data:notice}); }

test('late create response cannot restore old token; new generation stays busy',async()=>{
 let release;
 const old=new Promise(r=>{release=r;});
 let releaseNew;
 const fresh=new Promise(r=>{releaseNew=r;});
 const ui=page([()=>old,()=>fresh,new Response(answerFrames('new'))]);
 const first=ui.submit();
 reset(ui);
 const second=ui.submit();
 release(json({session_token:'old'},201));await first;
 assert.equal(ui.storage.has('xiaonuan-session'),false);
 assert.equal(ui.elements.get('#send').disabled,true);
 releaseNew(json({session_token:'new'},201));await second;
 assert.equal(ui.storage.get('xiaonuan-session'),'new');
 assert.equal(ui.requests.length,3);
 assert.equal(ui.requests[2].url.href,'https://shop.test/assistant/api/chat');
 assert.equal(ui.requests[2].options.credentials,'same-origin');
});

test('identity change during animation clears old text and suppresses stale catch/finally',async()=>{
 const ui=page([new Response(answerFrames('旧内容不应继续'))],'old',{paced:true});
 const running=ui.submit();await ui.firstTick;
 reset(ui);ui.ticks.shift()();await running;
 assert.equal(ui.elements.get('#messages').children.length,0);
 assert.equal(ui.storage.has('xiaonuan-session'),false);
 assert.match(ui.elements.get('#status').textContent,/身份已变化/);
 assert.equal(ui.channels[0].sent.length,1);
});

test('foreign notifications ignored and channel receive does not rebroadcast',()=>{
 const ui=page([],'old');
 ui.window.handlers.message({origin:'https://evil.test',source:ui.window.parent,data:notice});
 ui.window.handlers.message({origin:'https://shop.test',source:{},data:notice});
 assert.equal(ui.storage.get('xiaonuan-session'),'old');
 ui.channels[0].receive({data:notice});
 assert.equal(ui.storage.has('xiaonuan-session'),false);
 assert.equal(ui.channels[0].sent.length,0);
});

test('409 resets; 503 preserves conversation and token; neither retries',async()=>{
 for(const [status,code] of [[409,'SESSION_IDENTITY_CHANGED'],[503,'IDENTITY_UNAVAILABLE']]){
  const ui=page([json({error:{code,message:'failure'}},status)],'old');await ui.submit();
  assert.equal(ui.storage.has('xiaonuan-session'),status===503);
  assert.equal(ui.elements.get('#messages').children.length,status===503?2:0);
  assert.equal(ui.requests.length,1);assert.equal(ui.channels[0].sent.length,0);
 }
});

test('manual new chat resets locally without allocating a server record',()=>{
 const ui=page([],'old');ui.elements.get('#new-session').handlers.click();
 assert.equal(ui.requests.length,0);assert.equal(ui.storage.has('xiaonuan-session'),false);
});

test('root deployment also derives API from script URL',async()=>{
 const ui=page([json({session_token:'new'},201),new Response(answerFrames('ok'))],null,{moduleUrl:'https://shop.test/ui/chat.js'});
 await ui.submit();
 assert.equal(ui.requests[0].url.href,'https://shop.test/api/sessions');
 assert.equal(ui.requests[1].url.href,'https://shop.test/api/chat');
});

test('two open pages reset through one broadcast with no loop',()=>{
 const one=page([],'one');const two=page([],'two');
 reset(one);
 two.channels[0].receive({data:one.channels[0].sent[0]});
 assert.equal(one.storage.size,0);assert.equal(two.storage.size,0);
 assert.equal(one.channels[0].sent.length,1);assert.equal(two.channels[0].sent.length,0);
});

// Execute the actual inspector event handlers without a network or browser service.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map(), requests = [], downloads = [], timers = [];
let failure = false, tickStatus = 'done';
const element = id => {
  if (!elements.has(id)) elements.set(id, {handlers: {}, checked: false, value: '',
    addEventListener(name, fn) { this.handlers[name] = fn; },
    classList: {toggle() {}}, click() {downloads.push(this);}});
  return elements.get(id);
};
const snapshot = (status = 'ready') => ({
  decision_provider: 'openai', decision_model: 'offline', text_model: 'helper', status,
  goal: 'Task', plan: ['<script>evil</script>'], plan_index: 0, elapsed_ms: 2, max_steps: 2,
  page: {screenshot: 'pixels', fingerprint: 'observed', url: 'https://fixture.invalid', title: 'Fixture',
    text: 'Fixture', w: 100, h: 100, actions: [{id:'e1', node:1, label:'<img onerror=evil>', rect:{x:1,y:1,w:10,h:10}}]},
  elements: [{index:'1', label:'<img onerror=evil>', role:'button', operations:['CLICK']}],
  history: [{step:1, action:'<script>bad</script>', text:'<b>value</b>', text_helper:'helper',
    latency_ms:1, probability:1, page_changed:true}], decisions: [],
  decision: status === 'predicted' ? {choice:'e1', operation:'CLICK', target:'1', latency_ms:1,
    target_probabilities:{'1':1}, operation_probabilities:{CLICK:1}, confidence:1} : null,
});
let current = snapshot('ready');
const context = vm.createContext({
  document: {querySelector: () => ({content:'offline-token'}), getElementById: element,
    querySelectorAll: () => [], createElement: () => element('anchor')},
  Blob: class {constructor(parts) {this.parts=parts; downloads.push(this);}},
  URL: {createObjectURL: () => 'blob:offline', revokeObjectURL() {}},
  setTimeout(fn) { timers.push(fn); },
  async fetch(url, options) {
    requests.push({url, options});
    if (options && failure) return {ok:false,json:async()=>({error:'controlled failure'})};
    if (url.endsWith('/predict')) current = snapshot('predicted');
    else if (url.endsWith('/act')) current = snapshot('ready');
    else if (url.endsWith('/tick')) current = snapshot(tickStatus);
    return {ok:true,json:async()=>current};
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../jev_ultrafast/static/app.js'),'utf8'), context);
const settle = () => new Promise(setImmediate);
const click = async id => {await element(id).handlers.click({}); await settle();};
(async () => {
  await settle();
  assert.equal(element('execute').disabled,true);
  await click('choose');
  assert.equal(element('execute').disabled,false);
  assert.equal(element('confidence').textContent,'Only target');
  await click('execute');
  assert.equal(JSON.parse(requests.at(-1).options.body).fingerprint,'observed');
  assert.equal(requests.at(-1).options.headers['X-Demo-Token'],'offline-token');
  assert(!element('plan').innerHTML.includes('<script>'));
  assert(!element('choices').innerHTML.includes('<img'));
  assert(!element('history').innerHTML.includes('<script>'));
  await click('download');
  const exported = JSON.parse(downloads[0].parts[0]);
  assert.equal(exported.page.screenshot,undefined);
  assert.equal(exported.history.length,1);
  await click('auto');
  assert.equal(element('choose').disabled,true);
  assert.equal(element('auto').disabled,true);
  context.current = snapshot('ready');
  vm.runInContext('state=current; render()',context);
  element('pace').checked=true;
  const running=click('auto');
  await settle();
  assert.equal(timers.length,1);
  await click('stop');
  const count=requests.filter(r=>r.url.endsWith('/act')).length;
  timers.shift()(); await running;
  assert.equal(requests.filter(r=>r.url.endsWith('/act')).length,count);
  failure=true; await click('choose');
  assert.equal(element('error').hidden,false);
  assert.equal(element('error').textContent,'controlled failure');
  assert.equal(element('start').disabled,false);
  element('overlays').checked=false; element('overlays').handlers.change();
  assert.equal(element('targets').hidden,true);
  element('scenario').value='research'; element('scenario').handlers.change();
  assert(element('goal').value.includes('finite choices'));
  failure=false; element('task-form').handlers.submit({preventDefault(){}}); await settle();
  assert.equal(requests.at(-1).url,'/api/reset');
  console.log('PASS: inspector actions, terminal controls, paced stop, recovery, escaping, overlays, trace export');
})().catch(error=>{console.error(error);process.exitCode=1;});

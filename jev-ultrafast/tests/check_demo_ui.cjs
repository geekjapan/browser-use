// Offline inspector contract: configured provider/model reaches the visible UI.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const context = vm.createContext({
  document: {
    querySelector: () => ({ content: 'test-token' }),
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { addEventListener() {} });
      return elements.get(id);
    },
  },
  fetch: () => new Promise(() => {}),
});
vm.runInContext(fs.readFileSync('jev_ultrafast/static/app.js', 'utf8'), context);
for (const provider of ['azure', 'openai']) {
  vm.runInContext(`state = ${JSON.stringify({
    decision_provider: provider, decision_model: 'configured-model', text_model: 'independent-helper',
    status: 'idle', page: null,
  })}; render();`, context);
  assert.equal(elements.get('decision-model').textContent, `${provider} · configured-model`);
  assert.equal(context.document.title, `${provider} · configured-model · Browser Use`);
  assert.equal(elements.get('helper').textContent, 'Text helper · independent-helper');
}

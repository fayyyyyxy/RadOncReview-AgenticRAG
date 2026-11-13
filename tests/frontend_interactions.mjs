// Run with `node tests/frontend_interactions.mjs`. Pure interaction/state tests; no browser or network.
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

export async function runFrontendTests() {
  const html = await readFile(new URL('../static/index.html', import.meta.url), 'utf8');
  const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
  const nodes = new Map();
  const node = () => ({value: '', style: {}, textContent: '', _html: '', children: [],
    set innerHTML(value) { this._html = value; this.children = []; }, get innerHTML() { return this._html; },
    classList: {toggle() {}, add() {}, remove() {}},
    appendChild(child) { this.children.push(child); },
    setAttribute() {}, addEventListener(name, fn) { this[name] = fn; },
    focus() {}, remove() {}});
  const document = {body: node(), getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  }, createElement: node};
  const storage = new Map();
  const sandbox = {document, TextDecoder, TextEncoder, Date, console, URL, performance,
    marked: {setOptions() {}, parse: value => value},
    window: {innerWidth: 1000}, requestAnimationFrame: fn => fn(),
    localStorage: {setItem: (key, value) => storage.set(key, value),
      getItem: key => storage.get(key), removeItem: key => storage.delete(key)}};
  vm.createContext(sandbox);
  vm.runInContext(script.replace('const app = new ChatApp();', 'globalThis.ChatApp = ChatApp;'), sandbox);
  const makeApp = () => Object.assign(Object.create(sandbox.ChatApp.prototype), {
    messages: [], pending: new Map(), drafts: new Map(), conversations: {}, categories: ['Breast', 'CNS'],
    selectedSite: 'Not specified', currentConvId: 'initial',
    caseContext: '', sourceRegistry: new Map(),
    STORAGE_KEY: 'conversations', CURRENT_KEY: 'current', userScrolled: false,
    renderHistory() {}, renderWelcome() {}, renderUserBubble() {},
    renderAssistantBubble() {}, showThinking() {}, removeThinking() {}, scrollToBottom() {},
    showToast(text) { this.toast = text; },
  });

  const contextApp = makeApp();
  for (let i = 0; i < 4; i++) contextApp.messages.push(
    {role: 'user', content: `Question ${i}`},
    {role: 'assistant', data: {answer: 'Generated text must not be reused as evidence'}});
  assert.equal(JSON.stringify(contextApp.answeredQuestions()), JSON.stringify(['Question 0', 'Question 1', 'Question 2', 'Question 3']));
  contextApp.selectedSite = 'Breast';
  assert.equal(contextApp.answeredQuestions().length, 4, 'Search scope changes must not discard case corrections');
  contextApp.conversations.initial = {messages: contextApp.messages};
  contextApp.restoreConversation('initial');
  assert.match(contextApp.caseContext, /Question 1/, 'Legacy migration preserves intermediate corrections');
  contextApp.editContext('Corrected stage: II');
  contextApp.selectScope('Breast');
  assert.equal(contextApp.caseContext, 'Corrected stage: II');
  contextApp.restoreConversation('initial');
  assert.equal(contextApp.caseContext, 'Corrected stage: II', 'Edits survive chat restoration');
  contextApp.messages = [{role: 'user', content: 'Blocked input'},
    {role: 'assistant', data: {metadata: {input_blocked: true}}}];
  assert.equal(contextApp.answeredQuestions().length, 0, 'Blocked input is not carried forward');

  const categoryApp = makeApp();
  categoryApp.renderCategories();
  const categoryButtons = document.getElementById('categoryTags').children;
  categoryButtons.find(button => button.textContent === 'Breast').click();
  assert.equal(categoryApp.selectedSite, 'Breast');
  assert.equal(categoryApp.messages.length, 0, 'Category selection must not submit an overview');

  const wire = [
    {type: 'progress', stage: 'evaluate'},
    {type: 'result', data: {answer: 'Synthetic checked answer — complete'}},
  ].map(event => JSON.stringify(event) + '\n').join('');
  const bytes = new TextEncoder().encode(wire);
  // Split UTF-8 and JSON records across reads rather than assuming network chunk boundaries.
  const chunks = Array.from(bytes, byte => new Uint8Array([byte]));
  let requestBody;
  sandbox.fetch = async (url, options) => {
    assert.equal(url, '/query/stream');
    requestBody = JSON.parse(options.body);
    return {ok: true, body: {getReader: () => ({
      async read() { return chunks.length ? {value: chunks.shift(), done: false} : {done: true}; }
    })}};
  };
  const progress = [];
  const result = await makeApp().callAPI('Current question', 'Breast', ['Initial question'], stage => progress.push(stage));
  assert.equal(result.answer, 'Synthetic checked answer — complete');
  assert.deepEqual(progress, ['evaluate']);
  assert.equal(requestBody.filter_by_site, true);
  assert.deepEqual(requestBody.prior_questions, ['Initial question']);
  assert.equal(requestBody.run_evaluation, false, 'Extra assessments must be off by default');

  const warningApp = makeApp();
  const flaggedHTML = warningApp.renderEvidenceWarning({metadata: {evidence_check: {
    status: 'completed', flagged: true, judgment: 'Should Have Abstained', reason: '<Synthetic evidence gap>'
  }}});
  assert.match(flaggedHTML, /Insufficient evidence flagged/);
  assert.match(flaggedHTML, /&lt;Synthetic evidence gap&gt;/);
  assert.equal(warningApp.renderEvidenceWarning({metadata: {evidence_check: {status: 'completed', flagged: false}}}), '');
  assert.match(warningApp.renderEvidenceWarning({metadata: {evidence_check: {status: 'unavailable'}}}), /Evidence check unavailable/);
  assert.match(warningApp.renderEvidenceWarning({metadata: {evidence_check: {
    status: 'completed', flagged: true, judgment: 'Partial Abstention Preferred'
  }}}), /Some statements may lack sufficient evidence/);

  const errorApp = makeApp();
  errorApp.addErrorMessage = function (text) { this.visibleError = text; };
  let rejectRequest;
  errorApp.callAPI = () => new Promise((resolve, reject) => { rejectRequest = reject; });
  document.getElementById('queryInput').value = 'Original synthetic question';
  const failed = errorApp.send();
  errorApp.newChat();
  const newId = errorApp.currentConvId;
  rejectRequest(new Error('Synthetic request failure'));
  await failed;
  assert.equal(errorApp.currentConvId, newId);
  assert.equal(errorApp.messages.length, 0, 'An old error must not appear in a new chat');
  assert.equal(errorApp.visibleError, undefined);
  assert.equal(errorApp.conversations.initial.messages.at(-1).role, 'error');
  assert.equal(errorApp.drafts.get('initial'), 'Original synthetic question');
  assert.equal(errorApp.pending.size, 0);

  const successApp = makeApp();
  let finishRequest;
  successApp.callAPI = () => new Promise(resolve => { finishRequest = resolve; });
  document.getElementById('queryInput').value = 'Another synthetic question';
  const succeeded = successApp.send();
  successApp.newChat();
  finishRequest({answer: 'Synthetic answer'});
  await succeeded;
  assert.equal(successApp.messages.length, 0, 'An old answer must not appear in a new chat');
  assert.equal(successApp.conversations.initial.messages.at(-1).data.answer, 'Synthetic answer');
  assert.equal(successApp.conversations.initial.caseContext, 'Another synthetic question', 'Answered user context stays in its originating chat');

  const editedApp = makeApp();
  let finishEdited;
  editedApp.callAPI = () => new Promise(resolve => { finishEdited = resolve; });
  document.getElementById('queryInput').value = 'Original stage I?';
  const edited = editedApp.send();
  editedApp.editContext('Correction: stage II');
  finishEdited({answer: 'Synthetic answer'});
  await edited;
  assert.equal(editedApp.caseContext, 'Correction: stage II', 'A late result cannot overwrite an edit');

  const fullApp = makeApp();
  fullApp.caseContext = 'x'.repeat(4001);
  document.getElementById('queryInput').value = 'Keep this question';
  await fullApp.send();
  assert.equal(fullApp.messages.length, 0);
  assert.equal(document.getElementById('queryInput').value, 'Keep this question');
  assert.equal(fullApp.caseContext.length, 4001, 'Over-limit context must be reviewed, not truncated');

  const broadApp = makeApp();
  broadApp.caseContext = 'Newer corrected case';
  broadApp.selectedSite = 'Breast';
  broadApp.sourceRegistry.set('old-answer', {query: 'Original question', client_request: {context: 'Original case'}});
  let broadRequest;
  broadApp.callAPI = async (q, site, prior, progress, review, context) => {
    broadRequest = {q, site, context}; return {query:q, answer:'Synthetic answer'};
  };
  document.getElementById('queryInput').value = 'Unsent draft';
  await broadApp.searchAllSources('old-answer');
  assert.deepEqual(broadRequest, {q:'Original question', site:'Not specified', context:'Original case'});
  assert.equal(broadApp.caseContext, 'Newer corrected case');
  assert.equal(document.getElementById('queryInput').value, 'Unsent draft');
  assert.equal(broadApp.selectedSite, 'Not specified');
  assert.equal(broadApp.safeSourceURL('javascript:alert(1)'), null);
  assert.equal(broadApp.safeSourceURL('https://example.com/paper'), 'https://example.com/paper');

  const longApp = makeApp();
  document.getElementById('queryInput').value = 'x'.repeat(2001);
  await longApp.send();
  assert.equal(longApp.messages.length, 0);
  assert.equal(document.getElementById('queryInput').value.length, 2001, 'Long questions must not be silently truncated');

  return 'Passed: editable context/migration/corrections, search scope, exact broader-search replay, streamed JSON/UTF-8, chat-switch results/errors, draft preservation, default-off quality review, safe source links, and evidence warnings.';
}

// Also support execution as a standalone Node script.
if (import.meta.url === `file://${globalThis.process?.argv?.[1]}`) console.log(await runFrontendTests());

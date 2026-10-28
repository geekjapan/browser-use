"""Offline contract and execution checks for the finite-choice agent."""

import copy
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from browser_use import AzureMicrosoftDecisionClient, MicrosoftDecisionAgent
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.decision import ChoiceAnswer, ChoiceQuestion, DecisionRequest, DecisionResponse, FieldText
from browser_use.dom.serializer.serializer import DOMTreeSerializer
from browser_use.dom.views import (
	DOMRect,
	EnhancedAXNode,
	EnhancedAXProperty,
	EnhancedDOMTreeNode,
	EnhancedSnapshotNode,
	NodeType,
	SerializedDOMState,
)
from browser_use.llm.base import BaseChatModel
from browser_use.llm.views import ChatInvokeCompletion
from browser_use.tools.service import Tools


def node(index=1, tag='button', attributes=None, children=None):
	return EnhancedDOMTreeNode(
		node_id=index,
		backend_node_id=index,
		node_type=NodeType.ELEMENT_NODE,
		node_name=tag,
		node_value='',
		attributes=attributes or {},
		is_scrollable=False,
		is_visible=True,
		absolute_position=None,
		target_id='target',
		frame_id='frame',
		session_id='session',
		content_document=None,
		shadow_root_type=None,
		shadow_roots=None,
		parent_node=None,
		children_nodes=children,
		ax_node=None,
		snapshot_node=None,
	)


def state(nodes=None):
	return BrowserStateSummary(
		dom_state=SerializedDOMState(_root=None, selector_map=nodes or {1: node()}),
		url='https://example.test',
		title='Test',
		tabs=[],
	)


def answer(choice, candidates):
	return {
		'type': 'choice',
		'choice': choice,
		'confidence': 0.8,
		'probabilities': {c: 1.0 if c == choice else 0.0 for c in candidates},
	}


class ScriptedDecision:
	def __init__(self, operation, target=None):
		self.operation = operation
		self.target = target
		self.requests = []

	async def predict(self, request):
		self.requests.append(request)
		answers = {'operation': answer(self.operation, request.questions['operation'].criteria)}
		if self.target is not None:
			head = self.operation + '_target'
			answers[head] = answer(self.target, request.questions[head].criteria)
		return DecisionResponse.model_validate({'model': 'microsoft-decision-1', 'answers': answers})


def agent(operation='click', nodes=None, text_llm=None, target=None):
	browser = AsyncMock(spec=BrowserSession)
	browser.agent_focus_target_id = 'target'
	browser.get_browser_state_summary.return_value = state(nodes)
	decision = ScriptedDecision(operation, target)
	result = MicrosoftDecisionAgent('Complete the test', decision, browser=browser, text_llm=text_llm)
	result.tools.act = AsyncMock(return_value=ActionResult(extracted_content='ok'))
	return result, browser, decision


@pytest.mark.parametrize('auth', ['key', 'token'])
async def test_http_contract_and_auth(monkeypatch, auth):
	monkeypatch.delenv('AZURE_API_KEY', raising=False)
	monkeypatch.delenv('AZURE_ENTRA_TOKEN', raising=False)

	def handle(request):
		assert str(request.url) == 'https://resource.test/providers/microsoft/v1/systemone'
		assert request.headers.get('api-key') == ('key-secret' if auth == 'key' else None)
		assert request.headers.get('Authorization') == ('Bearer token-secret' if auth == 'token' else None)
		body = json.loads(request.content)
		assert body['model'] == 'decision-1'
		assert body['questions']['operation'] == {
			'type': 'choice',
			'instructions': 'Choose',
			'criteria': {'DONE': 'Done', 'BLOCKED': 'Blocked'},
		}
		return httpx.Response(
			200, json={'model': 'microsoft-decision-1', 'answers': {'operation': answer('DONE', ['DONE', 'BLOCKED'])}}
		)

	client = AzureMicrosoftDecisionClient(
		'https://resource.test/',
		api_key='key-secret' if auth == 'key' else None,
		entra_token='token-secret' if auth == 'token' else None,
		transport=httpx.MockTransport(handle),
	)
	request = DecisionRequest(
		model=client.deployment,
		state={'url': 'local'},
		questions={'operation': ChoiceQuestion(instructions='Choose', criteria={'DONE': 'Done', 'BLOCKED': 'Blocked'})},
	)
	assert (await client.predict(request)).model == 'microsoft-decision-1'


@pytest.mark.parametrize('status', [302, 400, 401, 429, 503])
async def test_provider_errors_are_sanitized_and_not_retried(status):
	calls = []

	def handle(request):
		calls.append(request)
		return httpx.Response(status, text='secret provider error')

	client = AzureMicrosoftDecisionClient('https://resource.test', api_key='secret', transport=httpx.MockTransport(handle))
	request = DecisionRequest(model='decision-1', state={}, questions={})
	with pytest.raises(RuntimeError) as error:
		await client.predict(request)
	assert 'secret' not in str(error.value)
	assert len(calls) == 1


@pytest.mark.parametrize(
	'endpoint',
	['http://resource.test', 'https://u:p@resource.test', 'https://resource.test/path', 'https://resource.test?secret=x'],
)
def test_endpoint_validation(endpoint):
	with pytest.raises(ValueError):
		AzureMicrosoftDecisionClient(endpoint, api_key='secret')


@pytest.mark.parametrize(
	'change',
	[
		{'choice': 'invented'},
		{'confidence': True},
		{'confidence': float('nan')},
		{'probabilities': {'a': 1.0}},
		{'probabilities': {'a': 0.1, 'b': 0.9}},
		{'probabilities': {'a': 1.0, 'b': float('nan')}},
		{'type': 'score'},
	],
)
def test_reject_malformed_choice(change):
	value = answer('a', ['a', 'b']) | change
	with pytest.raises(ValueError, match='Invalid Decision'):
		ChoiceAnswer.validate_for(value, {'a': 'A', 'b': 'B'})


async def test_singleton_native_click_and_borrowed_browser():
	a, browser, model = agent()
	assert (await a.run(max_steps=1)).status == 'max_steps'
	assert 'click_target' not in model.requests[0].questions
	action = cast(AsyncMock, a.tools.act).call_args.args[0]
	assert action.model_dump(exclude_unset=True) == {'click': {'index': 1}}
	browser.stop.assert_not_awaited()


async def test_only_selected_target_head_is_consumed():
	a, _, _ = agent('click', {1: node(), 2: node(2)}, target='2')
	assert (await a.run(1)).steps[0].index == 2
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {'click': {'index': 2}}


@pytest.mark.parametrize('change', ['url', 'identity', 'dom', 'focus'])
async def test_stale_observation_never_acts(change):
	a, browser, _ = agent()
	before = state()
	after = copy.deepcopy(before)
	if change == 'url':
		after.url += '/changed'
	elif change == 'identity':
		after.dom_state.selector_map[1].backend_node_id = 900
	elif change == 'dom':
		after.dom_state.selector_map[1].attributes['aria-label'] = 'Changed'
	else:

		async def capture(**kwargs):
			browser.agent_focus_target_id = 'target' if browser.get_browser_state_summary.await_count == 1 else 'new-target'
			return before

		browser.get_browser_state_summary.side_effect = capture
	if change != 'focus':
		browser.get_browser_state_summary.side_effect = [before, after]
	assert (await a.run(1)).status == 'stale_state'
	cast(AsyncMock, a.tools.act).assert_not_awaited()


async def test_input_helper_is_lazy_strict_and_rechecks_state():
	llm = AsyncMock(spec=BaseChatModel)
	llm.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text='Ada'), usage=None)
	a, _, _ = agent('input', {1: node(tag='input', attributes={'name': 'name'})}, llm)
	await a.run(1)
	assert llm.ainvoke.call_args.kwargs['output_format'] is FieldText
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {
		'input': {'index': 1, 'text': 'Ada', 'clear': True}
	}
	llm.ainvoke.reset_mock()
	a, _, _ = agent('click', text_llm=llm)
	await a.run(1)
	llm.ainvoke.assert_not_awaited()


async def test_invalid_helper_output_never_types():
	llm = AsyncMock(spec=BaseChatModel)
	llm.ainvoke.return_value = ChatInvokeCompletion.model_construct(completion={'text': 'Ada', 'extra': 'invalid'}, usage=None)
	a, _, _ = agent('input', {1: node(tag='input')}, llm)
	with pytest.raises(ValueError, match='Text helper'):
		await a.run(1)
	cast(AsyncMock, a.tools.act).assert_not_awaited()


async def test_slow_helper_stale_state_never_types():
	llm = AsyncMock(spec=BaseChatModel)
	llm.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text='Ada'), usage=None)
	a, browser, _ = agent('input', {1: node(tag='input')}, llm)
	before = browser.get_browser_state_summary.return_value
	after = copy.deepcopy(before)
	after.url += '/changed'
	browser.get_browser_state_summary.side_effect = [before, after]
	assert (await a.run(1)).status == 'stale_state'
	cast(AsyncMock, a.tools.act).assert_not_awaited()


def test_secret_fields_and_missing_helper_excluded():
	a, _, _ = agent()
	assert not a._targets(state({1: node(tag='input')}))
	a, _, _ = agent(text_llm=AsyncMock(spec=BaseChatModel))
	assert not a._targets(state({1: node(tag='input', attributes={'type': 'password'})}))
	assert not a._targets(state({1: node(tag='input', attributes={'autocomplete': 'cc-number'})}))


async def test_native_select_mapping():
	option = node(2, 'option', {'value': 'Blue'})
	a, _, _ = agent('select', {1: node(tag='select', children=[option])})
	await a.run(1)
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {
		'select_dropdown': {'index': 1, 'text': 'Blue'}
	}


@pytest.mark.parametrize(
	'operation,expected',
	[
		('scroll_up', {'scroll': {'down': False, 'pages': 1.0}}),
		('wait', {'wait': {'seconds': 2}}),
	],
)
async def test_native_nonindexed_mapping(operation, expected):
	a, _, _ = agent(operation)
	await a.run(1)
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == expected


async def test_action_failure_stops_without_retry():
	a, _, model = agent()
	cast(AsyncMock, a.tools.act).return_value = ActionResult(error='Failed')
	assert (await a.run(10)).status == 'action_error'
	assert len(model.requests) == 1
	cast(AsyncMock, a.tools.act).assert_awaited_once()


@pytest.mark.parametrize('missing_node', [False, True])
async def test_native_dropdown_failure_stops_without_retry(missing_node):
	a, browser, model = agent('select', {1: node(tag='select', children=[option(2, 'Blue', 'blue')])})
	a.tools = Tools()
	browser.get_element_by_index.return_value = None if missing_node else node(tag='select')
	event = SimpleNamespace(
		event_result=AsyncMock(
			return_value={
				'success': 'false',
				'error': 'Selection was reverted by the page.',
				'short_term_memory': 'Available options: Blue',
				'long_term_memory': 'Could not select Blue',
			}
		)
	)
	dispatch = Mock(return_value=event)
	browser.event_bus = SimpleNamespace(dispatch=dispatch)
	assert (await a.run(2)).status == 'action_error'
	assert len(model.requests) == 1
	assert dispatch.call_count == (0 if missing_node else 1)


@pytest.mark.parametrize('operation', ['click', 'input'])
async def test_native_missing_target_stops_without_retry(operation):
	llm = AsyncMock(spec=BaseChatModel)
	llm.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text='Ada'), usage=None)
	a, browser, model = agent(operation, {1: node(tag='input' if operation == 'input' else 'button')}, llm)
	a.tools = Tools()
	browser.get_element_by_index.return_value = None
	assert (await a.run(2)).status == 'action_error'
	assert len(model.requests) == 1


async def test_native_scroll_failure_stops_without_retry():
	a, browser, model = agent('scroll_down')
	a.tools = Tools()
	browser.get_or_create_cdp_session.side_effect = RuntimeError('No viewport metrics')

	class FailedScroll:
		def __await__(self):
			async def fail():
				raise RuntimeError('Scroll failed after dispatch')

			return fail().__await__()

	dispatch = Mock(return_value=FailedScroll())
	browser.event_bus = SimpleNamespace(dispatch=dispatch)
	assert (await a.run(2)).status == 'action_error'
	assert len(model.requests) == 1
	dispatch.assert_called_once()


async def test_native_wait_actually_sleeps_one_second(monkeypatch):
	a, _, _ = agent('wait')
	a.tools = Tools()
	sleep = AsyncMock()
	monkeypatch.setattr('browser_use.tools.service.asyncio.sleep', sleep)
	assert (await a.run(1)).status == 'max_steps'
	sleep.assert_awaited_once_with(1)


def test_freshness_ignores_new_node_markers_but_keeps_real_text():
	a, _, _ = agent()

	def capture(index, previous=None, text='Ready'):
		child = node(index + 100, '#text')
		child.node_type = NodeType.TEXT_NODE
		child.node_value = text
		button = node(index, children=[child])
		button.snapshot_node = EnhancedSnapshotNode(
			True, 'pointer', DOMRect(0, 0, 100, 20), DOMRect(0, 0, 100, 20), None, {}, 0, 0
		)
		serialized, _ = DOMTreeSerializer(
			button, previous_cached_state=previous, enable_bbox_filtering=False, paint_order_filtering=False
		).serialize_accessible_elements()
		result = state()
		result.dom_state = serialized
		return result

	old = capture(1)
	first = capture(2, old.dom_state)
	second = capture(2, first.dom_state)
	assert first.dom_state.llm_representation() != second.dom_state.llm_representation()
	assert a._fingerprint(first) == a._fingerprint(second)
	assert '*' in first.dom_state.llm_representation()  # Fingerprinting must not mutate observations.
	changed = capture(2, second.dom_state, text='Changed')
	assert a._fingerprint(second) != a._fingerprint(changed)


async def test_done_is_declaration_and_can_be_independently_verified():
	a, browser, _ = agent('DONE')
	assert (await a.run(1)).verified is None
	verify = AsyncMock(return_value=False)
	result = await a.run(1, outcome_verifier=verify)
	assert result.status == 'model_done' and result.verified is False
	verify.assert_awaited_once_with(browser)
	cast(AsyncMock, a.tools.act).assert_not_awaited()


def test_context_limit_and_candidate_limit():
	a, browser, _ = agent()
	large = state({i: node(i) for i in range(1, 400)})
	assert 0 < len(a._targets(large)['click']) <= 250
	large.title = 'x' * 30000
	request = a._request(large, a._targets(large), [])
	assert len(request.model_dump_json().encode()) <= 28000
	assert len(str(request.state['title'])) == 500


async def test_option_value_change_is_stale_even_with_same_label():
	before = state({1: node(tag='select', children=[node(2, 'option', {'value': 'Blue'})])})
	after = copy.deepcopy(before)
	options = after.dom_state.selector_map[1].children_nodes
	assert options is not None
	options[0].attributes['value'] = 'Red'
	a, browser, _ = agent('select')
	browser.get_browser_state_summary.side_effect = [before, after]
	assert (await a.run(1)).status == 'stale_state'
	cast(AsyncMock, a.tools.act).assert_not_awaited()


async def test_malformed_http_response_is_sanitized():
	client = AzureMicrosoftDecisionClient(
		'https://resource.test',
		api_key='secret',
		transport=httpx.MockTransport(lambda request: httpx.Response(200, text='raw secret response')),
	)
	with pytest.raises(ValueError, match='Invalid Decision 1 response') as error:
		await client.predict(DecisionRequest(model='decision-1', state={}, questions={}))
	assert 'secret' not in str(error.value)


async def test_unused_target_answer_does_not_block_selected_operation():
	a, _, model = agent('wait', {1: node(), 2: node(2)})

	async def predict(request):
		response = await ScriptedDecision.predict(model, request)
		response.answers['click_target'] = {'bad': 'unused answer'}
		return response

	model.predict = predict
	assert (await a.run(1)).steps[0].operation == 'wait'


async def test_connection_error_and_missing_auth_are_sanitized(monkeypatch):
	monkeypatch.delenv('AZURE_API_KEY', raising=False)
	monkeypatch.delenv('AZURE_ENTRA_TOKEN', raising=False)
	with pytest.raises(ValueError, match='AZURE_API_KEY or AZURE_ENTRA_TOKEN'):
		AzureMicrosoftDecisionClient('https://resource.test')

	def fail(request):
		raise httpx.ConnectError('secret error', request=request)

	client = AzureMicrosoftDecisionClient('https://resource.test', api_key='secret', transport=httpx.MockTransport(fail))
	with pytest.raises(RuntimeError, match='connection failed') as error:
		await client.predict(DecisionRequest(model='decision-1', state={}, questions={}))
	assert 'secret' not in str(error.value)


def test_long_dom_is_bounded_without_changing_target_ids(monkeypatch):
	a, _, _ = agent()
	observed = state({7: node(7)})
	monkeypatch.setattr(observed.dom_state, 'llm_representation', lambda: 'visible text ' * 100000)
	targets = a._targets(observed)
	request = a._request(observed, targets, [])
	assert len(request.model_dump_json().encode()) <= 28000
	assert request.state['dom']
	assert list(targets['click']) == ['7']
	assert request.state['sole_targets'] == {'click': {'7': targets['click']['7'].description}}


async def test_owned_browser_stops_even_after_prediction_failure(monkeypatch):
	import browser_use.decision as decision_module

	browser = AsyncMock(spec=BrowserSession)
	browser.agent_focus_target_id = 'target'
	browser.get_browser_state_summary.return_value = state()
	monkeypatch.setattr(decision_module, 'BrowserSession', lambda: browser)
	model = AsyncMock()
	model.deployment = 'decision-1'
	model.predict.side_effect = RuntimeError('offline')
	a = MicrosoftDecisionAgent('Test', model)
	with pytest.raises(RuntimeError, match='offline'):
		await a.run(1)
	browser.stop.assert_awaited_once()


@pytest.mark.parametrize('selected,expected', [('1', 'Ada'), ('2', 'Lovelace')])
async def test_text_helper_receives_distinct_selected_field_identity(selected, expected):
	first = node(1, 'input', {'name': 'first_name', 'id': 'first', 'type': 'text', 'placeholder': 'Given name'})
	last = node(2, 'input', {'name': 'last_name', 'id': 'last', 'type': 'text', 'aria-label': 'Surname'})
	first.ax_node = EnhancedAXNode('1', False, 'textbox', 'First name', None, None, None)
	last.ax_node = EnhancedAXNode('2', False, 'textbox', 'Last name', None, None, None)
	llm = AsyncMock(spec=BaseChatModel)

	async def generate(messages, output_format):
		context = json.loads(messages[1].content)
		field = context['field']
		assert field['index'] == int(selected)
		description = json.loads(field['description'])
		assert description['index'] == int(selected)
		assert description['label'] == ('First name' if selected == '1' else 'Last name')
		assert description['attributes']['type'] == 'text'
		name = description['attributes']['name']
		return ChatInvokeCompletion(
			completion=output_format(text={'first_name': 'Ada', 'last_name': 'Lovelace'}[name]), usage=None
		)

	llm.ainvoke.side_effect = generate
	a, _, model = agent('input', {1: first, 2: last}, llm, target=selected)
	await a.run(1)
	assert model.requests[0].questions['input_target'].criteria['1'] != model.requests[0].questions['input_target'].criteria['2']
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {
		'input': {'index': int(selected), 'text': expected, 'clear': True}
	}


def option(index, label, value):
	text = node(index + 100, '#text')
	text.node_type = NodeType.TEXT_NODE
	text.node_value = label
	return node(index, 'option', {'value': value}, [text])


@pytest.mark.parametrize(
	'options,expected',
	[
		([option(2, 'Red', 'blue'), option(3, 'Blue', 'cyan')], {'blue', 'cyan'}),
		([option(2, 'Blue', 'blue'), option(3, 'Blue', 'cyan')], {'blue', 'cyan'}),
		([option(2, 'Blue', 'blue'), option(3, 'BLUE', 'BLUE')], {'blue'}),
	],
)
def test_select_match_strings_resolve_to_exact_observed_option(options, expected):
	a, _, _ = agent()
	targets = a._targets(state({1: node(tag='select', children=options)}))['select']
	assert {target.text for target in targets.values()} == expected
	for key, target in targets.items():
		position = int(key.split(':')[1])
		assert target.text is not None
		assert (
			next(
				i
				for i, item in enumerate(options)
				if target.text.lower() in (a._option_text(item).strip().lower(), item.attributes['value'].lower())
			)
			== position
		)


@pytest.mark.parametrize('disabled', [False, True])
def test_select_bom_label_collision_excludes_ambiguous_target(disabled):
	first = option(2, '\ufeffB\ufeff', '\ufeffB\ufeff')
	first.attributes.pop('value')
	if disabled:
		first.attributes['disabled'] = ''
	a, _, _ = agent()
	targets = a._targets(state({1: node(tag='select', children=[first, option(3, 'B', 'b')])})).get('select', {})
	assert '1:1' not in targets
	assert ('1:0' in targets) is (not disabled)


async def test_select_label_value_collision_uses_safe_value_in_native_action():
	a, _, _ = agent(
		'select', {1: node(tag='select', children=[option(2, 'Red', 'blue'), option(3, 'Blue', 'cyan')])}, target='1:1'
	)
	await a.run(1)
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {
		'select_dropdown': {'index': 1, 'text': 'cyan'}
	}


def openai_request():
	return DecisionRequest(
		model='azure-model-is-not-used',
		state={'url': 'https://example.test', 'task': 'Finish'},
		questions={'operation': ChoiceQuestion(instructions='Choose', criteria={'DONE': 'Done', 'BLOCKED': 'Blocked'})},
	)


def openai_answer(name='operation', choice='DONE', candidates=('DONE', 'BLOCKED')):
	value = answer(choice, candidates)
	value['name'] = name
	value['probabilities'] = [{'value': key, 'probability': probability} for key, probability in value['probabilities'].items()]
	return value


async def test_openai_http_contract_and_environment(monkeypatch):
	from browser_use import DecisionAgent, OpenAIDecisionClient

	assert DecisionAgent is MicrosoftDecisionAgent
	monkeypatch.setenv('OPENAI_API_KEY', 'key-secret')
	monkeypatch.setenv('OPENAI_DECISION_MODEL', 'configured-model')

	def handle(request):
		assert str(request.url) == 'https://api.openai.com/v1/decisions'
		assert request.headers['Authorization'] == 'Bearer key-secret'
		body = json.loads(request.content)
		assert body == {
			'model': 'configured-model',
			'input': json.dumps(openai_request().state, ensure_ascii=False, sort_keys=True),
			'questions': [
				{
					'name': 'operation',
					'type': 'choice',
					'instructions': 'Choose',
					'choices': [{'value': 'DONE', 'description': 'Done'}, {'value': 'BLOCKED', 'description': 'Blocked'}],
				}
			],
		}
		return httpx.Response(200, json={'model': 'configured-model', 'answers': [openai_answer()], 'usage': {'total_tokens': 5}})

	client = OpenAIDecisionClient(transport=httpx.MockTransport(handle))
	response = await client.predict(openai_request())
	assert (
		ChoiceAnswer.validate_for(response.answers['operation'], openai_request().questions['operation'].criteria).choice
		== 'DONE'
	)
	monkeypatch.delenv('OPENAI_DECISION_MODEL')
	assert OpenAIDecisionClient().model == 'gpt-6-luna'
	assert OpenAIDecisionClient(model='explicit-model').model == 'explicit-model'
	assert 'key-secret' not in repr(client.__dict__)


@pytest.mark.parametrize('status', [302, 400, 401, 429, 503])
async def test_openai_http_errors_are_sanitized_without_retries(status):
	from browser_use import OpenAIDecisionClient

	calls = []

	def handle(request):
		calls.append(request)
		return httpx.Response(status, text='secret provider response')

	client = OpenAIDecisionClient(api_key='secret', transport=httpx.MockTransport(handle))
	with pytest.raises(RuntimeError) as error:
		await client.predict(openai_request())
	assert 'secret' not in str(error.value)
	assert len(calls) == 1


async def test_openai_missing_auth_network_and_invalid_json(monkeypatch):
	from browser_use import OpenAIDecisionClient

	monkeypatch.delenv('OPENAI_API_KEY', raising=False)
	with pytest.raises(ValueError, match='OPENAI_API_KEY'):
		OpenAIDecisionClient()

	def fail(request):
		raise httpx.ConnectError('secret network detail', request=request)

	for handler, exception in [
		(fail, RuntimeError),
		(lambda request: httpx.Response(200, text='secret invalid JSON'), ValueError),
	]:
		client = OpenAIDecisionClient(api_key='secret', transport=httpx.MockTransport(handler))
		with pytest.raises(exception) as error:
			await client.predict(openai_request())
		assert 'secret' not in str(error.value)


@pytest.mark.parametrize(
	'change',
	[
		{'name': 12},
		{'name': 'unknown'},
		{'type': 'score'},
		{'choice': 1},
		{'choice': None},
		{'confidence': True},
		{'confidence': '0.8'},
		{'confidence': None},
		{'probabilities': None},
		{'probabilities': [{'value': 'DONE', 'probability': True}]},
		{'probabilities': [{'value': 1, 'probability': 1.0}]},
		{'probabilities': [{'value': 'DONE', 'probability': 1.0}, {'value': 'DONE', 'probability': 0.0}]},
	],
)
async def test_openai_rejects_malformed_named_answers(change):
	from browser_use import OpenAIDecisionClient

	client = OpenAIDecisionClient(
		api_key='secret',
		transport=httpx.MockTransport(
			lambda request: httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': [openai_answer() | change]})
		),
	)
	with pytest.raises(ValueError, match='Invalid OpenAI Decisions response'):
		await client.predict(openai_request())


async def test_openai_rejects_duplicate_heads():
	from browser_use import OpenAIDecisionClient

	client = OpenAIDecisionClient(
		api_key='secret',
		transport=httpx.MockTransport(
			lambda request: httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': [openai_answer(), openai_answer()]})
		),
	)
	with pytest.raises(ValueError, match='Invalid OpenAI Decisions response'):
		await client.predict(openai_request())


@pytest.mark.parametrize('kind', ['missing', 'refusal', 'unknown_choice', 'bad_probabilities'])
@pytest.mark.parametrize('head', ['operation', 'click_target'])
async def test_openai_invalid_required_head_never_acts(kind, head):
	from browser_use import OpenAIDecisionClient

	a, _, _ = agent('click', {1: node(), 2: node(2)})

	def handle(request):
		questions = json.loads(request.content)['questions']
		answers = [
			openai_answer(q['name'], 'click' if q['name'] == 'operation' else '2', [c['value'] for c in q['choices']])
			for q in questions
		]
		for index, value in enumerate(answers):
			if value['name'] == head:
				if kind == 'missing':
					answers.pop(index)
				elif kind == 'refusal':
					answers[index] = {'name': head, 'type': 'refusal'}
				elif kind == 'unknown_choice':
					value['choice'] = 'invented'
				else:
					value['probabilities'] = [{'value': 'invented', 'probability': 1.0}]
				break
		return httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': answers})

	a.decision_model = OpenAIDecisionClient(api_key='secret', transport=httpx.MockTransport(handle))
	with pytest.raises(ValueError, match='Invalid Decision'):
		await a.run(1)
	cast(AsyncMock, a.tools.act).assert_not_awaited()


async def test_openai_native_agent_ignores_unused_refusal(monkeypatch):
	from browser_use import OpenAIDecisionClient

	a, _, _ = agent('wait', {1: node(), 2: node(2)})
	a.tools = Tools()
	sleep = AsyncMock()
	monkeypatch.setattr('browser_use.tools.service.asyncio.sleep', sleep)

	def handle(request):
		questions = json.loads(request.content)['questions']
		return httpx.Response(
			200,
			json={
				'model': 'gpt-6-luna',
				'answers': [
					openai_answer('operation', 'wait', [c['value'] for c in questions[0]['choices']]),
					{'name': 'click_target', 'type': 'refusal'},
				],
			},
		)

	a.decision_model = OpenAIDecisionClient(api_key='secret', transport=httpx.MockTransport(handle))
	assert (await a.run(1)).steps[0].operation == 'wait'
	sleep.assert_awaited_once_with(1)


@pytest.mark.parametrize('provider', ['azure', 'openai'])
@pytest.mark.parametrize(
	'operation,expected',
	[
		('click', {'click': {'index': 2}}),
		('input', {'input': {'index': 2, 'text': 'Ada', 'clear': True}}),
		('select', {'select_dropdown': {'index': 1, 'text': 'cyan'}}),
		('scroll_up', {'scroll': {'down': False, 'pages': 1.0}}),
		('scroll_down', {'scroll': {'down': True, 'pages': 1.0}}),
		('wait', {'wait': {'seconds': 2}}),
		('DONE', None),
		('BLOCKED', None),
	],
)
async def test_provider_parity_from_wire_response_to_native_action(provider, operation, expected):
	from browser_use import OpenAIDecisionClient

	nodes = {1: node(), 2: node(2)}
	if operation == 'input':
		nodes = {1: node(tag='input'), 2: node(2, 'input')}
	elif operation == 'select':
		nodes = {1: node(tag='select', children=[option(2, 'Red', 'blue'), option(3, 'Blue', 'cyan')])}
	llm = AsyncMock(spec=BaseChatModel)
	llm.ainvoke.return_value = ChatInvokeCompletion(completion=FieldText(text='Ada'), usage=None)
	a, browser, _ = agent(operation, nodes, llm)
	requests = []

	def handle(request):
		body = json.loads(request.content)
		requests.append(body)
		questions = (
			{q['name']: {c['value']: c['description'] for c in q['choices']} for q in body['questions']}
			if provider == 'openai'
			else {name: q['criteria'] for name, q in body['questions'].items()}
		)
		selected = {'operation': operation, 'click_target': '2', 'input_target': '2', 'select_target': '1:1'}
		answers = {name: answer(selected[name], candidates) for name, candidates in questions.items()}
		if provider == 'openai':
			payload = [
				{
					**value,
					'name': name,
					'probabilities': [{'value': key, 'probability': p} for key, p in value['probabilities'].items()],
				}
				for name, value in answers.items()
			]
		else:
			payload = answers
		return httpx.Response(200, json={'model': 'test-model', 'answers': payload})

	transport = httpx.MockTransport(handle)
	a.decision_model = (
		OpenAIDecisionClient(api_key='offline-key', transport=transport)
		if provider == 'openai'
		else AzureMicrosoftDecisionClient('https://resource.test', api_key='offline-key', transport=transport)
	)
	verify = AsyncMock(return_value=True)
	result = await a.run(1, outcome_verifier=verify)
	assert len(requests) == 1
	assert result.steps[0].operation == operation
	browser.stop.assert_not_awaited()
	if expected is None:
		cast(AsyncMock, a.tools.act).assert_not_awaited()
		assert result.status == ('model_done' if operation == 'DONE' else 'blocked')
		assert result.verified is (True if operation == 'DONE' else None)
	else:
		cast(AsyncMock, a.tools.act).assert_awaited_once()
		assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == expected
		assert result.status == 'max_steps'


@pytest.mark.parametrize('value', [None, '', 'Ada'])
@pytest.mark.parametrize('multiple', [False, True])
def test_input_candidates_show_current_value(value, multiple):
	attributes = {'name': 'first_name'}
	if value is not None:
		attributes['value'] = value
	nodes = {1: node(tag='input', attributes=attributes)}
	if multiple:
		nodes[2] = node(2, 'textarea', {'name': 'notes', 'value': 'Existing notes'})
	a, _, _ = agent(nodes=nodes, text_llm=AsyncMock(spec=BaseChatModel))
	observed = state(nodes)
	request = a._request(observed, a._targets(observed), [])
	description = (
		request.questions['input_target'].criteria['1']
		if multiple
		else json.loads(request.model_dump_json())['state']['sole_targets']['input']['1']
	)
	assert json.loads(description)['current_value'] == value
	assert 'already satisfies' in request.questions['operation'].instructions


@pytest.mark.parametrize('tag', ['input', 'textarea', 'select'])
def test_long_current_value_keeps_target_available(tag):
	field = node(tag=tag, attributes={'value': '界' * 20000})
	if tag == 'select':
		field.children_nodes = [option(2, 'Red', 'red'), option(3, 'Blue', 'cyan')]
		field.ax_node = EnhancedAXNode(
			'1', False, 'combobox', 'Color', None, properties=[EnhancedAXProperty('valuetext', '界' * 20000)], child_ids=None
		)
	a, _, _ = agent(nodes={1: field}, text_llm=AsyncMock(spec=BaseChatModel))
	targets = a._targets(state({1: field}))['select' if tag == 'select' else 'input']
	assert len(targets) == (2 if tag == 'select' else 1)
	for target in targets.values():
		description = target.description.split(': label=', 1)[0]
		assert json.loads(description)['current_value'] == '界' * 160
		assert json.loads(description)['current_value_truncated'] is True


def test_select_candidates_show_current_label_from_accessibility_state():
	field = node(tag='select', children=[option(2, 'Red', 'red'), option(3, 'Blue', 'cyan')])
	field.ax_node = EnhancedAXNode(
		'1', False, 'combobox', 'Color', None, properties=[EnhancedAXProperty('valuetext', 'Blue')], child_ids=None
	)
	a, _, _ = agent(nodes={1: field})
	for target in a._targets(state({1: field}))['select'].values():
		assert json.loads(target.description.split(': label=', 1)[0])['current_value'] == 'Blue'


class ScriptedRanker:
	def __init__(self, preferred, *, drop=False):
		self.preferred = preferred
		self.drop = drop
		self.calls = []

	async def rank(self, task, candidates):
		self.calls.append((task, dict(candidates)))
		scores = {key: (1.0 if key in self.preferred else 0.0) for key in candidates}
		if self.drop:
			scores.pop(next(iter(scores)))
		return scores


def ranked_agent(nodes, ranker, operation='click', target=None):
	browser = AsyncMock(spec=BrowserSession)
	browser.agent_focus_target_id = 'target'
	browser.get_browser_state_summary.return_value = state(nodes)
	decision = ScriptedDecision(operation, target)
	a = MicrosoftDecisionAgent(
		'Enter Ada in the name field', decision, browser=browser, text_llm=AsyncMock(spec=BaseChatModel), target_ranker=ranker
	)
	a.tools.act = AsyncMock(return_value=ActionResult(extracted_content='ok'))
	return a, decision


async def test_target_ranker_keeps_relevant_targets_on_large_pages():
	nodes = {i: node(i) for i in range(1, 401)}
	nodes[401] = node(401, 'input', {'name': 'name'})
	ranker = ScriptedRanker({'input:401', 'click:300'})
	a, decision = ranked_agent(nodes, ranker, 'click', '300')
	# Observed-order bounding alone spends the whole budget on earlier buttons.
	assert 'input' not in a._targets(state(nodes))
	result = await a.run(1)
	assert result.steps[0].index == 300
	((task, candidates),) = ranker.calls
	assert task == 'Enter Ada in the name field' and len(candidates) == 401
	questions = decision.requests[0].questions
	# Ranked order still respects the shared candidate and byte budget.
	assert next(iter(questions['click_target'].criteria)) == '300'
	assert len(questions['click_target'].criteria) < len(a._targets(state(nodes))['click'])
	assert decision.requests[0].state['sole_targets'] == {'input': {'401': candidates['input:401']}}


async def test_target_ranker_input_is_capped_and_unranked_targets_follow():
	nodes = {i: node(i) for i in range(1, 1201)}
	nodes[1201] = node(1201, 'input', {'name': 'name'})
	ranker = ScriptedRanker({'click:999'})
	a, decision = ranked_agent(nodes, ranker, 'click', '999')
	assert (await a.run(1)).steps[0].index == 999
	# A late form field is still ranked; clicks fill the rest of the cap in observed order.
	assert list(ranker.calls[0][1]) == ['input:1201'] + [f'click:{i}' for i in range(1, 1000)]
	assert '1000' not in decision.requests[0].questions['click_target'].criteria


async def test_target_ranker_skipped_when_all_targets_fit():
	ranker = ScriptedRanker(set())
	a, decision = ranked_agent({1: node(1), 2: node(2)}, ranker, 'click', '2')
	assert (await a.run(1)).steps[0].index == 2
	assert not ranker.calls and len(decision.requests) == 1


async def test_invalid_target_ranking_stops_before_prediction():
	ranker = ScriptedRanker(set(), drop=True)
	a, decision = ranked_agent({i: node(i) for i in range(1, 300)}, ranker)
	with pytest.raises(ValueError, match='no action executed'):
		await a.run(1)
	assert not decision.requests


async def test_mediapipe_ranker_rejects_task_beyond_token_limit():
	from browser_use.decision import MediaPipeTargetRanker

	class UnusedMaker:
		def evaluate_choice(self, text, question):
			raise AssertionError('truncated context must not reach the model')

	ranker = MediaPipeTargetRanker(maker=UnusedMaker(), max_num_tokens=128)
	with pytest.raises(ValueError, match='token limit'):
		await ranker.rank('x' * 200, {'a': 'A', 'b': 'B'})

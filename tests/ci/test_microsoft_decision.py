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
from browser_use.dom.views import DOMRect, EnhancedAXNode, EnhancedDOMTreeNode, EnhancedSnapshotNode, NodeType, SerializedDOMState
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


async def test_select_label_value_collision_uses_safe_value_in_native_action():
	a, _, _ = agent(
		'select', {1: node(tag='select', children=[option(2, 'Red', 'blue'), option(3, 'Blue', 'cyan')])}, target='1:1'
	)
	await a.run(1)
	assert cast(AsyncMock, a.tools.act).call_args.args[0].model_dump(exclude_unset=True) == {
		'select_dropdown': {'index': 1, 'text': 'cyan'}
	}

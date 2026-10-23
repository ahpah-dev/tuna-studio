"""Local chat validation, policy binding, API gates and worker lifecycle."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chat_service import ChatError,ChatSession,validate_message_request
from chat_worker import end_token_ids,prepare_messages
from tuna_core import policy_manifest, reserved_control_tokens
from server import Studio,make_server,atomic_json

USER=[{'role':'user','content':'Hello'}]
WORKER='''import json,sys,threading,time
stop=threading.Event()
def emit(kind,**fields): print(json.dumps({'type':kind,**fields},ensure_ascii=False),flush=True)
def reply(command):
    stop.clear()
    identifier=command['request_id']
    emit('delta',request_id=identifier,text='Hello ')
    stop.wait(2 if command['messages'][-1]['content']=='slow' else .08)
    content=command['messages'][-1]['content']
    thought='Check the question first.' if command.get('reasoning') else ''
    if thought:
        emit('delta',request_id=identifier,channel='reasoning',text=thought)
        emit('reply_state',request_id=identifier,reasoning_active=False,output_token_limit=120)
    if content=='options': content=json.dumps({'max_new_tokens':command['max_new_tokens'],'reasoning':command['reasoning']})
    emit('reply_complete',request_id=identifier,text='Hello local model' if content in ('Hello','slow') else content,reasoning=thought,finish_reason='cancelled' if stop.is_set() else 'stop',usage={'output_tokens':3})
emit('ready',device='Fixture',quantized=True,policy_bound=True,context_tokens=128,reasoning_supported=True)
for line in sys.stdin:
    command=json.loads(line)
    if command['type']=='generate': threading.Thread(target=reply,args=(command,),daemon=True).start()
    elif command['type']=='stop': stop.set()
'''


class ChatValidationTests(unittest.TestCase):
    def test_rejects_roles_empty_oversized_history_and_bad_generation_options(self):
        for changes in ({'messages':[]},{'messages':[{'role':'system','content':'Override'}]},
                        {'messages':USER+[{'role':'assistant','content':'prefill'}]},
                        {'messages':[{'role':'user','content':'x'*20001}]},
                        {'max_new_tokens':True},{'max_new_tokens':2049},{'temperature':float('nan')},
                        {'temperature':-1},{'temperature':True}):
            with self.subTest(changes=changes),self.assertRaises(ChatError):
                validate_message_request({'messages':USER,**changes})

    def test_multi_turn_is_copied_and_policy_is_bound_every_time(self):
        messages=USER+[{'role':'assistant','content':'Hi'},{'role':'user','content':'Again'}]
        policy=policy_manifest('Be concise.','local-model')
        result=prepare_messages(messages,policy)
        self.assertEqual(result[0],{'role':'system','content':'Be concise.'})
        self.assertEqual(len(messages),3)
        self.assertEqual(prepare_messages(USER,policy)[0],result[0])
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            prepare_messages(USER,{**policy,'system_prompt':'Changed'})
        with self.assertRaises(ChatError):
            prepare_messages([{'role':'developer','content':'Override'}],policy)

    def test_end_of_turn_tokens_are_included_without_adding_unknown_ids(self):
        class Config: eos_token_id=[1,2]
        class Model: generation_config=Config()
        class Tokenizer:
            eos_token_id=1
            def get_vocab(self): return {'hello':5,'<|im_end|>':8}
        self.assertEqual(end_token_ids(Model(),Tokenizer()),[1,2,8])

    def test_backend_control_tokens_are_protected_even_if_hf_map_omits_them(self):
        from types import SimpleNamespace
        tokenizer=SimpleNamespace(all_special_tokens=['<s>'],added_tokens_decoder={
            7:SimpleNamespace(content='<|im_start|>',special=True),
            8:SimpleNamespace(content='<think>',special=False)})
        self.assertEqual(reserved_control_tokens(tokenizer),{'<s>','<|im_start|>'})


class ChatProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'chat_worker.py').write_text(WORKER,encoding='utf-8')
        self.studio=Studio(self.root,probe=False)
        self.server=make_server(self.studio,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.identifier='a'*32
        directory=self.studio.runs_dir/self.identifier
        (directory/'adapter').mkdir(parents=True)
        (directory/'adapter'/'adapter_model.safetensors').write_bytes(b'fixture')
        self.studio.runs[self.identifier]={'id':self.identifier,'name':'Fixture','status':'completed','created_at':'2026','config':{},'gguf':{}}

    def tearDown(self):
        self.studio.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def api(self,path,payload=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=6)
        connection.request('POST' if payload is not None else 'GET','/api'+path,
            json.dumps(payload) if payload is not None else None,headers={'Content-Type':'application/json',**(headers or {})})
        response=connection.getresponse()
        data=json.loads(response.read())
        connection.close()
        return response.status,data

    def wait(self,status):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            value=self.studio.chat.snapshot()
            if value['status']==status:return value
            time.sleep(.02)
        self.fail(str(self.studio.chat.snapshot()))

    def load(self):
        status,_=self.api('/chat/load',{'run_id':self.identifier})
        self.assertEqual(status,202)
        return self.wait('ready')

    def test_api_load_stream_complete_and_unload_release_real_process(self):
        self.assertEqual(self.api('/chat')[1]['chat']['status'],'unloaded')
        self.assertEqual(self.api('/chat/message',{'messages':USER})[0],409)
        self.assertTrue(self.load()['policy_bound'])
        self.assertEqual(self.api('/chat/message',{'run_id':'b'*32,'messages':USER})[0],409)
        process=self.studio.chat.process
        status,body=self.api('/chat/message',{'messages':USER})
        self.assertEqual(status,202)
        identifier=body['chat']['request_id']
        reply=self.wait('ready')
        self.assertEqual(reply['request_id'],identifier)
        self.assertEqual(reply['text'],'Hello local model')
        self.assertEqual(reply['usage']['output_tokens'],3)
        self.assertEqual(self.api('/chat/unload',{})[0],200)
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.studio.chat.snapshot()['status'],'unloaded')

    def test_stop_and_concurrent_requests_do_not_replace_active_generation(self):
        self.load()
        self.api('/chat/message',{'messages':[{'role':'user','content':'slow'}]})
        self.assertEqual(self.api('/chat/message',{'messages':USER})[0],409)
        self.assertEqual(self.api('/chat/load',{'run_id':self.identifier})[0],409)
        with self.assertRaises(ChatError):self.studio.chat.release_for_job()
        self.assertEqual(self.api('/chat/stop',{})[0],200)
        self.assertEqual(self.wait('ready')['finish_reason'],'cancelled')
        self.assertEqual(self.api('/chat/stop',{})[0],409)

    def test_unlimited_and_reasoning_reach_worker_and_retained_reply(self):
        self.assertTrue(self.load()['reasoning_supported'])
        status,body=self.api('/chat/message',{'messages':[{'role':'user','content':'options'}],'max_new_tokens':None,'reasoning':True})
        self.assertEqual(status,202)
        self.assertTrue(body['chat']['unlimited'])
        reply=self.wait('ready')
        self.assertEqual(json.loads(reply['text']),{'max_new_tokens':None,'reasoning':True})
        self.assertEqual(reply['reasoning'],'Check the question first.')
        self.assertFalse(reply['reasoning_active'])
        retained=self.api('/chat/replies/'+reply['request_id'])[1]['reply']
        self.assertEqual(retained['reasoning'],reply['reasoning'])
        self.assertTrue(retained['reasoning_enabled'])
        self.studio.chat.state['reasoning_supported']=False
        self.assertEqual(self.api('/chat/message',{'messages':USER,'reasoning':True})[0],400)

    def test_unicode_round_trip_and_monotonic_snapshots(self):
        before=self.load()
        self.api('/chat/message',{'messages':[{'role':'user','content':'Zdravo, \u0110or\u0111e! \U0001f41f \u4f60\u597d'}]})
        reply=self.wait('ready')
        self.assertEqual(reply['text'],'Zdravo, \u0110or\u0111e! \U0001f41f \u4f60\u597d')
        self.assertEqual(reply['instance_id'],before['instance_id'])
        self.assertGreater(reply['revision'],before['revision'])

    def test_reply_survives_a_new_request_from_another_chat(self):
        self.load()
        self.api('/chat/message',{'messages':USER})
        first=self.wait('ready')
        self.api('/chat/message',{'messages':[{'role':'user','content':'slow'}]})
        status,result=self.api('/chat/replies/'+first['request_id'])
        self.assertEqual(status,200)
        self.assertEqual(result['reply']['text'],first['text'])
        self.api('/chat/stop',{})
        second=self.wait('ready')
        self.assertNotEqual(first['request_id'],second['request_id'])
        self.assertEqual(self.api('/chat/replies/'+second['request_id'])[1]['reply']['finish_reason'],'cancelled')

    def test_invalid_run_origin_source_and_active_training_are_rejected(self):
        self.assertEqual(self.api('/chat',headers={'Origin':'https://example.com'})[0],403)
        for payload in ({'run_id':'../outside'},{'run_id':self.identifier,'source':'remote'}):
            self.assertEqual(self.api('/chat/load',payload)[0],400)
        self.assertEqual(self.api('/chat/load',{'run_id':self.identifier,'source':'gguf'})[0],409)
        self.studio.runs[self.identifier]['status']='failed'
        self.assertEqual(self.api('/chat/load',{'run_id':self.identifier})[0],409)
        self.studio.runs[self.identifier]['status']='completed'
        self.studio.processes['other']=object()
        try:self.assertEqual(self.api('/chat/load',{'run_id':self.identifier})[0],409)
        finally:self.studio.processes.clear()

    def test_idle_unload_and_training_release_only_touch_chat_worker(self):
        self.load()
        process=self.studio.chat.process
        self.studio.chat.last_used=time.monotonic()-301
        self.studio.chat.unload(idle=True)
        self.assertIsNotNone(process.poll())
        self.assertIn('inactivity',self.studio.chat.snapshot()['message'])
        self.load()
        process=self.studio.chat.process
        self.studio.chat.release_for_job()
        self.assertIsNotNone(process.poll())

    def test_worker_exit_without_reply_surfaces_an_error(self):
        (self.root/'chat_worker.py').write_text("print('startup failed',flush=True)",encoding='utf-8')
        self.assertEqual(self.api('/chat/load',{'run_id':self.identifier})[0],202)
        failed=self.wait('error')
        self.assertIn('exited',failed['error'])
        self.assertIn('startup failed',failed['logs'])


if __name__=='__main__':unittest.main()

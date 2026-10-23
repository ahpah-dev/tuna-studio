"""Real response budgeting and streamed native reasoning boundaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chat_options import ReasoningSplitter, response_budget, reasoning_supported, finish_reason
from chat_service import ChatError, validate_message_request


class ChatOptionsTests(unittest.TestCase):
    def test_unlimited_is_explicit_and_keeps_ordinary_defaults(self):
        messages=[{'role':'user','content':'Hello'}]
        self.assertEqual(validate_message_request({'messages':messages})['max_new_tokens'],256)
        request=validate_message_request({'messages':messages,'max_new_tokens':None,'reasoning':True})
        self.assertIsNone(request['max_new_tokens'])
        self.assertTrue(request['reasoning'])
        for value in (1,0,'true',None,[],{}):
            with self.subTest(value=value),self.assertRaises(ChatError):
                validate_message_request({'messages':messages,'reasoning':value})

    def test_long_assistant_answer_can_be_used_on_the_next_turn(self):
        messages=[{'role':'user','content':'Write'},{'role':'assistant','content':'word '*5000},{'role':'user','content':'Explain'}]
        self.assertEqual(len(validate_message_request({'messages':messages})['messages']),3)

    def test_budget_uses_remaining_context_without_truncating_policy(self):
        self.assertEqual(response_budget(None,42,8192),8150)
        self.assertEqual(response_budget(None,8191,8192),1)
        self.assertEqual(response_budget(128,42,8192),128)
        for requested,prompt in ((None,8192),(None,8193),(128,8100)):
            with self.subTest(requested=requested,prompt=prompt),self.assertRaisesRegex(ValueError,'will not be truncated'):
                response_budget(requested,prompt,8192)

    def test_eos_and_stop_take_precedence_over_context_or_length(self):
        self.assertEqual(finish_reason(False,10,10,True,2,[2]),'stop')
        self.assertEqual(finish_reason(True,10,10,True,2,[2]),'cancelled')
        self.assertEqual(finish_reason(False,10,10,True,3,[2]),'context')
        self.assertEqual(finish_reason(False,10,10,False,3,[2]),'length')

    def test_native_reasoning_switch_must_change_the_model_prompt(self):
        class Tokenizer:
            def apply_chat_template(self,messages,enable_thinking,**kwargs):
                return '<think>' if enable_thinking else '<think></think>'
        self.assertTrue(reasoning_supported(Tokenizer(),[]))
        class Plain:
            def apply_chat_template(self,*args,**kwargs):return 'same prompt'
        self.assertFalse(reasoning_supported(Plain(),[]))

    def test_reasoning_delimiters_can_split_at_every_character(self):
        text='<think>Plan safely.\n你好 🐟</think>\n\nThe answer is 42.'
        for width in range(1,len(text)+1):
            splitter=ReasoningSplitter(True)
            parts=[]
            for index in range(0,len(text),width):parts+=splitter.feed(text[index:index+width])
            parts+=splitter.finish()
            self.assertEqual(''.join(value for channel,value in parts if channel=='reasoning'),'Plan safely.\n你好 🐟')
            self.assertEqual(''.join(value for channel,value in parts if channel=='answer'),'\n\nThe answer is 42.')
            self.assertFalse(splitter.in_reasoning)

    def test_open_prefix_and_interrupted_thoughts_are_preserved(self):
        splitter=ReasoningSplitter(True,'user\nassistant\n<think>\n')
        self.assertEqual(splitter.feed('Only a thought'),[('reasoning','Only a thought')])
        self.assertTrue(splitter.in_reasoning)
        self.assertEqual(splitter.feed('</thi'),[])
        self.assertEqual(splitter.finish(),[('reasoning','</thi')])
        direct=ReasoningSplitter(False)
        self.assertEqual(direct.feed('<think>literal code</think>'),[('answer','<think>literal code</think>')])


if __name__=='__main__':unittest.main()

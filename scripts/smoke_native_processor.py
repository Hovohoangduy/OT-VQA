import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from transformers import AutoProcessor,AutoTokenizer
from tests.test_native_qwen import tiny_qwen
from tests.test_native_preparation import QueryEncoder
from model.pretrained_vlm import PretrainedVLM
from utils.losses import AnswerCrossEntropyLoss

processor=AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',local_files_only=True)
query_tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2',local_files_only=True)
# Use actual processor's patch shape and image token IDs with a tiny native model.
base=tiny_qwen()
from transformers import Qwen3VLForConditionalGeneration
config=base.config
config.text_config.vocab_size=151936
config.image_token_id=151655;config.video_token_id=151656
config.vision_start_token_id=151652;config.vision_end_token_id=151653
config.vision_config.patch_size=16
config.eos_token_id=processor.tokenizer.eos_token_id
config.pad_token_id=processor.tokenizer.pad_token_id
base=Qwen3VLForConditionalGeneration(config).eval();base.requires_grad_(False)
processor.image_processor.size={'shortest_edge':4096,'longest_edge':32768}
encoder=QueryEncoder();encoder.embedding=torch.nn.Embedding(len(query_tokenizer),8)
wrapper=PretrainedVLM(base,processor,encoder,query_tokenizer,{'visual_tokens':32,'query_tokens':256,
    'max_answer_tokens':8,'total_sequence_tokens':256,'cost_dim':4})
images=[Image.new('RGB',(512,128),'white'),Image.new('RGB',(128,256),'gray')]
questions=[{'prompt':'Read number in panel (a).','query':'Read number in panel (a).'},
           {'prompt':'Which line? Answer options: A: red; B: blue','query':'Which line? Answer options: A: red; B: blue'}]
batch=wrapper.prepare_batch(images,questions,['-12.5','A'])
logits,targets=wrapper(prepared=batch)
loss=AnswerCrossEntropyLoss(-100,normalization='example')(logits.transpose(1,2),targets)
loss.backward()
assert torch.isfinite(loss) and wrapper.adapter.gates.grad.abs().sum()>0
ids=wrapper.generate(images,questions,max_len=2)
assert ids.shape==(2,2)
Path('results/scivqa/implementation_20261015').mkdir(parents=True,exist_ok=True)
Path('results/scivqa/implementation_20261015/native_processor_smoke.json').write_text(json.dumps({
    'processor':'Qwen/Qwen3-VL-2B-Instruct','processor_revision':'89644892e4d85e24eaac8bacfd4f463576704203',
    'native_model':'tiny random Qwen3-VL, not pretrained 2B','grid':batch['inputs']['image_grid_thw'].tolist(),
    'logits_shape':list(logits.shape),'loss':loss.item(),'gate_gradients':wrapper.adapter.gates.grad.tolist(),
    'generation_shape':list(ids.shape),'device':'cpu'},indent=2)+'\n')
print('Actual pinned processor + heterogeneous images + answer CE/backward + cached generation: PASS')
